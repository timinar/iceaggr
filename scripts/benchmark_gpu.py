#!/usr/bin/env python3
"""
Benchmark GPU throughput for a given config (forward + backward pass).

Run:
    uv run python scripts/benchmark_gpu.py --config configs/train_flat_v2_none_K84_5M_finetune_hi.yaml
"""

import argparse
import time

import torch
import yaml

from iceaggr.data import IceCubeDataset, GeometryLoader, make_collate_flat, BatchAwareSampler
from iceaggr.models import FlatTransformerV2
from iceaggr.models.losses import angular_distance_loss
from iceaggr.utils import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)

N_WARMUP = 3
N_BENCH = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Model
    model_config = {
        "max_pulses_per_dom": config["model"]["max_pulses_per_dom"],
        "d_model": config["model"]["d_model"],
        "max_doms": config["model"]["max_doms"],
        "num_heads": config["model"]["num_heads"],
        "num_layers": config["model"]["num_layers"],
        "hidden_dim": config["model"]["hidden_dim"],
        "head_hidden_dim": config["model"]["head_hidden_dim"],
        "dropout": config["model"]["dropout"],
        "input_mode": config["model"]["input_mode"],
    }
    model = FlatTransformerV2(model_config).to(device)
    model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # Data (small slice for benchmarking)
    geometry = GeometryLoader(config["data"]["geometry_path"])
    dataset = IceCubeDataset(
        split="train",
        batch_range=(1, 5),
        min_pulses=config["data"].get("min_pulses"),
        cache_size=1,
    )
    logger.info(f"Dataset: {len(dataset):,} events")

    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=config["model"]["max_pulses_per_dom"],
        max_doms=config["model"]["max_doms"],
    )
    sampler = BatchAwareSampler(dataset.metadata)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))
    scaler = torch.amp.GradScaler(enabled=config["training"]["use_amp"])

    model.train()
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    # Warmup
    logger.info(f"Warming up ({N_WARMUP} batches) …")
    for _ in range(N_WARMUP):
        batch = next_batch()
        dom_vectors = batch["dom_vectors"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config["training"]["use_amp"]):
            y_pred = model(dom_vectors, padding_mask)
            loss = angular_distance_loss(y_pred, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    logger.info(f"Benchmarking ({N_BENCH} batches) …")

    times = []
    events_per_batch = []

    for _ in range(N_BENCH):
        batch = next_batch()
        dom_vectors = batch["dom_vectors"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)

        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config["training"]["use_amp"]):
            y_pred = model(dom_vectors, padding_mask)
            loss = angular_distance_loss(y_pred, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        times.append(time.perf_counter() - t0)
        events_per_batch.append(dom_vectors.shape[0])

    avg_ms = 1000 * sum(times) / len(times)
    avg_events = sum(events_per_batch) / len(events_per_batch)
    events_per_sec = avg_events / (sum(times) / len(times))

    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
    mem_reserved = torch.cuda.memory_reserved(device) / 1e9

    print(f"\n=== GPU Benchmark: max_doms={config['model']['max_doms']}, "
          f"batch_size={config['training']['batch_size']} ===")
    print(f"  Avg batch shape : ({avg_events:.0f}, {config['model']['max_doms']}, {config['model']['d_model']})")
    print(f"  ms / batch      : {avg_ms:.1f}")
    print(f"  events / sec    : {events_per_sec:.0f}")
    print(f"  GPU mem alloc   : {mem_alloc:.2f} GB")
    print(f"  GPU mem reserved: {mem_reserved:.2f} GB")


if __name__ == "__main__":
    main()

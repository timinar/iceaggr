#!/usr/bin/env python3
"""
Profile dataloader throughput and memory for different max_doms settings.

Run:
    uv run python scripts/profile_dataloader.py

Tests combinations of max_doms × batch_size that are relevant for
high-activity fine-tuning (min_pulses=1000).
"""

import time
from pathlib import Path

import torch
import yaml

from iceaggr.data import IceCubeDataset, GeometryLoader, make_collate_flat, BatchAwareSampler
from iceaggr.utils import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)

N_WARMUP = 2
N_PROFILE = 10
MIN_PULSES = 1000
GEOMETRY_PATH = "/groups/pheno/inar/icecube_kaggle/sensor_geometry_normalized.csv"


def profile(max_doms: int, batch_size: int, max_pulses_per_dom: int = 84):
    dataset = IceCubeDataset(
        split="train",
        batch_range=(1, 5),       # use 5 batch files → fast load
        min_pulses=MIN_PULSES,
        cache_size=1,
    )
    logger.info(f"  Dataset: {len(dataset):,} events")

    geometry = GeometryLoader(GEOMETRY_PATH)
    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=max_pulses_per_dom,
        max_doms=max_doms,
    )
    sampler = BatchAwareSampler(dataset.metadata)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # Warmup
    it = iter(loader)
    for _ in range(N_WARMUP):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

    # Profile
    times = []
    dom_shape = None
    for _ in range(N_PROFILE):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        times.append(time.perf_counter() - t0)
        dom_shape = batch["dom_vectors"].shape

    mean_ms = 1000 * sum(times) / len(times)
    events_per_sec = batch_size / (sum(times) / len(times))

    # Memory estimate: dom_vectors float32
    mem_mb = dom_shape[0] * dom_shape[1] * dom_shape[2] * 4 / 1e6

    return {
        "max_doms": max_doms,
        "batch_size": batch_size,
        "dom_shape": dom_shape,
        "batch_mem_mb": mem_mb,
        "mean_ms": mean_ms,
        "events_per_sec": events_per_sec,
    }


def main():
    # Baseline: what the model currently uses (no min_pulses filter)
    configs = [
        # (max_doms, batch_size)
        (128,  4096),   # current baseline
        (256,  2048),
        (512,  512),
        (512,  256),
        (1024, 128),
        (1024, 64),
    ]

    print(f"\n{'max_doms':>8}  {'batch_size':>10}  {'dom_shape':>20}  "
          f"{'batch_MB':>8}  {'ms/batch':>9}  {'events/s':>9}")
    print("-" * 80)

    for max_doms, batch_size in configs:
        try:
            r = profile(max_doms, batch_size)
            print(f"{r['max_doms']:>8}  {r['batch_size']:>10}  "
                  f"{str(r['dom_shape']):>20}  "
                  f"{r['batch_mem_mb']:>8.1f}  "
                  f"{r['mean_ms']:>9.1f}  "
                  f"{r['events_per_sec']:>9.0f}")
        except Exception as e:
            print(f"{max_doms:>8}  {batch_size:>10}  ERROR: {e}")


if __name__ == "__main__":
    main()

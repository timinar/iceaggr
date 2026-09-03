#!/usr/bin/env python3
"""
Diagnose the training dataloader bottleneck for FlatTransformerV2.

Training runs are GPU-starved: the production pretraining config sustains only
~4.8 batches/s (bs=4096 -> ~19.6k events/s) while the model forward alone can do
~29k events/s. This script attributes the gap.

Three parts:

  A. COMPUTE CEILING  — forward-only and forward+backward events/s at the
                        pretraining shape (bs=4096, max_doms=128), pre-collated
                        on GPU. If fwd+bwd >> observed step rate, training is
                        dataloader-bound.

  B. DATALOADER SWEEP — dataloader-only events/s (parquet -> collate -> pinned
                        H2D copy, NO model) over the same shape, sweeping
                        num_workers in {2,4,8,16} x torch-threads-per-worker in
                        {default, 1}. Isolates worker scaling and CPU
                        oversubscription (this node: 64 CPUs, torch default
                        num_threads=32, so 8 workers x 32 = 256 threads).

  C. COLLATE BREAKDOWN — cProfile of the collator at num_workers=0 to attribute
                        CPU time inside collate_fn (torch.unique / argsort /
                        topk / scatter / geometry).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_dataloader.py
"""

import argparse
import cProfile
import io
import pstats
import time

import torch
from torch.utils.data import DataLoader, Subset

from iceaggr.data import IceCubeDataset, GeometryLoader, make_collate_flat, BatchAwareSampler
from iceaggr.models import FlatTransformerV2
from iceaggr.models.losses import angular_distance_loss
from iceaggr.utils import get_logger

logger = get_logger(__name__)

GEOMETRY_PATH = "/groups/pheno/inar/icecube_kaggle/sensor_geometry_normalized.csv"

# Production pretraining shape.
MODEL_CONFIG = {
    "max_pulses_per_dom": 84,
    "d_model": 256,
    "max_doms": 128,
    "num_heads": 8,
    "num_layers": 6,
    "hidden_dim": 1024,
    "head_hidden_dim": 1024,
    "dropout": 0.0,
    "input_mode": "none",
}
BATCH_SIZE = 4096
BATCH_RANGE = (651, 652)


def _limit_threads_worker_init(_worker_id):
    """DataLoader worker_init_fn: pin each worker to a single torch intra-op
    thread so N workers don't collectively oversubscribe the CPU."""
    torch.set_num_threads(1)


def compute_ceiling(device):
    """Forward-only and forward+backward events/s at bs=4096, max_doms=128."""
    model = FlatTransformerV2(dict(MODEL_CONFIG)).to(device)
    dom = torch.randn(BATCH_SIZE, MODEL_CONFIG["max_doms"],
                      4 + 3 * MODEL_CONFIG["max_pulses_per_dom"], device=device)
    mask = torch.ones(BATCH_SIZE, MODEL_CONFIG["max_doms"], dtype=torch.bool, device=device)
    targets = torch.randn(BATCH_SIZE, 2, device=device)

    # forward-only
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5):
            model(dom, mask)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            model(dom, mask)
        torch.cuda.synchronize()
        fwd_eps = BATCH_SIZE * 30 / (time.perf_counter() - t0)

    # forward + backward + step (real training step)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(5):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(dom, mask)
            loss = angular_distance_loss(out["direction"], targets)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(dom, mask)
            loss = angular_distance_loss(out["direction"], targets)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    fwdbwd_eps = BATCH_SIZE * 30 / (time.perf_counter() - t0)

    logger.info(f"  forward-only     : {fwd_eps:>10,.0f} ev/s ({BATCH_SIZE / fwd_eps * 1000:.1f} ms/batch)")
    logger.info(f"  forward+backward : {fwdbwd_eps:>10,.0f} ev/s ({BATCH_SIZE / fwdbwd_eps * 1000:.1f} ms/batch)")
    del model, opt
    torch.cuda.empty_cache()
    return fwd_eps, fwdbwd_eps


def dataloader_throughput(dataset, geometry, n_events, num_workers, limit_threads, device):
    """Events/s for parquet -> collate -> pinned H2D copy (no model)."""
    subset = Subset(dataset, list(range(min(n_events, len(dataset)))))
    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=MODEL_CONFIG["max_pulses_per_dom"],
        max_doms=MODEL_CONFIG["max_doms"],
    )
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        worker_init_fn=_limit_threads_worker_init if limit_threads else None,
    )

    def run_pass():
        seen = 0
        for batch in loader:
            dv = batch["dom_vectors"].to(device, non_blocking=True)
            _ = batch["padding_mask"].to(device, non_blocking=True)
            seen += dv.shape[0]
        torch.cuda.synchronize()
        return seen

    run_pass()  # warmup: spawn workers, prime caches
    t0 = time.perf_counter()
    seen = run_pass()
    elapsed = time.perf_counter() - t0
    del loader
    return seen / elapsed


def collate_breakdown(dataset, geometry, n_batches):
    """cProfile the collator at num_workers=0 to attribute CPU time."""
    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=MODEL_CONFIG["max_pulses_per_dom"],
        max_doms=MODEL_CONFIG["max_doms"],
    )
    # Pre-fetch raw items (dataset.__getitem__) so the profile is the collator,
    # not parquet IO. Use contiguous events (BatchAware-like: one parquet file).
    items = [dataset[i] for i in range(BATCH_SIZE * n_batches)]
    batches = [items[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(n_batches)]

    # warmup
    collate_fn(batches[0])
    pr = cProfile.Profile()
    pr.enable()
    for b in batches:
        collate_fn(b)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
    return s.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=40000)
    parser.add_argument("--geometry", default=GEOMETRY_PATH)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    import os
    logger.info(f"CPUs: {os.cpu_count()} | torch.get_num_threads()={torch.get_num_threads()}")

    geometry = GeometryLoader(args.geometry)

    logger.info("=== A. COMPUTE CEILING (bs=4096, max_doms=128, bf16) ===")
    fwd_eps, fwdbwd_eps = compute_ceiling(device)

    logger.info("=== C. COLLATE BREAKDOWN (cProfile, num_workers=0) ===")
    dataset = IceCubeDataset(split="train", batch_range=BATCH_RANGE, cache_size=2)
    prof_text = collate_breakdown(dataset, geometry, n_batches=8)
    print(prof_text)

    logger.info("=== B. DATALOADER SWEEP (parquet -> collate -> H2D, no model) ===")
    sweep = {}
    for limit_threads in (False, True):
        for nw in (2, 4, 8, 16):
            ds = IceCubeDataset(split="train", batch_range=BATCH_RANGE, cache_size=2)
            eps = dataloader_throughput(ds, geometry, args.n_events, nw, limit_threads, device)
            tag = "1thr/worker" if limit_threads else "default-threads"
            sweep[(limit_threads, nw)] = eps
            logger.info(f"  workers={nw:>2} [{tag:>15}] : {eps:>10,.0f} ev/s")
            del ds

    # summary
    print("\n" + "=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"CPUs={os.cpu_count()} torch_threads={torch.get_num_threads()}")
    print("=" * 70)
    print("A. COMPUTE CEILING (bs=4096, max_doms=128, bf16)")
    print(f"   forward-only     : {fwd_eps:>10,.0f} ev/s")
    print(f"   forward+backward : {fwdbwd_eps:>10,.0f} ev/s   <- training GPU ceiling")
    print("\nB. DATALOADER-ONLY THROUGHPUT (events/s), no model")
    print(f"   {'workers':>8} | {'default-threads':>16} | {'1 thread/worker':>16}")
    print("   " + "-" * 46)
    for nw in (2, 4, 8, 16):
        print(f"   {nw:>8} | {sweep[(False, nw)]:>16,.0f} | {sweep[(True, nw)]:>16,.0f}")
    best = max(sweep.values())
    print(f"\n   best dataloader: {best:,.0f} ev/s")
    print(f"   fwd+bwd ceiling: {fwdbwd_eps:,.0f} ev/s")
    verdict = "DATALOADER-BOUND" if best < fwdbwd_eps else "COMPUTE-BOUND"
    print(f"   => training is {verdict} "
          f"(min({best:,.0f}, {fwdbwd_eps:,.0f}) = {min(best, fwdbwd_eps):,.0f} ev/s)")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Measure real inference throughput for the production FlatTransformerV2 (N2-T2).

Two numbers are reported:

  A. MODEL-ONLY   — forward-pass throughput on pre-collated, already-on-GPU
                    tensors (no dataloader, no host->device copy in the timed
                    region). This is the pure compute number for the paper's
                    computational-mechanism section. Swept over several batch
                    sizes; the headline number is batch size 512.

  B. END-TO-END   — the full production pipeline: parquet -> flat collator
                    (tokenization) -> pinned host->device copy -> forward, over
                    ~50k val events at batch size 512 with num_workers=8. This
                    is dataloader-bound and reflects deployed throughput.

Both use eval() + bf16 autocast + torch.no_grad(), matching scripts/inference.py
(the model is NOT torch.compile'd — that matches the deployed inference path).

Usage:
    uv run python scripts/benchmark_inference.py
    uv run python scripts/benchmark_inference.py --n-events 50000 --e2e-workers 8
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from iceaggr.data import IceCubeDataset, GeometryLoader, make_collate_flat
from iceaggr.models import FlatTransformerV2
from iceaggr.utils import get_logger

logger = get_logger(__name__)

CHECKPOINT = "checkpoints/v2-none-K84-5M-proper-split-10ep/best.pt"
GEOMETRY_PATH = "/groups/pheno/inar/icecube_kaggle/sensor_geometry_normalized.csv"

# Production K=84 architecture (matches the checkpoint config).
MODEL_CONFIG = {
    "max_pulses_per_dom": 84,
    "d_model": 256,
    "max_doms": 128,
    "num_heads": 8,
    "num_layers": 6,
    "hidden_dim": 1024,
    "head_hidden_dim": 1024,
    "dropout": 0.0,  # no dropout at inference
    "input_mode": "none",
}

MODEL_ONLY_BATCH_SIZES = [1, 64, 256, 512, 1024]
N_WARMUP = 10
N_TIMED = 100


def load_model(checkpoint: str, device: str) -> FlatTransformerV2:
    """Build the production model and load weights, stripping _orig_mod. prefix."""
    model = FlatTransformerV2(dict(MODEL_CONFIG))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model ({n_params:,} params) from {checkpoint}")
    return model


def benchmark_model_only(model, dom_vectors, padding_mask, device):
    """Time forward passes on already-on-GPU tensors at several batch sizes.

    dom_vectors / padding_mask hold at least max(batch_sizes) real, collated
    events on the GPU; each batch size slices the first `bs` of them.
    """
    results = []
    for bs in MODEL_ONLY_BATCH_SIZES:
        dv = dom_vectors[:bs].contiguous()
        pm = padding_mask[:bs].contiguous()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(N_WARMUP):
                model(dv, pm)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(N_TIMED):
                model(dv, pm)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        per_fwd_s = elapsed / N_TIMED
        events_per_s = bs / per_fwd_s
        ms_per_event = 1000.0 * per_fwd_s / bs
        results.append((bs, events_per_s, ms_per_event, 1000.0 * per_fwd_s))
        logger.info(
            f"  bs={bs:>4}: {events_per_s:>10,.0f} ev/s | "
            f"{ms_per_event:.4f} ms/event | {1000.0 * per_fwd_s:.2f} ms/batch"
        )
    return results


def benchmark_end_to_end(model, geometry, n_events, batch_size, num_workers, device):
    """Time the full parquet -> collate -> GPU -> forward pipeline.

    A warmup epoch primes the OS page cache, the dataset's in-memory parquet
    cache, and the persistent workers; the timed epoch then measures
    steady-state end-to-end throughput (data resident, tokenization live).
    """
    dataset = IceCubeDataset(split="train", batch_range=(651, 652), cache_size=5)
    subset = Subset(dataset, list(range(min(n_events, len(dataset)))))
    n = len(subset)
    logger.info(f"End-to-end over {n:,} events, bs={batch_size}, workers={num_workers}")

    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=MODEL_CONFIG["max_pulses_per_dom"],
        max_doms=MODEL_CONFIG["max_doms"],
    )
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    def run_epoch():
        seen = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in loader:
                dv = batch["dom_vectors"].to(device, non_blocking=True)
                pm = batch["padding_mask"].to(device, non_blocking=True)
                model(dv, pm)
                seen += dv.shape[0]
        torch.cuda.synchronize()
        return seen

    logger.info("  warmup epoch (prime cache + workers) ...")
    run_epoch()

    logger.info("  timed epoch ...")
    t0 = time.perf_counter()
    seen = run_epoch()
    elapsed = time.perf_counter() - t0

    events_per_s = seen / elapsed
    logger.info(
        f"  end-to-end: {events_per_s:,.0f} ev/s "
        f"({seen:,} events in {elapsed:.2f}s)"
    )
    return events_per_s, seen, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--geometry", default=GEOMETRY_PATH)
    parser.add_argument("--n-events", type=int, default=50000)
    parser.add_argument("--e2e-batch-size", type=int, default=512)
    parser.add_argument("--e2e-workers", type=int, default=8)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU: {gpu_name} | torch {torch.__version__}")

    geometry = GeometryLoader(args.geometry)
    model = load_model(args.checkpoint, device)

    # Pre-collate one real mini-batch of max(batch_sizes) events onto the GPU.
    n_pool = max(MODEL_ONLY_BATCH_SIZES)
    pool_ds = IceCubeDataset(split="train", batch_range=(651, 652), cache_size=2)
    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=MODEL_CONFIG["max_pulses_per_dom"],
        max_doms=MODEL_CONFIG["max_doms"],
    )
    pool_batch = collate_fn([pool_ds[i] for i in range(n_pool)])
    dom_vectors = pool_batch["dom_vectors"].to(device)
    padding_mask = pool_batch["padding_mask"].to(device)
    logger.info(
        f"Pre-collated pool on GPU: dom_vectors {tuple(dom_vectors.shape)}, "
        f"dtype {dom_vectors.dtype}"
    )

    logger.info("=== A. MODEL-ONLY throughput (pre-collated, on-GPU) ===")
    model_only = benchmark_model_only(model, dom_vectors, padding_mask, device)

    logger.info("=== B. END-TO-END throughput (parquet -> collate -> GPU -> fwd) ===")
    e2e_eps, e2e_seen, e2e_elapsed = benchmark_end_to_end(
        model, geometry, args.n_events, args.e2e_batch_size, args.e2e_workers, device
    )

    headline = next(eps for bs, eps, _, _ in model_only if bs == 512)

    print("\n" + "=" * 68)
    print(f"GPU: {gpu_name}   |   torch {torch.__version__}")
    print(f"Model: FlatTransformerV2 K=84 d=256 6L/8H "
          f"({sum(p.numel() for p in model.parameters()):,} params), bf16, eval")
    print("=" * 68)
    print("A. MODEL-ONLY (forward only, tensors already on GPU)")
    print(f"{'batch':>8} | {'events/s':>12} | {'ms/event':>10} | {'ms/batch':>10}")
    print("-" * 48)
    for bs, eps, mspe, mspb in model_only:
        print(f"{bs:>8} | {eps:>12,.0f} | {mspe:>10.4f} | {mspb:>10.2f}")
    print("-" * 48)
    print(f"\nHEADLINE (model-only, bs=512): {headline:,.0f} events/s")
    print(
        f"\nB. END-TO-END (bs={args.e2e_batch_size}, workers={args.e2e_workers}): "
        f"{e2e_eps:,.0f} events/s  ({e2e_seen:,} events in {e2e_elapsed:.2f}s)"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()

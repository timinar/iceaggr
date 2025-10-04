#!/usr/bin/env python
"""
Test GPU utilization by overfitting a single large batch.

This verifies if I/O is the bottleneck:
- If GPU util is HIGH (>80%) when reusing same batch → I/O is the bottleneck
- If GPU util is LOW (<50%) when reusing same batch → Model/compute is the bottleneck

Usage:
    uv run python scripts/test_gpu_utilization.py
    uv run python scripts/test_gpu_utilization.py --batch-size 512 --iterations 1000
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.models import HierarchicalTransformer
from iceaggr.training import AngularDistanceLoss
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Test GPU utilization with single batch overfitting"
    )

    # Training config
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (events)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Model config (match baseline_1m)
    parser.add_argument("--d-model", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--t1-layers", type=int, default=4, help="T1 layers")
    parser.add_argument("--t2-layers", type=int, default=4, help="T2 layers")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("GPU UTILIZATION TEST")
    logger.info("="*60)
    logger.info(f"Batch size: {args.batch_size} events")
    logger.info(f"Iterations: {args.iterations}")
    logger.info("Expected: >80% GPU util if I/O is bottleneck, <50% if compute is bottleneck")
    logger.info("="*60)

    # Load single batch
    logger.info("\nLoading single batch...")
    dataset = IceCubeDataset(split="train", max_events=1000)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't care about shuffle for this test
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=args.max_seq_len),
    )

    batch = next(iter(loader))
    logger.info(f"✓ Loaded batch with {len(batch['metadata']['event_ids'])} events")
    logger.info(f"  Total DOMs: {batch['metadata']['total_doms']}")
    logger.info(f"  Packed sequences shape: {batch['packed_sequences'].shape}")

    # Create model (same as baseline_1m config)
    logger.info("\nCreating model...")
    model = HierarchicalTransformer(
        d_model=args.d_model,
        t1_n_heads=4,
        t1_n_layers=args.t1_layers,
        t1_max_seq_len=args.max_seq_len,
        t1_max_batch_size=64,
        t2_n_heads=4,
        t2_n_layers=args.t2_layers,
        t2_max_doms=2048,
        dropout=0.0,
        sensor_geometry_path=None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"✓ Model on {device}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move batch to device
    def batch_to_device(batch, device):
        device_batch = {}
        for key, value in batch.items():
            if key == "metadata":
                device_batch[key] = {}
                for meta_key, meta_value in value.items():
                    if isinstance(meta_value, torch.Tensor):
                        device_batch[key][meta_key] = meta_value.to(device)
                    else:
                        device_batch[key][meta_key] = meta_value
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value
        return device_batch

    batch = batch_to_device(batch, device)
    targets = batch["metadata"]["targets"]

    # Loss and optimizer
    loss_fn = AngularDistanceLoss(use_unit_vectors=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Warmup
    logger.info("\nWarmup (3 iterations)...")
    for _ in range(3):
        optimizer.zero_grad()
        predictions = model(batch)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    # Training loop - measure throughput
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING TRAINING - Monitor GPU utilization now!")
    logger.info(f"{'='*60}")
    logger.info(f"Run in another terminal: watch -n 0.5 nvidia-smi")
    logger.info(f"{'='*60}\n")

    losses = []
    start_time = time.time()

    for iteration in range(args.iterations):
        iter_start = time.time()

        optimizer.zero_grad()
        predictions = model(batch)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        iter_time = time.time() - iter_start
        losses.append(loss.item())

        # Log progress
        if (iteration + 1) % 50 == 0 or iteration == 0:
            elapsed = time.time() - start_time
            iters_per_sec = (iteration + 1) / elapsed
            ms_per_iter = (elapsed / (iteration + 1)) * 1000
            events_per_sec = (iteration + 1) * args.batch_size / elapsed

            logger.info(
                f"Iter {iteration + 1:4d}/{args.iterations} | "
                f"Loss: {loss.item():.6f} | "
                f"{ms_per_iter:.1f}ms/iter | "
                f"{events_per_sec:.0f} events/sec"
            )

    total_time = time.time() - start_time

    # Results
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Throughput: {args.iterations / total_time:.1f} iters/sec")
    logger.info(f"Throughput: {args.iterations * args.batch_size / total_time:.0f} events/sec")
    logger.info(f"Time per iter: {(total_time / args.iterations) * 1000:.1f}ms")
    logger.info(f"\nInitial loss: {losses[0]:.6f}")
    logger.info(f"Final loss: {losses[-1]:.6f}")
    logger.info(f"Loss reduction: {(1 - losses[-1] / losses[0]) * 100:.1f}%")

    # Interpretation
    logger.info(f"\n{'='*60}")
    logger.info("INTERPRETATION")
    logger.info(f"{'='*60}")
    logger.info("Check your nvidia-smi output:")
    logger.info("  • GPU util >80%: I/O is the bottleneck (data loading is slow)")
    logger.info("  • GPU util <50%: Model/compute is the bottleneck (need bigger model/batch)")
    logger.info("  • GPU util 50-80%: Mixed bottleneck (optimize both)")
    logger.info(f"{'='*60}")

    # Check overfitting
    if losses[-1] < losses[0] * 0.1:
        logger.info("\n✅ Model successfully overfitted (loss reduced >90%)")
    else:
        logger.warning(f"\n⚠️  Model did not fully overfit (may need more iterations)")


if __name__ == "__main__":
    main()

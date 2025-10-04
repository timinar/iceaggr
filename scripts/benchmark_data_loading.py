#!/usr/bin/env python
"""
Benchmark data loading pipeline to identify bottlenecks.

Usage:
    uv run python scripts/benchmark_data_loading.py
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def benchmark_dataset_creation():
    """Benchmark dataset initialization."""
    logger.info("=" * 60)
    logger.info("Benchmarking dataset creation...")

    start = time.time()
    dataset = IceCubeDataset(split="train", max_events=10000)
    elapsed = time.time() - start

    logger.info(f"Dataset creation: {elapsed:.2f}s for {len(dataset)} events")
    logger.info(f"Time per event: {elapsed/len(dataset)*1000:.2f}ms")
    return dataset


def benchmark_getitem(dataset, n_samples=100):
    """Benchmark dataset __getitem__."""
    logger.info("=" * 60)
    logger.info(f"Benchmarking dataset __getitem__ ({n_samples} samples)...")

    start = time.time()
    for i in range(n_samples):
        _ = dataset[i]
    elapsed = time.time() - start

    logger.info(f"__getitem__: {elapsed:.2f}s for {n_samples} samples")
    logger.info(f"Time per sample: {elapsed/n_samples*1000:.2f}ms")
    logger.info(f"Throughput: {n_samples/elapsed:.1f} samples/sec")


def benchmark_collation(dataset, batch_size=32, n_batches=10):
    """Benchmark collation function."""
    logger.info("=" * 60)
    logger.info(f"Benchmarking collation (batch_size={batch_size}, {n_batches} batches)...")

    # Get batches manually
    total_time = 0.0
    for i in range(n_batches):
        batch_events = [dataset[j] for j in range(i*batch_size, (i+1)*batch_size)]

        start = time.time()
        _ = collate_dom_packing(batch_events, max_seq_len=512)
        elapsed = time.time() - start
        total_time += elapsed

    avg_time = total_time / n_batches
    logger.info(f"Collation: {avg_time*1000:.2f}ms per batch (avg over {n_batches} batches)")
    logger.info(f"Events per second: {batch_size/avg_time:.1f}")


def benchmark_dataloader(dataset, batch_size=32, num_workers=4, n_batches=20):
    """Benchmark full DataLoader pipeline."""
    logger.info("=" * 60)
    logger.info(f"Benchmarking DataLoader (batch_size={batch_size}, workers={num_workers})...")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=512),
        pin_memory=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Warmup
    logger.info("Warming up...")
    for i, batch in enumerate(loader):
        if i >= 2:
            break

    # Benchmark
    logger.info("Benchmarking...")
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
    elapsed = time.time() - start

    avg_time = elapsed / n_batches
    logger.info(f"DataLoader: {avg_time*1000:.2f}ms per batch (avg over {n_batches} batches)")
    logger.info(f"Throughput: {batch_size/avg_time:.1f} events/sec")
    logger.info(f"Total time for {n_batches} batches: {elapsed:.2f}s")


def benchmark_forward_pass(batch_size=32):
    """Benchmark model forward pass (for comparison)."""
    logger.info("=" * 60)
    logger.info(f"Benchmarking model forward pass (batch_size={batch_size})...")

    from iceaggr.models import HierarchicalTransformer

    # Create model
    model = HierarchicalTransformer(
        d_model=128,
        t1_n_layers=4,
        t2_n_layers=4,
    ).cuda()

    # Get a real batch
    dataset = IceCubeDataset(split="train", max_events=1000)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=512),
    )

    batch = next(iter(loader))

    # Move to GPU
    def to_device(batch, device):
        result = {}
        for k, v in batch.items():
            if k == 'metadata':
                result[k] = {}
                for mk, mv in v.items():
                    result[k][mk] = mv.to(device) if isinstance(mv, torch.Tensor) else mv
            else:
                result[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return result

    batch = to_device(batch, 'cuda')

    # Warmup
    for _ in range(5):
        _ = model(batch)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    n_iters = 50
    for _ in range(n_iters):
        _ = model(batch)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_time = elapsed / n_iters
    logger.info(f"Forward pass: {avg_time*1000:.2f}ms per batch (avg over {n_iters} iterations)")
    logger.info(f"Throughput: {batch_size/avg_time:.1f} events/sec")


def main():
    logger.info("Data Loading Bottleneck Analysis")
    logger.info("=" * 60)

    # 1. Dataset creation
    dataset = benchmark_dataset_creation()

    # 2. Dataset __getitem__
    benchmark_getitem(dataset, n_samples=100)

    # 3. Collation function
    benchmark_collation(dataset, batch_size=32, n_batches=10)

    # 4. DataLoader with different worker counts
    for num_workers in [0, 2, 4, 8]:
        benchmark_dataloader(dataset, batch_size=32, num_workers=num_workers, n_batches=20)

    # 5. Model forward pass (for comparison)
    benchmark_forward_pass(batch_size=32)

    logger.info("=" * 60)
    logger.info("Benchmarking complete!")


if __name__ == "__main__":
    main()

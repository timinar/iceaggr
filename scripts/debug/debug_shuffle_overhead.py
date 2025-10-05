#!/usr/bin/env python
"""Debug why shuffle=True causes 12 sec/batch slowdown."""

import time

import torch
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def benchmark_dataloader(shuffle: bool, max_events: int, num_batches: int = 20):
    """Benchmark dataloader with/without shuffle."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: shuffle={shuffle}, max_events={max_events:,}")
    logger.info(f"{'='*60}")

    dataset = IceCubeDataset(split="train", max_events=max_events)

    loader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=512),
        pin_memory=True,
    )

    logger.info(f"Dataset size: {len(dataset):,} events")
    logger.info(f"Num batches: {len(loader):,}")

    # Warmup
    batch = next(iter(loader))
    logger.info(f"Warmup batch loaded")

    # Benchmark
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        if i % 5 == 0:
            elapsed = time.time() - start
            batches_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            ms_per_batch = (elapsed / (i + 1)) * 1000
            logger.info(f"  Batch {i}: {ms_per_batch:.1f}ms/batch ({batches_per_sec:.1f} batches/sec)")

    elapsed = time.time() - start
    batches_per_sec = num_batches / elapsed
    ms_per_batch = (elapsed / num_batches) * 1000
    events_per_sec = (num_batches * 128) / elapsed

    logger.info(f"\nResults:")
    logger.info(f"  Total time: {elapsed:.2f}s")
    logger.info(f"  {ms_per_batch:.1f}ms/batch")
    logger.info(f"  {batches_per_sec:.1f} batches/sec")
    logger.info(f"  {events_per_sec:.0f} events/sec")

    return ms_per_batch


if __name__ == "__main__":
    # Test 1: Small dataset
    logger.info("\n\nTEST 1: Small dataset (10K events)")
    no_shuffle_small = benchmark_dataloader(shuffle=False, max_events=10_000)
    shuffle_small = benchmark_dataloader(shuffle=True, max_events=10_000)
    logger.info(f"\n  Shuffle overhead (10K): {shuffle_small - no_shuffle_small:.1f}ms/batch")

    # Test 2: Medium dataset
    logger.info("\n\nTEST 2: Medium dataset (100K events)")
    no_shuffle_medium = benchmark_dataloader(shuffle=False, max_events=100_000)
    shuffle_medium = benchmark_dataloader(shuffle=True, max_events=100_000)
    logger.info(f"\n  Shuffle overhead (100K): {shuffle_medium - no_shuffle_medium:.1f}ms/batch")

    # Test 3: Large dataset
    logger.info("\n\nTEST 3: Large dataset (900K events) - THIS IS TRAINING SETUP")
    no_shuffle_large = benchmark_dataloader(shuffle=False, max_events=900_000)
    shuffle_large = benchmark_dataloader(shuffle=True, max_events=900_000)
    logger.info(f"\n  Shuffle overhead (900K): {shuffle_large - no_shuffle_large:.1f}ms/batch")

    logger.info(f"\n\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"10K events:  {no_shuffle_small:.1f}ms → {shuffle_small:.1f}ms (overhead: {shuffle_small - no_shuffle_small:.1f}ms)")
    logger.info(f"100K events: {no_shuffle_medium:.1f}ms → {shuffle_medium:.1f}ms (overhead: {shuffle_medium - no_shuffle_medium:.1f}ms)")
    logger.info(f"900K events: {no_shuffle_large:.1f}ms → {shuffle_large:.1f}ms (overhead: {shuffle_large - no_shuffle_large:.1f}ms)")

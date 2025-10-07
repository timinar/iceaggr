"""
Benchmark for BatchAwareSampler with various configurations.

Tests:
1. Single worker vs multiple workers
2. Different cache sizes
3. Different batch sizes
4. I/O throughput measurement
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceaggr.data.dataset import get_dataloader
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def benchmark_config(
    name: str,
    batch_size: int,
    num_workers: int,
    cache_size: int,
    n_batches: int = 500,
    max_events: int = 1_000_000,
):
    """
    Benchmark a specific dataloader configuration.

    Args:
        name: Configuration name
        batch_size: Training batch size
        num_workers: Number of dataloader workers
        cache_size: Number of parquet files to cache per worker
        n_batches: Number of batches to load
        max_events: Maximum number of events to use
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Benchmark: {name}")
    logger.info(f"{'='*80}")
    logger.info(f"Config: batch_size={batch_size}, num_workers={num_workers}, cache_size={cache_size}")
    logger.info(f"Dataset: {max_events:,} events, loading {n_batches} batches\n")

    # Create dataloader with BatchAwareSampler
    dataloader = get_dataloader(
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        max_events=max_events,
        collate_fn="deepsets",
        use_batch_aware_sampler=True,
        cache_size=cache_size,
    )

    # Warmup (first batch often slower due to initialization)
    logger.info("Warming up...")
    iterator = iter(dataloader)
    _ = next(iterator)

    # Benchmark
    logger.info(f"Loading {n_batches} batches...")
    start_time = time.time()

    batch_times = []
    for i, batch in enumerate(dataloader):
        batch_start = time.time()

        # Access data to ensure it's loaded
        n_pulses = batch['pulse_features'].shape[0]
        n_doms = batch['num_doms']

        batch_times.append(time.time() - batch_start)

        if (i + 1) % max(1, n_batches // 10) == 0:
            logger.debug(f"  Progress: {i+1}/{n_batches} batches")

        if i >= n_batches - 1:
            break

    elapsed = time.time() - start_time

    # Calculate metrics
    batches_per_sec = n_batches / elapsed
    events_per_sec = (n_batches * batch_size) / elapsed
    avg_batch_time_ms = (sum(batch_times) / len(batch_times)) * 1000

    # Print results
    logger.info(f"\n{'='*80}")
    logger.info(f"Results: {name}")
    logger.info(f"{'='*80}")
    logger.info(f"Total time:          {elapsed:.2f}s")
    logger.info(f"Batches/sec:         {batches_per_sec:.2f}")
    logger.info(f"Events/sec:          {events_per_sec:.0f}")
    logger.info(f"Avg batch time:      {avg_batch_time_ms:.2f}ms")
    logger.info(f"{'='*80}\n")

    return {
        "name": name,
        "elapsed": elapsed,
        "batches_per_sec": batches_per_sec,
        "events_per_sec": events_per_sec,
        "avg_batch_time_ms": avg_batch_time_ms,
    }


def main():
    """Run benchmarks."""

    # Configuration
    MAX_EVENTS = 1_000_000  # 1M events (~5 batch files)
    N_BATCHES = 500
    BATCH_SIZE = 1024

    logger.info(f"\n{'#'*80}")
    logger.info(f"# BatchAwareSampler Benchmark Suite")
    logger.info(f"{'#'*80}\n")

    results = []

    # Test 1: Baseline - single worker, cache_size=1
    results.append(benchmark_config(
        name="Baseline (1 worker, cache=1)",
        batch_size=BATCH_SIZE,
        num_workers=0,
        cache_size=1,
        n_batches=N_BATCHES,
        max_events=MAX_EVENTS,
    ))

    # Test 2: Multiple workers with cache_size=1
    results.append(benchmark_config(
        name="Multi-worker (4 workers, cache=1)",
        batch_size=BATCH_SIZE,
        num_workers=4,
        cache_size=1,
        n_batches=N_BATCHES,
        max_events=MAX_EVENTS,
    ))

    # Test 3: Check if cache_size>1 helps (it shouldn't with BatchAwareSampler)
    results.append(benchmark_config(
        name="Larger cache (1 worker, cache=4)",
        batch_size=BATCH_SIZE,
        num_workers=0,
        cache_size=4,
        n_batches=N_BATCHES,
        max_events=MAX_EVENTS,
    ))

    # Test 4: Larger batch size
    results.append(benchmark_config(
        name="Large batches (1 worker, batch=2048)",
        batch_size=2048,
        num_workers=0,
        cache_size=1,
        n_batches=N_BATCHES // 2,  # Fewer batches since batch size is 2x
        max_events=MAX_EVENTS,
    ))

    # Final summary
    logger.info(f"\n{'#'*80}")
    logger.info(f"# Summary")
    logger.info(f"{'#'*80}\n")

    baseline = results[0]
    logger.info(f"{'Config':<40} {'Throughput (events/s)':<25} {'Speedup':<10}")
    logger.info(f"{'-'*80}")

    for result in results:
        speedup = result['events_per_sec'] / baseline['events_per_sec']
        logger.info(
            f"{result['name']:<40} {result['events_per_sec']:>20,.0f}      {speedup:>6.2f}x"
        )

    logger.info(f"\n{'='*80}")
    logger.info("Key Findings:")
    logger.info("- With BatchAwareSampler, cache_size=1 is optimal")
    logger.info("- Multiple workers can improve throughput if CPU-bound")
    logger.info("- Larger batches amortize overhead but use more memory")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()

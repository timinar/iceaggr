"""
Benchmark throughput of the subsampled dataloader.

Tests the end-to-end performance with BSZ=4096, aiming for 10+ batches/sec (40,960+ events/sec).

Usage:
    uv run python tests/benchmarks/test_subsampled_throughput.py
"""

import time
import argparse
from pathlib import Path
import torch

from iceaggr.data import get_subsampled_dataloader
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def benchmark_dataloader(
    dataloader,
    n_batches: int = 100,
    warmup_batches: int = 10,
    device: str = "cpu",
):
    """
    Benchmark dataloader throughput.

    Args:
        dataloader: PyTorch DataLoader to benchmark
        n_batches: Number of batches to measure (after warmup)
        warmup_batches: Number of batches to skip for warmup
        device: Device to transfer data to ('cpu' or 'cuda')

    Returns:
        Dict with performance metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARKING DATALOADER")
    logger.info(f"{'='*80}")
    logger.info(f"Device: {device}")
    logger.info(f"Warmup batches: {warmup_batches}")
    logger.info(f"Measurement batches: {n_batches}")
    logger.info(f"{'='*80}\n")

    # Warmup phase
    logger.info("Warming up...")
    iter_dl = iter(dataloader)
    for i in range(warmup_batches):
        try:
            batch = next(iter_dl)
            if device == "cuda":
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        except StopIteration:
            logger.warning(f"Dataloader exhausted during warmup at batch {i}")
            iter_dl = iter(dataloader)
            batch = next(iter_dl)

    # Measurement phase
    logger.info("Starting measurement...")
    batch_times = []
    total_events = 0
    total_pulses = 0
    padding_stats = []

    start_time = time.time()

    for i in range(n_batches):
        try:
            batch_start = time.time()
            batch = next(iter_dl)

            # Transfer to device
            if device == "cuda":
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            # Collect statistics
            batch_size = len(batch["event_lengths"])
            total_events += batch_size
            total_pulses += batch["event_lengths"].sum().item()

            # Padding efficiency: valid_pulses / total_positions
            valid_pulses = batch["event_lengths"].sum().item()
            total_positions = batch["pulse_features"].shape[0] * batch["pulse_features"].shape[1]
            padding_efficiency = valid_pulses / total_positions if total_positions > 0 else 0
            padding_stats.append(padding_efficiency)

            if (i + 1) % 10 == 0:
                avg_time = sum(batch_times[-10:]) / 10
                logger.info(f"  Batch {i+1}/{n_batches}: {avg_time*1000:.1f}ms/batch")

        except StopIteration:
            logger.warning(f"Dataloader exhausted at batch {i}")
            break

    end_time = time.time()
    total_time = end_time - start_time

    # Compute metrics
    batches_per_sec = len(batch_times) / total_time
    events_per_sec = total_events / total_time
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_padding_efficiency = sum(padding_stats) / len(padding_stats) if padding_stats else 0

    metrics = {
        "total_time": total_time,
        "n_batches": len(batch_times),
        "total_events": total_events,
        "total_pulses": total_pulses,
        "batches_per_sec": batches_per_sec,
        "events_per_sec": events_per_sec,
        "avg_batch_time": avg_batch_time,
        "avg_padding_efficiency": avg_padding_efficiency,
    }

    # Print results
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Batches processed: {len(batch_times)}")
    logger.info(f"Events processed: {total_events:,}")
    logger.info(f"Pulses processed: {total_pulses:,}")
    logger.info(f"")
    logger.info(f"Throughput:")
    logger.info(f"  Batches/sec: {batches_per_sec:.2f}")
    logger.info(f"  Events/sec: {events_per_sec:,.0f}")
    logger.info(f"  Pulses/sec: {total_pulses/total_time:,.0f}")
    logger.info(f"")
    logger.info(f"Latency:")
    logger.info(f"  Avg batch time: {avg_batch_time*1000:.1f}ms")
    logger.info(f"")
    logger.info(f"Efficiency:")
    logger.info(f"  Padding efficiency: {avg_padding_efficiency*100:.1f}%")
    logger.info(f"{'='*80}\n")

    # Check target performance
    target_batches_per_sec = 10
    if batches_per_sec >= target_batches_per_sec:
        logger.info(f"✓ TARGET ACHIEVED: {batches_per_sec:.2f} batches/sec >= {target_batches_per_sec} target")
    else:
        logger.warning(
            f"✗ TARGET MISSED: {batches_per_sec:.2f} batches/sec < {target_batches_per_sec} target "
            f"({batches_per_sec/target_batches_per_sec*100:.1f}% of target)"
        )

    return metrics


def run_benchmarks(
    metadata_path: str,
    data_root: str,
    batch_sizes: list = [4096],
    num_workers_list: list = [1, 2, 4, 8],
    max_events: int = None,
    n_batches: int = 100,
):
    """
    Run benchmarks with different configurations.

    Args:
        metadata_path: Path to event metadata
        data_root: Root directory with data
        batch_sizes: List of batch sizes to test
        num_workers_list: List of num_workers to test
        max_events: Optional limit on events
        n_batches: Number of batches to measure per config
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running benchmarks on device: {device}")

    results = []

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            logger.info(f"\n{'#'*80}")
            logger.info(f"CONFIGURATION: batch_size={batch_size}, num_workers={num_workers}")
            logger.info(f"{'#'*80}")

            # Create dataloader
            dataloader = get_subsampled_dataloader(
                metadata_path=metadata_path,
                data_root=data_root,
                split="train",
                batch_size=batch_size,
                max_seq_len=512,
                num_workers=num_workers,
                max_events=max_events,
                drop_last=True,  # Ensure consistent batch sizes
                cache_size=1,
            )

            # Run benchmark
            metrics = benchmark_dataloader(
                dataloader=dataloader,
                n_batches=n_batches,
                warmup_batches=10,
                device=device,
            )

            metrics["batch_size"] = batch_size
            metrics["num_workers"] = num_workers
            results.append(metrics)

            # Clean up
            del dataloader

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Batch Size':<12} {'Workers':<10} {'Batches/s':<12} {'Events/s':<15} {'Padding %':<12}")
    logger.info(f"{'-'*80}")
    for r in results:
        logger.info(
            f"{r['batch_size']:<12} {r['num_workers']:<10} "
            f"{r['batches_per_sec']:<12.2f} {r['events_per_sec']:<15,.0f} "
            f"{r['avg_padding_efficiency']*100:<12.1f}"
        )
    logger.info(f"{'='*80}\n")

    # Find best configuration
    best = max(results, key=lambda x: x["batches_per_sec"])
    logger.info("BEST CONFIGURATION:")
    logger.info(f"  batch_size={best['batch_size']}, num_workers={best['num_workers']}")
    logger.info(f"  Throughput: {best['batches_per_sec']:.2f} batches/sec, {best['events_per_sec']:,.0f} events/sec")
    logger.info(f"  Padding efficiency: {best['avg_padding_efficiency']*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark subsampled dataloader throughput")
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="event_metadata.parquet",
        help="Path to event metadata file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/groups/pheno/inar/icecube_kaggle",
        help="Root directory containing data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size to test (default: 4096)",
    )
    parser.add_argument(
        "--num-workers",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of num_workers to test (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to use (default: all)",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=100,
        help="Number of batches to measure (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer configurations",
    )

    args = parser.parse_args()

    # Parse num_workers list
    num_workers_list = [int(x) for x in args.num_workers.split(",")]

    if args.quick:
        logger.info("Running quick test...")
        num_workers_list = [4]
        args.n_batches = 20

    # Run benchmarks
    run_benchmarks(
        metadata_path=args.metadata_path,
        data_root=args.data_root,
        batch_sizes=[args.batch_size],
        num_workers_list=num_workers_list,
        max_events=args.max_events,
        n_batches=args.n_batches,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark tests comparing legacy vs vectorized DOM grouping collators.
"""

import time
import torch
import pytest
from iceaggr.data import (
    IceCubeDataset,
    collate_with_dom_grouping,
    collate_with_dom_grouping_legacy,
)


def benchmark_collator(collate_fn, batches, n_warmup=2, n_runs=10):
    """Benchmark a collator function.

    Args:
        collate_fn: Collator function to benchmark
        batches: List of batches (each batch is a list of events)
        n_warmup: Number of warmup iterations
        n_runs: Number of timed iterations

    Returns:
        dict with timing statistics
    """
    # Warmup
    for i in range(n_warmup):
        for batch in batches:
            _ = collate_fn(batch)

    # Timed runs
    times = []
    total_pulses = 0
    total_doms = 0

    for _ in range(n_runs):
        run_time = 0.0
        for batch in batches:
            start = time.perf_counter()
            result = collate_fn(batch)
            end = time.perf_counter()
            run_time += end - start

            if total_pulses == 0:  # Only count once
                total_pulses += result['pulse_features'].shape[0]
                total_doms += result['total_doms']

        times.append(run_time)

    times = torch.tensor(times)
    return {
        'mean_time': times.mean().item(),
        'std_time': times.std().item(),
        'min_time': times.min().item(),
        'max_time': times.max().item(),
        'total_pulses': total_pulses,
        'total_doms': total_doms,
        'batches_per_sec': len(batches) / times.mean().item(),
    }


class TestCollatorPerformance:
    """Benchmark tests for collator performance."""

    @pytest.fixture(scope="class")
    def dataset(self):
        """Load dataset for benchmarking."""
        return IceCubeDataset(split="train", max_events=5000)

    @pytest.fixture(scope="class")
    def batches_32(self, dataset):
        """Create batches of size 32."""
        batch_size = 32
        n_batches = 50
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            if end <= len(dataset):
                batches.append([dataset[j] for j in range(start, end)])
        return batches

    @pytest.fixture(scope="class")
    def batches_64(self, dataset):
        """Create batches of size 64."""
        batch_size = 64
        n_batches = 25
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            if end <= len(dataset):
                batches.append([dataset[j] for j in range(start, end)])
        return batches

    def test_vectorized_faster_than_legacy_bs32(self, batches_32):
        """Test vectorized collator is faster than legacy (batch_size=32)."""
        legacy_stats = benchmark_collator(
            collate_with_dom_grouping_legacy, batches_32, n_warmup=2, n_runs=5
        )
        vectorized_stats = benchmark_collator(
            collate_with_dom_grouping, batches_32, n_warmup=2, n_runs=5
        )

        speedup = legacy_stats['mean_time'] / vectorized_stats['mean_time']

        print(f"\n=== Batch Size 32 ===")
        print(f"Legacy:     {legacy_stats['mean_time']:.4f}s ± {legacy_stats['std_time']:.4f}s")
        print(f"Vectorized: {vectorized_stats['mean_time']:.4f}s ± {vectorized_stats['std_time']:.4f}s")
        print(f"Speedup:    {speedup:.1f}x")
        print(f"Batches/s (legacy):     {legacy_stats['batches_per_sec']:.1f}")
        print(f"Batches/s (vectorized): {vectorized_stats['batches_per_sec']:.1f}")

        # Vectorized should be faster
        assert speedup > 1.0, f"Vectorized should be faster, got {speedup:.2f}x"

    def test_vectorized_faster_than_legacy_bs64(self, batches_64):
        """Test vectorized collator is faster than legacy (batch_size=64)."""
        legacy_stats = benchmark_collator(
            collate_with_dom_grouping_legacy, batches_64, n_warmup=2, n_runs=5
        )
        vectorized_stats = benchmark_collator(
            collate_with_dom_grouping, batches_64, n_warmup=2, n_runs=5
        )

        speedup = legacy_stats['mean_time'] / vectorized_stats['mean_time']

        print(f"\n=== Batch Size 64 ===")
        print(f"Legacy:     {legacy_stats['mean_time']:.4f}s ± {legacy_stats['std_time']:.4f}s")
        print(f"Vectorized: {vectorized_stats['mean_time']:.4f}s ± {vectorized_stats['std_time']:.4f}s")
        print(f"Speedup:    {speedup:.1f}x")
        print(f"Batches/s (legacy):     {legacy_stats['batches_per_sec']:.1f}")
        print(f"Batches/s (vectorized): {vectorized_stats['batches_per_sec']:.1f}")

        # Vectorized should be faster, especially for larger batches
        assert speedup > 1.0, f"Vectorized should be faster, got {speedup:.2f}x"

    def test_memory_usage_similar(self, batches_32):
        """Test memory usage is similar between implementations."""
        import gc

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Run legacy and measure peak memory (approximation via tensor allocation)
        batch = batches_32[0]
        legacy_result = collate_with_dom_grouping_legacy(batch)
        legacy_tensors = sum(
            t.numel() * t.element_size()
            for t in legacy_result.values()
            if isinstance(t, torch.Tensor)
        )

        vectorized_result = collate_with_dom_grouping(batch)
        vectorized_tensors = sum(
            t.numel() * t.element_size()
            for t in vectorized_result.values()
            if isinstance(t, torch.Tensor)
        )

        print(f"\n=== Memory Usage (single batch) ===")
        print(f"Legacy output:     {legacy_tensors / 1024:.1f} KB")
        print(f"Vectorized output: {vectorized_tensors / 1024:.1f} KB")

        # Output size should be essentially identical
        assert abs(legacy_tensors - vectorized_tensors) < 1024, \
            "Output tensor sizes should be similar"


class TestScalingBehavior:
    """Test how performance scales with batch size."""

    @pytest.fixture(scope="class")
    def dataset(self):
        """Load dataset for scaling tests."""
        return IceCubeDataset(split="train", max_events=2000)

    def test_scaling_with_batch_size(self, dataset):
        """Test performance scaling with batch size."""
        batch_sizes = [8, 16, 32, 64]
        n_events = 256  # Fixed number of events to process

        print("\n=== Scaling with Batch Size ===")
        print(f"{'Batch Size':<12} {'Legacy (s)':<12} {'Vectorized (s)':<15} {'Speedup':<10}")
        print("-" * 50)

        for bs in batch_sizes:
            n_batches = n_events // bs
            batches = []
            for i in range(n_batches):
                start = i * bs
                batches.append([dataset[j] for j in range(start, start + bs)])

            legacy_stats = benchmark_collator(
                collate_with_dom_grouping_legacy, batches, n_warmup=1, n_runs=3
            )
            vectorized_stats = benchmark_collator(
                collate_with_dom_grouping, batches, n_warmup=1, n_runs=3
            )

            speedup = legacy_stats['mean_time'] / vectorized_stats['mean_time']
            print(f"{bs:<12} {legacy_stats['mean_time']:<12.4f} {vectorized_stats['mean_time']:<15.4f} {speedup:<10.1f}x")

            # All batch sizes should show improvement
            assert speedup > 1.0, f"Batch size {bs}: vectorized should be faster"


def run_detailed_benchmark():
    """Run detailed benchmark with output suitable for documentation."""
    print("\n" + "=" * 70)
    print("Collator Performance Benchmark")
    print("=" * 70)

    dataset = IceCubeDataset(split="train", max_events=5000)

    batch_sizes = [16, 32, 64, 128]
    n_batches = 50

    results = []

    for bs in batch_sizes:
        batches = []
        for i in range(n_batches):
            start = i * bs
            end = start + bs
            if end <= len(dataset):
                batches.append([dataset[j] for j in range(start, end)])

        if len(batches) < 10:
            continue

        legacy_stats = benchmark_collator(
            collate_with_dom_grouping_legacy, batches, n_warmup=3, n_runs=10
        )
        vectorized_stats = benchmark_collator(
            collate_with_dom_grouping, batches, n_warmup=3, n_runs=10
        )

        speedup = legacy_stats['mean_time'] / vectorized_stats['mean_time']

        results.append({
            'batch_size': bs,
            'legacy_time': legacy_stats['mean_time'],
            'vectorized_time': vectorized_stats['mean_time'],
            'speedup': speedup,
            'legacy_batches_per_sec': legacy_stats['batches_per_sec'],
            'vectorized_batches_per_sec': vectorized_stats['batches_per_sec'],
        })

        print(f"\nBatch Size: {bs}")
        print(f"  Legacy:     {legacy_stats['mean_time']:.4f}s ({legacy_stats['batches_per_sec']:.1f} batches/s)")
        print(f"  Vectorized: {vectorized_stats['mean_time']:.4f}s ({vectorized_stats['batches_per_sec']:.1f} batches/s)")
        print(f"  Speedup:    {speedup:.1f}x")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"Average speedup: {avg_speedup:.1f}x")

    return results


if __name__ == "__main__":
    run_detailed_benchmark()

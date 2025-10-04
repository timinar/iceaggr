#!/usr/bin/env python3
"""
Test and benchmark the IceCube Dataset implementation.
"""

import sys
sys.path.insert(0, 'src')

import torch
import time
import numpy as np
from iceaggr.data import IceCubeDataset, get_dataloader


def test_basic_functionality():
    """Test basic dataset functionality."""
    print("=" * 70)
    print("Test 1: Basic Dataset Functionality")
    print("=" * 70)

    # Load small subset for testing
    dataset = IceCubeDataset(
        config_path="data_config.yaml", split="train", max_events=1000
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single event access
    event = dataset[0]
    print(f"\nFirst event:")
    print(f"  Event ID: {event['event_id'].item()}")
    print(f"  N pulses: {event['n_pulses'].item()}")
    print(f"  Pulse features shape: {event['pulse_features'].shape}")
    print(f"  Target (az, zen): {event['target'].numpy()}")

    # Check a few more events
    for idx in [10, 100, 500]:
        event = dataset[idx]
        print(
            f"\nEvent {idx}: {event['n_pulses'].item()} pulses, "
            f"event_id={event['event_id'].item()}"
        )

    print("\n✓ Basic functionality test passed")


def test_collate_function():
    """Test the variable-length collate function."""
    print("\n" + "=" * 70)
    print("Test 2: Collate Function")
    print("=" * 70)

    dataloader = get_dataloader(
        config_path="data_config.yaml",
        split="train",
        batch_size=8,
        shuffle=False,
        max_events=100,
    )

    batch = next(iter(dataloader))

    print(f"\nBatch structure:")
    print(f"  Batch size: {len(batch['event_ids'])}")
    print(f"  Total pulses in batch: {len(batch['pulse_features'])}")
    print(f"  Pulse features shape: {batch['pulse_features'].shape}")
    print(f"  Pulse-to-event mapping shape: {batch['pulse_to_event_idx'].shape}")
    print(f"  Event lengths: {batch['event_lengths'].tolist()}")
    print(f"  Targets shape: {batch['targets'].shape}")

    # Verify consistency
    assert batch["pulse_features"].shape[0] == batch["pulse_to_event_idx"].shape[0]
    assert batch["event_lengths"].sum().item() == batch["pulse_features"].shape[0]
    assert len(batch["event_ids"]) == len(batch["event_lengths"])

    print("\n✓ Collate function test passed")


def benchmark_throughput(n_batches=100, batch_size=32):
    """Benchmark data loading throughput."""
    print("\n" + "=" * 70)
    print(f"Test 3: Throughput Benchmark ({n_batches} batches, bs={batch_size})")
    print("=" * 70)

    dataloader = get_dataloader(
        config_path="data_config.yaml",
        split="train",
        batch_size=batch_size,
        shuffle=False,
        max_events=n_batches * batch_size,
        num_workers=0,  # Single process for fair comparison
    )

    # Warmup
    print("\nWarming up...")
    _ = next(iter(dataloader))

    # Benchmark
    print(f"Loading {n_batches} batches...")
    start_time = time.time()
    total_events = 0
    total_pulses = 0

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        total_events += len(batch["event_ids"])
        total_pulses += batch["pulse_features"].shape[0]

    elapsed_time = time.time() - start_time

    events_per_sec = total_events / elapsed_time
    pulses_per_sec = total_pulses / elapsed_time
    batches_per_sec = n_batches / elapsed_time

    print(f"\nResults:")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Batches/sec: {batches_per_sec:.1f}")
    print(f"  Events/sec: {events_per_sec:.1f}")
    print(f"  Pulses/sec: {pulses_per_sec:,.0f}")
    print(f"  Avg pulses/event: {total_pulses/total_events:.1f}")

    print("\n✓ Throughput benchmark complete")
    return {
        "events_per_sec": events_per_sec,
        "pulses_per_sec": pulses_per_sec,
        "batches_per_sec": batches_per_sec,
    }


def test_variable_lengths():
    """Test handling of events with varying pulse counts."""
    print("\n" + "=" * 70)
    print("Test 4: Variable Length Handling")
    print("=" * 70)

    # Load larger subset to get diverse event sizes
    dataset = IceCubeDataset(
        config_path="data_config.yaml", split="train", max_events=10000
    )

    # Sample events and collect pulse counts
    sample_indices = np.random.choice(len(dataset), 1000, replace=False)
    pulse_counts = [dataset[idx]["n_pulses"].item() for idx in sample_indices]

    print(f"\nPulse count statistics (n=1000 events):")
    print(f"  Min: {np.min(pulse_counts)}")
    print(f"  Max: {np.max(pulse_counts)}")
    print(f"  Mean: {np.mean(pulse_counts):.1f}")
    print(f"  Median: {np.median(pulse_counts):.1f}")
    print(f"  Std: {np.std(pulse_counts):.1f}")
    print(f"  95th %ile: {np.percentile(pulse_counts, 95):.1f}")
    print(f"  99th %ile: {np.percentile(pulse_counts, 99):.1f}")

    # Test batch with very different event sizes
    # Find one small and one large event
    small_idx = sample_indices[np.argmin(pulse_counts)]
    large_idx = sample_indices[np.argmax(pulse_counts)]

    small_event = dataset[small_idx]
    large_event = dataset[large_idx]

    print(f"\nExtreme examples:")
    print(
        f"  Smallest event: {small_event['n_pulses'].item()} pulses "
        f"(event_id={small_event['event_id'].item()})"
    )
    print(
        f"  Largest event: {large_event['n_pulses'].item()} pulses "
        f"(event_id={large_event['event_id'].item()})"
    )

    print("\n✓ Variable length test passed")


def main():
    print("\nIceCube Dataset Test Suite")
    print("=" * 70)

    test_basic_functionality()
    test_collate_function()
    test_variable_lengths()
    results = benchmark_throughput(n_batches=100, batch_size=32)

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

    # Save benchmark results
    import json
    from pathlib import Path

    output_path = Path("notes/dataset_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to {output_path}")


if __name__ == "__main__":
    main()

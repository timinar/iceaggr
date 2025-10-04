#!/usr/bin/env python3
"""
Benchmark I/O performance for IceCube data loading on Lustre.
Tests PyArrow loading speeds and analyzes data characteristics.
"""

import pyarrow.parquet as pq
import time
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path("/groups/pheno/inar/icecube_kaggle")

def benchmark_single_batch_load(batch_id=1):
    """Benchmark loading a single batch file."""
    batch_path = DATA_DIR / "train" / f"batch_{batch_id}.parquet"

    print(f"\n{'='*70}")
    print(f"Benchmarking batch_{batch_id}.parquet")
    print(f"{'='*70}")

    # Cold read
    start = time.time()
    table = pq.read_table(batch_path)
    cold_time = time.time() - start

    n_pulses = len(table)
    file_size_mb = batch_path.stat().st_size / (1024**2)

    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Pulses: {n_pulses:,}")
    print(f"Cold read: {cold_time:.3f}s ({file_size_mb/cold_time:.1f} MB/s)")

    # Warm read
    start = time.time()
    table = pq.read_table(batch_path)
    warm_time = time.time() - start
    print(f"Warm read: {warm_time:.3f}s ({file_size_mb/warm_time:.1f} MB/s)")

    # Analyze data structure
    print(f"\nColumns: {table.column_names}")
    print(f"Schema: {table.schema}")

    return {
        'batch_id': batch_id,
        'file_size_mb': file_size_mb,
        'n_pulses': n_pulses,
        'cold_time_s': cold_time,
        'warm_time_s': warm_time,
        'cold_throughput_mbs': file_size_mb / cold_time,
        'warm_throughput_mbs': file_size_mb / warm_time
    }

def analyze_metadata():
    """Analyze metadata structure and event characteristics."""
    print(f"\n{'='*70}")
    print("Analyzing metadata")
    print(f"{'='*70}")

    meta_path = DATA_DIR / "train_meta.parquet"

    start = time.time()
    meta = pq.read_table(meta_path)
    load_time = time.time() - start

    file_size_mb = meta_path.stat().st_size / (1024**2)
    n_events = len(meta)

    print(f"Metadata file size: {file_size_mb:.1f} MB")
    print(f"Load time: {load_time:.3f}s ({file_size_mb/load_time:.1f} MB/s)")
    print(f"Total events: {n_events:,}")
    print(f"Columns: {meta.column_names}")

    # Compute per-event pulse counts
    first_pulse = meta.column('first_pulse_index').to_numpy()
    last_pulse = meta.column('last_pulse_index').to_numpy()
    pulse_counts = last_pulse - first_pulse + 1

    print(f"\nPulses per event statistics:")
    print(f"  Mean: {pulse_counts.mean():.1f}")
    print(f"  Median: {np.median(pulse_counts):.1f}")
    print(f"  Std: {pulse_counts.std():.1f}")
    print(f"  Min: {pulse_counts.min()}")
    print(f"  Max: {pulse_counts.max()}")
    print(f"  95th percentile: {np.percentile(pulse_counts, 95):.1f}")
    print(f"  99th percentile: {np.percentile(pulse_counts, 99):.1f}")
    print(f"  99.9th percentile: {np.percentile(pulse_counts, 99.9):.1f}")

    # Analyze batch distribution
    batch_ids = meta.column('batch_id').to_numpy()
    unique_batches = np.unique(batch_ids)
    events_per_batch = np.bincount(batch_ids)[unique_batches]

    print(f"\nBatch statistics:")
    print(f"  Number of batches: {len(unique_batches)}")
    print(f"  Events per batch (typical): {events_per_batch[0]}")
    print(f"  Events per batch (min): {events_per_batch.min()}")
    print(f"  Events per batch (max): {events_per_batch.max()}")

    return {
        'n_events': int(n_events),
        'n_batches': int(len(unique_batches)),
        'events_per_batch': int(events_per_batch[0]),
        'pulses_stats': {
            'mean': float(pulse_counts.mean()),
            'median': float(np.median(pulse_counts)),
            'std': float(pulse_counts.std()),
            'min': int(pulse_counts.min()),
            'max': int(pulse_counts.max()),
            'p95': float(np.percentile(pulse_counts, 95)),
            'p99': float(np.percentile(pulse_counts, 99)),
            'p999': float(np.percentile(pulse_counts, 99.9))
        }
    }

def benchmark_sequential_batches(n_batches=10):
    """Benchmark sequential loading of multiple batches."""
    print(f"\n{'='*70}")
    print(f"Sequential loading of {n_batches} batches")
    print(f"{'='*70}")

    batch_ids = list(range(1, n_batches + 1))

    start = time.time()
    total_pulses = 0
    total_size_mb = 0

    for batch_id in batch_ids:
        batch_path = DATA_DIR / "train" / f"batch_{batch_id}.parquet"
        table = pq.read_table(batch_path)
        total_pulses += len(table)
        total_size_mb += batch_path.stat().st_size / (1024**2)

    total_time = time.time() - start

    print(f"Total batches: {n_batches}")
    print(f"Total size: {total_size_mb:.1f} MB")
    print(f"Total pulses: {total_pulses:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Throughput: {total_size_mb/total_time:.1f} MB/s")
    print(f"Time per batch: {total_time/n_batches:.3f}s")

    return {
        'n_batches': n_batches,
        'total_size_mb': total_size_mb,
        'total_time_s': total_time,
        'throughput_mbs': total_size_mb / total_time,
        'time_per_batch_s': total_time / n_batches
    }

def test_event_extraction_speed():
    """Test speed of extracting individual events from a batch."""
    print(f"\n{'='*70}")
    print("Event extraction speed test")
    print(f"{'='*70}")

    # Load metadata for batch 1
    meta = pq.read_table(DATA_DIR / "train_meta.parquet")
    batch_1_mask = meta.column('batch_id').to_numpy() == 1
    batch_1_meta = meta.filter(batch_1_mask)

    # Load batch 1 data
    batch_table = pq.read_table(DATA_DIR / "train" / "batch_1.parquet")

    n_events = len(batch_1_meta)
    print(f"Events in batch 1: {n_events}")

    # Test extracting 100 random events
    n_test = 100
    event_indices = np.random.choice(n_events, n_test, replace=False)

    first_pulse = batch_1_meta.column('first_pulse_index').to_numpy()
    last_pulse = batch_1_meta.column('last_pulse_index').to_numpy()

    start = time.time()
    for idx in event_indices:
        offset = first_pulse[idx]
        length = last_pulse[idx] - first_pulse[idx] + 1
        event_table = batch_table.slice(offset, length)
        # Convert to numpy to simulate actual usage
        _ = event_table.column('time').to_numpy()
        _ = event_table.column('charge').to_numpy()
        _ = event_table.column('sensor_id').to_numpy()

    extraction_time = time.time() - start

    print(f"Extracted {n_test} events in {extraction_time:.3f}s")
    print(f"Time per event: {extraction_time/n_test*1000:.2f}ms")

    return {
        'n_events_extracted': n_test,
        'total_time_s': extraction_time,
        'time_per_event_ms': extraction_time / n_test * 1000
    }

def main():
    print("IceCube Data Loading Benchmark")
    print("="*70)

    results = {}

    # Single batch benchmark
    results['single_batch'] = benchmark_single_batch_load(batch_id=1)

    # Metadata analysis
    results['metadata'] = analyze_metadata()

    # Sequential batch loading
    results['sequential_batches'] = benchmark_sequential_batches(n_batches=10)

    # Event extraction speed
    results['event_extraction'] = test_event_extraction_speed()

    # Save results
    output_path = Path("notes/io_benchmark_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

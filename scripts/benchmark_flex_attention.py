#!/usr/bin/env python3
"""
Benchmark FlexAttention approaches for DOM-level aggregation on real IceCube data.

Compares:
1. Loop-based (baseline)
2. FlexAttention dense (no BlockMask)
3. Padded standard attention

Usage:
    uv run python scripts/benchmark_flex_attention.py [--max-events N] [--batch-sizes 4,8,16]
"""

import time
import logging
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from typing import Dict, List
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceaggr.utils import get_logger

logger = get_logger(__name__)

# Check if FlexAttention is available
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False
    logger.warning("FlexAttention not available. Install PyTorch 2.5+ with CUDA.")


def benchmark_on_real_data(
    batch_sizes: List[int] = [4, 8, 16],
    n_trials: int = 5,
    device: str = 'cpu',
    max_events: int = 1000
) -> pl.DataFrame:
    """
    Benchmark on real IceCube events.

    Args:
        batch_sizes: List of batch sizes to test
        n_trials: Number of trials per configuration
        device: 'cpu' or 'cuda'
        max_events: Number of events to load from dataset

    Returns:
        Polars DataFrame with benchmark results
    """
    from iceaggr.data import get_dataloader

    logger.info(f"Loading real IceCube data (max {max_events} events)...")

    # Create models
    encoder = SimplePulseEncoder(d_model=128).to(device)
    attention = SimpleAttentionLayer(d_model=128).to(device)
    encoder.eval()
    attention.eval()

    results = []

    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch_size={batch_size} on real data")

        # Create dataloader with DOM grouping
        dataloader = get_dataloader(
            split='train',
            batch_size=batch_size,
            shuffle=False,
            max_events=max_events,
            collate_fn='dom_grouping'
        )

        # Get first batch
        batch = next(iter(dataloader))

        total_pulses = len(batch['pulse_features'])
        total_doms = batch['total_doms']

        logger.info(f"  Total pulses: {total_pulses}, Total DOMs: {total_doms}")
        logger.info(f"  Pulse distribution: min={batch['dom_pulse_counts'].min().item()}, "
              f"median={batch['dom_pulse_counts'].median().item():.0f}, "
              f"max={batch['dom_pulse_counts'].max().item()}")

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = process_doms_loop(batch, encoder, attention, device)

        if device == 'cuda':
            torch.cuda.synchronize()

        # 1. Loop-based
        times_loop = []
        with torch.no_grad():
            for _ in range(n_trials):
                start = time.perf_counter()
                _ = process_doms_loop(batch, encoder, attention, device)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times_loop.append(time.perf_counter() - start)

        logger.info(f"  Loop: {np.mean(times_loop)*1000:.2f}ms ± {np.std(times_loop)*1000:.2f}ms")

        # 2. Padded
        times_padded = []
        with torch.no_grad():
            for _ in range(n_trials):
                start = time.perf_counter()
                _ = process_doms_padded(batch, encoder, attention, device)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times_padded.append(time.perf_counter() - start)

        logger.info(f"  Padded: {np.mean(times_padded)*1000:.2f}ms ± {np.std(times_padded)*1000:.2f}ms")

        # 3. FlexAttention dense
        times_flex_dense = []
        if FLEX_AVAILABLE:
            with torch.no_grad():
                for _ in range(n_trials):
                    start = time.perf_counter()
                    _ = process_doms_flex_attention(batch, encoder, attention, device, use_block_mask=False)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    times_flex_dense.append(time.perf_counter() - start)

            logger.info(f"  FlexAttention (dense): {np.mean(times_flex_dense)*1000:.2f}ms ± {np.std(times_flex_dense)*1000:.2f}ms")
        else:
            logger.warning(f"  FlexAttention (dense): SKIPPED (not available)")

        results.append({
            'batch_size': batch_size,
            'total_pulses': total_pulses,
            'total_doms': total_doms,
            'pulses_per_dom_median': batch['dom_pulse_counts'].median().item(),
            'pulses_per_dom_max': batch['dom_pulse_counts'].max().item(),
            'loop_ms': np.mean(times_loop) * 1000,
            'loop_std_ms': np.std(times_loop) * 1000,
            'padded_ms': np.mean(times_padded) * 1000,
            'padded_std_ms': np.std(times_padded) * 1000,
            'flex_dense_ms': np.mean(times_flex_dense) * 1000 if FLEX_AVAILABLE else None,
            'flex_dense_std_ms': np.std(times_flex_dense) * 1000 if FLEX_AVAILABLE else None,
        })

    return pl.DataFrame(results)


def test_correctness(device: str = 'cpu'):
    """Test that all methods produce similar results."""
    logger.info("Running correctness tests...")

    encoder = SimplePulseEncoder(d_model=128).to(device)
    attention = SimpleAttentionLayer(d_model=128).to(device)
    encoder.eval()
    attention.eval()

    batch = create_icecube_like_batch(batch_size=4)

    with torch.no_grad():
        # Loop-based (ground truth)
        embeddings_loop = process_doms_loop(batch, encoder, attention, device)

        # Padded
        embeddings_padded = process_doms_padded(batch, encoder, attention, device)

        # Check if close
        if torch.allclose(embeddings_loop, embeddings_padded, atol=1e-4):
            logger.info("✓ Loop vs Padded: PASS")
        else:
            max_diff = (embeddings_loop - embeddings_padded).abs().max().item()
            logger.error(f"✗ Loop vs Padded: FAIL (max diff: {max_diff:.6f})")

        # FlexAttention (if available)
        if FLEX_AVAILABLE:
            embeddings_flex_dense = process_doms_flex_attention(batch, encoder, attention, device, use_block_mask=False)

            if torch.allclose(embeddings_loop, embeddings_flex_dense, atol=1e-4):
                logger.info("✓ Loop vs FlexAttention (dense): PASS")
            else:
                max_diff = (embeddings_loop - embeddings_flex_dense).abs().max().item()
                logger.error(f"✗ Loop vs FlexAttention (dense): FAIL (max diff: {max_diff:.6f})")
        else:
            logger.warning("⊘ FlexAttention tests: SKIPPED (not available)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark FlexAttention on real IceCube data')
    parser.add_argument('--max-events', type=int, default=1000, help='Max events to load from dataset')
    parser.add_argument('--batch-sizes', type=str, default='4,8,16', help='Comma-separated batch sizes')
    parser.add_argument('--n-trials', type=int, default=5, help='Number of trials per batch size')
    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("="*80)
    logger.info("BENCHMARKING ON REAL ICECUBE DATA")
    logger.info("="*80)

    results = benchmark_on_real_data(
        batch_sizes=batch_sizes,
        n_trials=args.n_trials,
        device=device,
        max_events=args.max_events
    )

    # Print results
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*80)
    print(results)  # Polars DataFrame prints nicely

    # Calculate speedups relative to loop
    if FLEX_AVAILABLE:
        logger.info("\nSpeedup relative to loop-based approach:")
        for row in results.iter_rows(named=True):
            bs = int(row['batch_size'])
            logger.info(f"  Batch size {bs}:")
            logger.info(f"    Padded: {row['loop_ms']/row['padded_ms']:.2f}x")
            if row['flex_dense_ms'] is not None:
                logger.info(f"    FlexAttention (dense): {row['loop_ms']/row['flex_dense_ms']:.2f}x")

    # Save results
    results.write_csv('/tmp/flex_attention_benchmark.csv')
    logger.info(f"\nResults saved to: /tmp/flex_attention_benchmark.csv")

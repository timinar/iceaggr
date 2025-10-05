#!/usr/bin/env python3
"""
Benchmark DOM packing approach with real extreme events.

Tests memory usage and performance on events with:
- High pulse counts (>100K pulses)
- High DOM counts (>2K DOMs)
- Variable distributions

Compares to the statistics from personal_notes/03_t1_memory_constraints.md
"""

import torch
from torch.utils.data import DataLoader
import psutil
import os
import gc
from iceaggr.data.dataset import IceCubeDataset, collate_dom_packing
from iceaggr.models.dom_transformer import DOMTransformer
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3  # Convert to GB


def find_extreme_events(dataset, n_samples=1000, top_k=10):
    """
    Scan dataset to find extreme events.

    Returns:
        List of (idx, n_pulses, n_doms, max_pulses_per_dom) tuples
    """
    logger.info(f"Scanning {n_samples} events for extremes...")

    extreme_events = []

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        pulse_features = sample['pulse_features']
        n_pulses = pulse_features.shape[0]

        # Count DOMs
        sensor_ids = pulse_features[:, 2].long()
        unique_doms = torch.unique(sensor_ids)
        n_doms = len(unique_doms)

        # Max pulses per DOM
        max_pulses_per_dom = 0
        for dom_id in unique_doms:
            dom_mask = (sensor_ids == dom_id)
            n_pulses_in_dom = dom_mask.sum().item()
            max_pulses_per_dom = max(max_pulses_per_dom, n_pulses_in_dom)

        extreme_events.append((i, n_pulses, n_doms, max_pulses_per_dom))

        if (i + 1) % 200 == 0:
            logger.info(f"  Scanned {i+1}/{n_samples} events...")

    # Sort by total pulses (descending)
    extreme_events.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"\nTop {top_k} most extreme events (by total pulses):")
    for i, (idx, n_pulses, n_doms, max_p_per_dom) in enumerate(extreme_events[:top_k]):
        logger.info(
            f"  #{i+1}: idx={idx}, {n_pulses:,} pulses, {n_doms} DOMs, "
            f"max {max_p_per_dom} pulses/DOM"
        )

    return extreme_events[:top_k]


def benchmark_batch(model, batch, max_seq_len, device='cuda'):
    """Benchmark a single batch through the model."""
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mem_before = get_memory_usage()

    # Collate
    collated = collate_dom_packing(batch, max_seq_len=max_seq_len)

    # Move to device
    collated = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in collated.items()
    }
    if 'metadata' in collated:
        collated['metadata'] = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in collated['metadata'].items()
        }

    mem_after_collate = get_memory_usage()

    # Forward pass
    with torch.no_grad():
        dom_embeddings, metadata = model(collated)

    mem_after_forward = get_memory_usage()

    # GPU memory if applicable
    gpu_mem = 0
    if device == 'cuda':
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB

    # Get batch statistics
    bsz = collated['packed_sequences'].shape[0]
    total_doms = metadata['total_doms']
    n_events = len(batch)
    total_pulses = sum(e['pulse_features'].shape[0] for e in batch)

    return {
        'n_events': n_events,
        'total_pulses': total_pulses,
        'total_doms': total_doms,
        'n_sequences': bsz,
        'mem_before': mem_before,
        'mem_after_collate': mem_after_collate,
        'mem_after_forward': mem_after_forward,
        'gpu_mem': gpu_mem,
        'dom_embeddings_shape': dom_embeddings.shape,
    }


def main():
    logger.info("="*80)
    logger.info("DOM PACKING BENCHMARK")
    logger.info("="*80)

    # Configuration
    max_seq_len = 512
    d_model = 128
    n_heads = 8
    n_layers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"\nConfiguration:")
    logger.info(f"  Device: {device}")
    logger.info(f"  max_seq_len: {max_seq_len}")
    logger.info(f"  d_model: {d_model}, n_heads: {n_heads}, n_layers: {n_layers}")

    # Load dataset
    logger.info(f"\nLoading dataset...")
    dataset = IceCubeDataset(
        config_path="src/iceaggr/data/data_config.yaml",
        split="train"
    )
    logger.info(f"  Dataset size: {len(dataset):,} events")

    # Find extreme events
    extreme_indices = find_extreme_events(dataset, n_samples=10000, top_k=20)

    # Initialize model
    logger.info(f"\nInitializing DOMTransformer...")
    model = DOMTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
    ).to(device)
    model.eval()

    # Test scenarios
    test_cases = [
        # (batch_indices, description)
        ([extreme_indices[0][0]], "Single most extreme event"),
        ([extreme_indices[i][0] for i in range(5)], "Top 5 extreme events"),
        ([extreme_indices[i][0] for i in range(10)], "Top 10 extreme events"),
        (list(range(64)), "64 random events (baseline)"),
    ]

    logger.info("\n" + "="*80)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*80)

    for indices, description in test_cases:
        logger.info(f"\n{description}:")
        logger.info("-" * 60)

        # Get batch
        batch = [dataset[i] for i in indices]

        try:
            stats = benchmark_batch(model, batch, max_seq_len, device)

            logger.info(f"  Events: {stats['n_events']}")
            logger.info(f"  Total pulses: {stats['total_pulses']:,}")
            logger.info(f"  Total DOMs: {stats['total_doms']:,}")
            logger.info(f"  Packed sequences (batch size): {stats['n_sequences']}")
            logger.info(f"  Output shape: {stats['dom_embeddings_shape']}")
            logger.info(f"  CPU Memory before: {stats['mem_before']:.2f} GB")
            logger.info(f"  CPU Memory after collate: {stats['mem_after_collate']:.2f} GB")
            logger.info(f"  CPU Memory after forward: {stats['mem_after_forward']:.2f} GB")
            if device == 'cuda':
                logger.info(f"  GPU Memory peak: {stats['gpu_mem']:.2f} GB")

            # Derived metrics
            mem_per_pulse = (stats['mem_after_forward'] - stats['mem_before']) / stats['total_pulses'] * 1024**3  # bytes
            logger.info(f"  Memory/pulse: {mem_per_pulse:.2f} bytes")

            logger.info("  ✓ SUCCESS")

        except RuntimeError as e:
            logger.error(f"  ✗ FAILED: {e}")
        except Exception as e:
            logger.error(f"  ✗ ERROR: {e}")

    logger.info("\n" + "="*80)
    logger.info("Benchmark complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

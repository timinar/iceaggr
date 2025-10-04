#!/usr/bin/env python3
"""
Benchmark end-to-end T1 → T2 pipeline performance.

Measures:
- Throughput (events/sec)
- Latency breakdown (T1 vs T2)
- Memory usage
- Scaling with event size
"""

import torch
import time
import psutil
import os
from torch.utils.data import DataLoader

from iceaggr.models import DOMTransformer, EventTransformer, EventAccumulator
from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


def benchmark_e2e(
    n_events: int = 128,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    max_seq_len: int = 512,
    device: str = 'cuda',
):
    """
    Benchmark end-to-end pipeline.

    Args:
        n_events: Number of events to process
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        max_seq_len: Maximum sequence length for T1
        device: 'cuda' or 'cpu'
    """
    logger.info(f"Benchmarking E2E pipeline: {n_events} events, d_model={d_model}")

    # Initialize models
    t1 = DOMTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len
    ).to(device)

    t2 = EventTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    ).to(device)

    t1.eval()
    t2.eval()

    # Load data
    dataset = IceCubeDataset(
        config_path="src/iceaggr/data/data_config.yaml",
        split="train",
        max_events=n_events
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=max_seq_len),
        shuffle=False
    )

    # Warm-up
    logger.info("Warming up...")
    with torch.no_grad():
        batch = next(iter(loader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if 'metadata' in batch:
            batch['metadata'] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch['metadata'].items()
            }
        dom_emb, meta = t1(batch)
        _ = t2(dom_emb, meta['sensor_ids'], meta['dom_to_event_idx'], batch_size=len(batch['metadata']['event_ids']))

    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    logger.info("Running benchmark...")
    mem_before = get_memory_usage()

    t1_time = 0
    t2_time = 0
    total_events = 0
    total_doms = 0
    total_pulses = 0

    accumulator = EventAccumulator()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if 'metadata' in batch:
                batch['metadata'] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['metadata'].items()
                }

            # T1 forward
            t1_start = time.time()
            dom_embeddings, metadata = t1(batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1_time += time.time() - t1_start

            # Accumulate
            accumulator.add_batch(dom_embeddings, metadata)

            # Track stats
            n_events = len(metadata['event_ids'])
            total_events += n_events
            total_doms += metadata['total_doms']
            total_pulses += batch['packed_sequences'].shape[0] * batch['packed_sequences'].shape[1]

        # T2 forward
        t2_start = time.time()
        for t2_batch in accumulator.get_complete_events(batch_size=32):
            t2_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t2_batch.items()}
            _ = t2(
                t2_batch['dom_embeddings'],
                t2_batch['dom_ids'],
                t2_batch['dom_to_event_idx'],
                t2_batch['batch_size']
            )
        if device == 'cuda':
            torch.cuda.synchronize()
        t2_time = time.time() - t2_start

    total_time = time.time() - start_time
    mem_after = get_memory_usage()

    # Results
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*80)
    logger.info(f"Total events: {total_events}")
    logger.info(f"Total DOMs: {total_doms:,}")
    logger.info(f"Total pulses: {total_pulses:,}")
    logger.info(f"")
    logger.info(f"T1 time: {t1_time:.3f}s ({t1_time/total_time*100:.1f}%)")
    logger.info(f"T2 time: {t2_time:.3f}s ({t2_time/total_time*100:.1f}%)")
    logger.info(f"Total time: {total_time:.3f}s")
    logger.info(f"")
    logger.info(f"Throughput: {total_events/total_time:.1f} events/sec")
    logger.info(f"Latency per event: {total_time/total_events*1000:.2f} ms")
    logger.info(f"")
    logger.info(f"CPU Memory before: {mem_before:.2f} GB")
    logger.info(f"CPU Memory after: {mem_after:.2f} GB")
    logger.info(f"CPU Memory delta: {mem_after - mem_before:.2f} GB")

    if device == 'cuda':
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"GPU Memory peak: {gpu_mem:.2f} GB")

    logger.info("="*80)


def main():
    logger.info("="*80)
    logger.info("E2E PIPELINE BENCHMARK")
    logger.info("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}\n")

    # Test different configurations
    configs = [
        {'n_events': 64, 'd_model': 64, 'n_layers': 2, 'max_seq_len': 256},
        {'n_events': 128, 'd_model': 128, 'n_layers': 4, 'max_seq_len': 512},
    ]

    for config in configs:
        logger.info(f"\nConfiguration: {config}")
        logger.info("-"*80)
        benchmark_e2e(device=device, **config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze how events get packed into sequences to understand memory usage.
"""

import torch
from iceaggr.data.dataset import IceCubeDataset, collate_dom_packing
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def analyze_packing(dataset, event_idx, max_seq_len=512):
    """Analyze how a single event gets packed."""
    event = dataset[event_idx]
    batch = [event]

    pulse_features = event['pulse_features']
    n_pulses = pulse_features.shape[0]
    sensor_ids = pulse_features[:, 2].long()
    unique_doms = torch.unique(sensor_ids)
    n_doms = len(unique_doms)

    logger.info(f"\nEvent {event_idx}:")
    logger.info(f"  Total pulses: {n_pulses:,}")
    logger.info(f"  Unique DOMs: {n_doms}")

    # Count pulses per DOM
    pulses_per_dom = []
    for dom_id in unique_doms:
        dom_mask = (sensor_ids == dom_id)
        n_pulses_in_dom = dom_mask.sum().item()
        pulses_per_dom.append(n_pulses_in_dom)

    logger.info(f"  Pulses per DOM - min: {min(pulses_per_dom)}, max: {max(pulses_per_dom)}, median: {sorted(pulses_per_dom)[len(pulses_per_dom)//2]}")

    # Collate with packing
    result = collate_dom_packing(batch, max_seq_len=max_seq_len)

    bsz = result['packed_sequences'].shape[0]
    total_elements = bsz * max_seq_len

    logger.info(f"  Packed into {bsz} sequences of length {max_seq_len}")
    logger.info(f"  Total sequence elements: {total_elements:,}")

    # Attention memory estimation
    # For FlexAttention: O(bsz * n_heads * seq_len^2) for attention scores
    # With our architecture: (bsz, n_heads, seq_len, seq_len) attention matrix per layer
    n_heads = 8
    n_layers = 4
    bytes_per_float32 = 4

    # Attention scores per layer
    attn_scores_mem = bsz * n_heads * max_seq_len * max_seq_len * bytes_per_float32
    total_attn_mem = attn_scores_mem * n_layers

    logger.info(f"  Estimated attention memory per layer: {attn_scores_mem / 1024**3:.2f} GB")
    logger.info(f"  Estimated total attention memory ({n_layers} layers): {total_attn_mem / 1024**3:.2f} GB")

    # Check actual packing efficiency
    valid_pulses = result['dom_mask'].sum().item()
    padding = total_elements - valid_pulses
    efficiency = valid_pulses / total_elements * 100

    logger.info(f"  Valid pulses: {int(valid_pulses):,}")
    logger.info(f"  Padding: {int(padding):,}")
    logger.info(f"  Packing efficiency: {efficiency:.1f}%")


def main():
    logger.info("="*80)
    logger.info("PACKING ANALYSIS")
    logger.info("="*80)

    dataset = IceCubeDataset(
        config_path="src/iceaggr/data/data_config.yaml",
        split="train"
    )

    # Test different max_seq_len values
    extreme_idx = 8489  # 110K pulses, 2015 DOMs
    moderate_idx = 100   # Typical event

    for max_seq_len in [256, 512, 1024]:
        logger.info(f"\n{'='*80}")
        logger.info(f"max_seq_len = {max_seq_len}")
        logger.info(f"{'='*80}")

        analyze_packing(dataset, extreme_idx, max_seq_len)
        analyze_packing(dataset, moderate_idx, max_seq_len)


if __name__ == "__main__":
    main()

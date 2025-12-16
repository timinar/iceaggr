"""
Collate functions for batching IceCube events.

This module provides different batching strategies:
- collate_variable_length: Continuous batching (no padding)
- collate_with_dom_grouping: Hierarchical DOM-based grouping
- collate_padded_subsampled: Padded batching with subsampling (for standard transformers)
- make_collate_with_geometry: Factory for collators with geometry lookup
"""

from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .geometry import GeometryLoader


def collate_variable_length(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length events using continuous batching.

    Instead of padding to max length, we flatten all pulses and track which
    event each pulse belongs to. This is memory-efficient for the hierarchical
    transformer architecture.

    Args:
        batch: List of event dicts from IceCubeDataset

    Returns:
        Dict with:
            - pulse_features: (total_pulses, 4) - all pulses flattened
            - pulse_to_event_idx: (total_pulses,) - which event each pulse belongs to
            - event_lengths: (batch_size,) - number of pulses per event
            - targets: (batch_size, 2) - azimuth, zenith (if available)
            - event_ids: (batch_size,) - event IDs
    """
    batch_size = len(batch)

    # Collect all pulses from all events
    pulse_features_list = [item["pulse_features"] for item in batch]
    event_lengths = torch.tensor([item["n_pulses"].item() for item in batch], dtype=torch.long)

    # Flatten all pulses
    pulse_features = torch.cat(pulse_features_list, dim=0)  # (total_pulses, 4)

    # Create pulse-to-event mapping
    pulse_to_event_idx = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long), event_lengths
    )

    # Collate other fields
    result = {
        "pulse_features": pulse_features,
        "pulse_to_event_idx": pulse_to_event_idx,
        "event_lengths": event_lengths,
        "event_ids": torch.stack([item["event_id"] for item in batch]),
    }

    # Add targets if available
    if "target" in batch[0]:
        result["targets"] = torch.stack([item["target"] for item in batch])

    return result


def collate_with_dom_grouping(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function that groups pulses by DOM for hierarchical transformer.

    Creates pulse_to_dom_idx mapping where each (event, DOM) pair gets a unique index.
    This is the correct grouping for T1 (DOM-level) transformer.

    Args:
        batch: List of event dicts from IceCubeDataset

    Returns:
        Dict with:
            - pulse_features: (total_pulses, 4) - all pulses flattened
            - pulse_to_dom_idx: (total_pulses,) - which DOM each pulse belongs to
            - pulse_idx_in_dom: (total_pulses,) - index of pulse within its DOM
            - n_pulses_in_dom: (total_pulses,) - total pulses in DOM (broadcast)
            - dom_pulse_counts: (total_doms,) - number of pulses per DOM
            - dom_to_event_idx: (total_doms,) - which event each DOM belongs to
            - dom_ids: (total_doms,) - original sensor IDs
            - event_dom_counts: (batch_size,) - number of DOMs per event
            - targets: (batch_size, 2) - azimuth, zenith (if available)
            - event_ids: (batch_size,) - event IDs
    """
    batch_size = len(batch)

    all_pulse_features = []
    pulse_to_dom_idx_list = []
    pulse_idx_in_dom_list = []
    n_pulses_in_dom_list = []
    dom_pulse_counts = []
    dom_to_event_idx = []
    dom_ids = []
    event_dom_counts = []

    current_dom_idx = 0

    for event_idx, event in enumerate(batch):
        pulse_features = event['pulse_features']  # (n_pulses, 4)
        n_pulses = pulse_features.shape[0]

        # Extract sensor IDs (column 2 of pulse features)
        sensor_ids = pulse_features[:, 2].long()

        # Find unique DOMs in this event
        unique_doms = torch.unique(sensor_ids, sorted=True)
        n_doms_in_event = len(unique_doms)
        event_dom_counts.append(n_doms_in_event)

        # Group pulses by DOM
        for dom_id in unique_doms:
            # Find pulses belonging to this DOM
            dom_mask = (sensor_ids == dom_id)
            dom_pulses = pulse_features[dom_mask]
            n_pulses_in_this_dom = dom_pulses.shape[0]

            # Add to batch
            all_pulse_features.append(dom_pulses)
            dom_pulse_counts.append(n_pulses_in_this_dom)
            dom_to_event_idx.append(event_idx)
            dom_ids.append(dom_id.item())

            # Track which DOM index each pulse belongs to
            pulse_to_dom_idx_list.extend([current_dom_idx] * n_pulses_in_this_dom)

            # Track pulse index within DOM (0, 1, 2, ..., n-1)
            pulse_idx_in_dom_list.extend(range(n_pulses_in_this_dom))

            # Broadcast DOM size to each pulse
            n_pulses_in_dom_list.extend([n_pulses_in_this_dom] * n_pulses_in_this_dom)

            current_dom_idx += 1

    # Stack everything
    result = {
        # Pulse-level (for T1)
        'pulse_features': torch.cat(all_pulse_features, dim=0),  # (total_pulses, 4)
        'pulse_to_dom_idx': torch.tensor(pulse_to_dom_idx_list, dtype=torch.long),
        'pulse_idx_in_dom': torch.tensor(pulse_idx_in_dom_list, dtype=torch.long),
        'n_pulses_in_dom': torch.tensor(n_pulses_in_dom_list, dtype=torch.long),
        'dom_pulse_counts': torch.tensor(dom_pulse_counts, dtype=torch.long),

        # DOM-level metadata (for T2)
        'dom_to_event_idx': torch.tensor(dom_to_event_idx, dtype=torch.long),
        'dom_ids': torch.tensor(dom_ids, dtype=torch.long),
        'event_dom_counts': torch.tensor(event_dom_counts, dtype=torch.long),

        # Event-level
        'event_ids': torch.stack([b['event_id'] for b in batch]),
        'total_doms': current_dom_idx,
        'batch_size': batch_size
    }

    # Add targets if available
    if 'target' in batch[0]:
        result['targets'] = torch.stack([b['target'] for b in batch])

    return result


def collate_padded_subsampled(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for padded batching with subsampled events.

    Pads all events to the maximum length in the batch. Designed for use with
    BucketBatchSampler, which groups events by similar length to minimize padding waste.

    Args:
        batch: List of event dicts from IceCubeSubsampledDataset

    Returns:
        Dict with:
            - pulse_features: (batch_size, max_len, 4) - padded pulse features
            - padding_mask: (batch_size, max_len) - True for valid positions, False for padding
            - event_lengths: (batch_size,) - actual number of pulses per event
            - targets: (batch_size, 2) - azimuth, zenith (if available)
            - event_ids: (batch_size,) - event IDs
            - bucket_ids: (batch_size,) - bucket IDs
    """
    batch_size = len(batch)

    # Get lengths and find max
    lengths = [item["n_pulses"].item() for item in batch]
    max_len = max(lengths)

    # Pre-allocate padded tensor
    pulse_features = torch.zeros(batch_size, max_len, 4, dtype=torch.float32)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Fill in actual data
    for i, (item, length) in enumerate(zip(batch, lengths)):
        pulse_features[i, :length, :] = item["pulse_features"]
        padding_mask[i, :length] = True

    # Collate other fields
    result = {
        "pulse_features": pulse_features,
        "padding_mask": padding_mask,
        "event_lengths": torch.tensor(lengths, dtype=torch.long),
        "event_ids": torch.stack([item["event_id"] for item in batch]),
        "bucket_ids": torch.stack([item["bucket_id"] for item in batch]),
    }

    # Add targets if available
    if "target" in batch[0]:
        result["targets"] = torch.stack([item["target"] for item in batch])

    return result


def make_collate_with_geometry(
    geometry: "GeometryLoader",
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """
    Factory function to create a collator with geometry lookup.

    This wraps collate_with_dom_grouping to add DOM positions from the geometry file.

    Args:
        geometry: GeometryLoader instance with sensor positions

    Returns:
        Collate function that adds 'dom_positions' to the batch dict

    Example:
        >>> from iceaggr.data import GeometryLoader, make_collate_with_geometry
        >>> geometry = GeometryLoader("/path/to/sensor_geometry.csv")
        >>> collate_fn = make_collate_with_geometry(geometry)
        >>> dataloader = DataLoader(dataset, collate_fn=collate_fn)
    """

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # First, apply standard DOM grouping
        result = collate_with_dom_grouping(batch)

        # Add DOM positions from geometry
        dom_ids = result['dom_ids']  # (total_doms,)
        result['dom_positions'] = geometry[dom_ids]  # (total_doms, 3)

        return result

    return collate_fn

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


def collate_with_dom_grouping_legacy(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Legacy collate function that groups pulses by DOM (O(B × D × P) complexity).

    Kept for A/B testing against the vectorized version.
    See collate_with_dom_grouping for the optimized implementation.
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


# Maximum sensor ID in IceCube detector (5160 DOMs total, 0-indexed)
MAX_SENSOR_ID = 5160


def collate_with_dom_grouping(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Vectorized collate function that groups pulses by DOM for hierarchical transformer.

    This is an optimized O(N log N) implementation using torch.unique instead of
    nested Python loops (which were O(B × D × P)).

    Creates pulse_to_dom_idx mapping where each (event, DOM) pair gets a unique index.
    This is the correct grouping for T1 (DOM-level) transformer.

    Args:
        batch: List of event dicts from IceCubeDataset

    Returns:
        Dict with:
            - pulse_features: (total_pulses, 4) - all pulses sorted by DOM
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

    # 1. Concatenate all events
    pulse_features_list = [event['pulse_features'] for event in batch]
    event_lengths = torch.tensor(
        [pf.shape[0] for pf in pulse_features_list], dtype=torch.long
    )
    all_features = torch.cat(pulse_features_list, dim=0)  # (N, 4)
    total_pulses = all_features.shape[0]

    # Extract sensor IDs from column 2
    sensor_ids = all_features[:, 2].long()

    # Create pulse-to-event mapping
    pulse_event_idx = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long), event_lengths
    )

    # 2. Combined key = event_idx * MAX_SENSOR_ID + sensor_id
    # This creates unique keys for each (event, DOM) pair
    combined_key = pulse_event_idx * MAX_SENSOR_ID + sensor_ids

    # 3. Single torch.unique call replaces ALL nested loops
    # sorted=True ensures DOMs are ordered by (event_idx, sensor_id)
    unique_keys, inverse_idx, dom_counts = torch.unique(
        combined_key, return_inverse=True, return_counts=True, sorted=True
    )
    total_doms = unique_keys.shape[0]

    # 4. Sort pulses by DOM (stable=True preserves temporal order within each DOM)
    sort_order = torch.argsort(inverse_idx, stable=True)
    sorted_features = all_features[sort_order]

    # Map inverse indices to sorted positions
    sorted_pulse_to_dom = inverse_idx[sort_order]

    # 5. Within-DOM indices via cumsum trick
    # dom_starts[i] = index of first pulse in DOM i
    dom_starts = torch.cat([
        torch.zeros(1, dtype=torch.long),
        dom_counts.cumsum(0)[:-1]
    ])
    # For each pulse, subtract the start index of its DOM
    pulse_idx_in_dom = torch.arange(total_pulses, dtype=torch.long) - dom_starts[sorted_pulse_to_dom]

    # Broadcast DOM counts to each pulse
    n_pulses_in_dom = dom_counts[sorted_pulse_to_dom]

    # 6. DOM metadata from key decoding
    dom_event_idx = unique_keys // MAX_SENSOR_ID
    dom_sensor_ids = unique_keys % MAX_SENSOR_ID

    # 7. Count DOMs per event
    event_dom_counts = torch.bincount(dom_event_idx, minlength=batch_size)

    # 8. Compute physics features per DOM (critical for direction reconstruction)
    # These preserve exact timing/energy info that pooling would destroy
    sorted_times = sorted_features[:, 0]  # time is column 0
    sorted_charges = sorted_features[:, 1]  # charge is column 1

    # min_time per DOM (first photon arrival - critical for direction)
    dom_min_time = torch.zeros(total_doms, dtype=sorted_features.dtype)
    dom_min_time.fill_(float('inf'))
    dom_min_time.scatter_reduce_(0, sorted_pulse_to_dom, sorted_times, reduce='amin', include_self=False)

    # sum_charge per DOM (total energy deposited)
    dom_sum_charge = torch.zeros(total_doms, dtype=sorted_features.dtype)
    dom_sum_charge.scatter_add_(0, sorted_pulse_to_dom, sorted_charges)

    # std_time per DOM (temporal spread - helps distinguish track vs cascade)
    # Compute via E[X^2] - E[X]^2
    dom_sum_time = torch.zeros(total_doms, dtype=sorted_features.dtype)
    dom_sum_time.scatter_add_(0, sorted_pulse_to_dom, sorted_times)
    dom_mean_time = dom_sum_time / dom_counts.float().clamp(min=1)

    dom_sum_time_sq = torch.zeros(total_doms, dtype=sorted_features.dtype)
    dom_sum_time_sq.scatter_add_(0, sorted_pulse_to_dom, sorted_times ** 2)
    dom_mean_time_sq = dom_sum_time_sq / dom_counts.float().clamp(min=1)
    dom_std_time = (dom_mean_time_sq - dom_mean_time ** 2).clamp(min=0).sqrt()

    # Build result
    result = {
        # Pulse-level (for T1)
        'pulse_features': sorted_features,  # (total_pulses, 4)
        'pulse_to_dom_idx': sorted_pulse_to_dom,
        'pulse_idx_in_dom': pulse_idx_in_dom,
        'n_pulses_in_dom': n_pulses_in_dom,
        'dom_pulse_counts': dom_counts,

        # DOM-level metadata (for T2)
        'dom_to_event_idx': dom_event_idx,
        'dom_ids': dom_sensor_ids,
        'event_dom_counts': event_dom_counts,

        # DOM-level physics features (preserves exact timing/energy info)
        'dom_min_time': dom_min_time,      # (total_doms,) - first photon arrival
        'dom_sum_charge': dom_sum_charge,  # (total_doms,) - total energy
        'dom_std_time': dom_std_time,      # (total_doms,) - temporal spread

        # Event-level
        'event_ids': torch.stack([b['event_id'] for b in batch]),
        'total_doms': total_doms,
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


def make_collate_flat(
    geometry: "GeometryLoader",
    max_pulses_per_dom: int = 84,
    max_doms: int = 128,
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """
    Single-pass collator for the flat transformer model.

    Fuses DOM grouping + flat vector building + geometry lookup + event padding
    into one function. Avoids creating intermediate sorted_features tensor and
    skips physics features (dom_sum_charge, dom_std_time) that the flat model
    doesn't use.

    Compared to collate_with_dom_grouping + build_flat_dom_vectors + pad_to_event_batch:
    - Skips full N-pulse gather (sorted_features) — only gathers first K pulses per DOM
    - Skips dom_sum_charge, dom_std_time, n_pulses_in_dom broadcast
    - One function call instead of three, no intermediate tensors

    Args:
        geometry: GeometryLoader instance with sensor positions
        max_pulses_per_dom: K, first K pulses kept per DOM (default: 84)
        max_doms: max DOMs per event, rest subsampled by earliest time (default: 128)

    Returns:
        Collate function producing a dict with:
            - dom_vectors: (batch_size, max_doms, input_dim) padded flat DOM features
            - padding_mask: (batch_size, max_doms) True = valid DOM
            - targets: (batch_size, 2) azimuth, zenith (if available)
            - event_ids: (batch_size,)
            - batch_size: int

    Example:
        >>> geometry = GeometryLoader("/path/to/sensor_geometry.csv")
        >>> collate_fn = make_collate_flat(geometry, max_pulses_per_dom=84, max_doms=128)
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>> batch = next(iter(loader))
        >>> batch['dom_vectors'].shape  # (32, 128, 256)
    """
    K = max_pulses_per_dom
    input_dim = 4 + 3 * K  # [x, y, z, log_n_pulses, t1, q1, a1, ..., tK, qK, aK]

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)

        # --- 1. Concatenate all events ---
        pulse_features_list = [event['pulse_features'] for event in batch]
        event_lengths = torch.tensor(
            [pf.shape[0] for pf in pulse_features_list], dtype=torch.long
        )
        all_features = torch.cat(pulse_features_list, dim=0)  # (N, 4)
        total_pulses = all_features.shape[0]

        sensor_ids = all_features[:, 2].long()
        pulse_event_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), event_lengths
        )

        # --- 2. DOM grouping via torch.unique ---
        combined_key = pulse_event_idx * MAX_SENSOR_ID + sensor_ids
        unique_keys, inverse_idx, dom_counts = torch.unique(
            combined_key, return_inverse=True, return_counts=True, sorted=True
        )
        total_doms = unique_keys.shape[0]

        # --- 3. Compute pulse_idx_in_dom without creating sorted_features ---
        # argsort to group by DOM (stable preserves time order within DOM)
        sort_order = torch.argsort(inverse_idx, stable=True)
        sorted_dom_idx = inverse_idx[sort_order]

        dom_starts = torch.zeros(total_doms + 1, dtype=torch.long)
        dom_starts[1:] = dom_counts.cumsum(0)
        pulse_idx_in_dom_sorted = (
            torch.arange(total_pulses, dtype=torch.long) - dom_starts[sorted_dom_idx]
        )

        # Map back to original pulse order
        pulse_idx_in_dom = torch.empty(total_pulses, dtype=torch.long)
        pulse_idx_in_dom[sort_order] = pulse_idx_in_dom_sorted

        # --- 4. Keep only first K pulses per DOM, scatter into flat vectors ---
        keep_mask = pulse_idx_in_dom < K
        kept_features = all_features[keep_mask]       # only the pulses we need
        kept_dom_idx = inverse_idx[keep_mask]
        kept_pulse_idx = pulse_idx_in_dom[keep_mask]

        # Normalize
        time_norm = (kept_features[:, 0] - 1e4) / 3e4
        charge_norm = torch.log10(kept_features[:, 1].clamp(min=1e-6)) / 3.0
        aux_norm = kept_features[:, 3] - 0.5

        # Scatter into (total_doms, K, 3)
        pulse_tensor = torch.zeros(total_doms, K, 3, dtype=all_features.dtype)
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 0] = time_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 1] = charge_norm
        pulse_tensor[kept_dom_idx, kept_pulse_idx, 2] = aux_norm

        # --- 5. Build full DOM vectors with geometry ---
        dom_sensor_ids = unique_keys % MAX_SENSOR_ID
        dom_positions = geometry[dom_sensor_ids]  # (total_doms, 3)

        n_pulses_norm = (torch.log1p(dom_counts.float()) / 3.0 - 1.0).unsqueeze(1)

        # [x, y, z, n_pulses, pulse_flat] → (total_doms, input_dim)
        dom_vectors = torch.cat([
            dom_positions,                                  # (D, 3)
            n_pulses_norm,                                  # (D, 1)
            pulse_tensor.reshape(total_doms, K * 3),        # (D, K*3)
        ], dim=1)

        # --- 6. DOM min_time for subsampling (cheap: just first pulse per DOM) ---
        first_pulse_mask = pulse_idx_in_dom == 0
        dom_min_time = torch.zeros(total_doms, dtype=all_features.dtype)
        dom_min_time[inverse_idx[first_pulse_mask]] = all_features[first_pulse_mask, 0]

        # --- 7. Pad to (batch_size, max_doms, input_dim) with attention mask ---
        dom_event_idx = unique_keys // MAX_SENSOR_ID
        event_dom_counts = torch.bincount(dom_event_idx, minlength=batch_size)

        dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        dom_event_starts[1:] = event_dom_counts.cumsum(0)
        dom_idx_in_event = (
            torch.arange(total_doms, dtype=torch.long) - dom_event_starts[dom_event_idx]
        )

        needs_subsample = event_dom_counts > max_doms
        if needs_subsample.any():
            priority = -dom_min_time
            keep = torch.ones(total_doms, dtype=torch.bool)

            for ev in needs_subsample.nonzero(as_tuple=True)[0]:
                s = dom_event_starts[ev]
                e = dom_event_starts[ev + 1]
                _, top = priority[s:e].topk(max_doms, largest=True)
                keep[s:e] = False
                keep[s + top] = True

            kept_idx = keep.nonzero(as_tuple=True)[0]
            dom_vectors = dom_vectors[kept_idx]
            dom_event_idx = dom_event_idx[kept_idx]

            clamped = event_dom_counts.clamp(max=max_doms)
            kept_starts = torch.zeros(batch_size + 1, dtype=torch.long)
            kept_starts[1:] = clamped.cumsum(0)
            dom_idx_in_event = (
                torch.arange(dom_vectors.shape[0], dtype=torch.long)
                - kept_starts[dom_event_idx]
            )

        valid = dom_idx_in_event < max_doms
        ev_idx = dom_event_idx[valid]
        d_idx = dom_idx_in_event[valid]

        padded = torch.zeros(batch_size, max_doms, input_dim, dtype=dom_vectors.dtype)
        mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)

        padded[ev_idx, d_idx] = dom_vectors[valid]
        mask[ev_idx, d_idx] = True

        # --- Build result ---
        result = {
            'dom_vectors': padded,    # (B, max_doms, input_dim)
            'padding_mask': mask,     # (B, max_doms)
            'event_ids': torch.stack([b['event_id'] for b in batch]),
            'batch_size': batch_size,
        }

        if 'target' in batch[0]:
            result['targets'] = torch.stack([b['target'] for b in batch])

        return result

    return collate_fn

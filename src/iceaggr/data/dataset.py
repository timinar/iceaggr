"""
IceCube Dataset with PyArrow backend for efficient variable-length event loading.

This module implements Phase 1 of the data loading strategy:
- Simple batch caching
- Full event loading (no subsampling)
- Variable-length collation
"""

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import yaml
import random
from collections import defaultdict

from iceaggr.utils import get_logger

logger = get_logger(__name__)


class BatchAwareSampler(Sampler):
    """
    PyTorch Sampler that groups data by batch_id to improve cache efficiency.

    This sampler ensures events from the same parquet file are accessed together,
    dramatically reducing disk I/O when cache_size < num_batch_files.

    The sampler:
    1. Groups events by their batch_id (parquet file)
    2. Shuffles the order of batch files each epoch
    3. Shuffles events within each batch file
    4. Yields events file-by-file for sequential access

    With this sampler, set cache_size=1 in IceCubeDataset since only one file
    is accessed at a time.

    Args:
        metadata: PyArrow table with 'batch_id' column
    """

    def __init__(self, metadata):
        self.n_events = len(metadata)
        batch_ids = metadata.column("batch_id").to_numpy()

        # Group event indices by their batch_id
        self.grouped_indices = defaultdict(list)
        for idx, batch_id in enumerate(batch_ids):
            self.grouped_indices[batch_id].append(idx)

        self.batch_keys = list(self.grouped_indices.keys())

    def __iter__(self):
        # 1. Shuffle the order of batch files (epoch-level randomness)
        random.shuffle(self.batch_keys)

        # 2. For each batch file, shuffle events and yield them
        for batch_key in self.batch_keys:
            indices_in_batch = self.grouped_indices[batch_key].copy()
            random.shuffle(indices_in_batch)
            yield from indices_in_batch

    def __len__(self):
        return self.n_events


class IceCubeDataset(Dataset):
    """
    PyTorch Dataset for IceCube neutrino events.

    Loads full events (variable length) from Parquet batch files with caching.
    Each event contains pulses with features: time, charge, sensor_id, auxiliary.

    Args:
        config_path: Path to data_config.yaml with data paths
        split: 'train' or 'test'
        max_events: Optional limit on number of events (for testing)
        cache_size: Number of batch files to keep in LRU cache (default: 1)

    Returns:
        Dict with keys:
            - 'pulse_features': (n_pulses, 4) array [time, charge, sensor_id, auxiliary]
            - 'target': (2,) array [azimuth, zenith] (if split=='train')
            - 'event_id': int
            - 'n_pulses': int
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        split: str = "train",
        max_events: Optional[int] = None,
        cache_size: int = 1,
    ):
        assert split in ["train", "test"], f"split must be 'train' or 'test', got {split}"

        # Load config - default to data_config.yaml in this directory
        if config_path is None:
            config_path = Path(__file__).parent / "data_config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.data_root = Path(config["data"]["root"])
        self.split = split

        # Load metadata
        meta_path = self.data_root / f"{split}_meta.parquet"
        logger.info(f"Loading metadata from {meta_path}...")
        self.metadata = pq.read_table(meta_path)

        if max_events is not None:
            self.metadata = self.metadata.slice(0, max_events)

        self.n_events = len(self.metadata)
        logger.info(f"Loaded {self.n_events:,} events from {split} split")

        # Extract metadata columns as numpy arrays for fast access
        self.batch_ids = self.metadata.column("batch_id").to_numpy()
        self.event_ids = self.metadata.column("event_id").to_numpy()
        self.first_pulse_idx = self.metadata.column("first_pulse_index").to_numpy()
        self.last_pulse_idx = self.metadata.column("last_pulse_index").to_numpy()

        if split == "train":
            self.azimuth = self.metadata.column("azimuth").to_numpy()
            self.zenith = self.metadata.column("zenith").to_numpy()

        # Simple LRU cache for batch files
        self.cache_size = cache_size
        self.batch_cache: Dict[int, pa.Table] = {}
        self.cache_access_order: List[int] = []

    def __len__(self) -> int:
        return self.n_events

    def _load_batch(self, batch_id: int) -> pa.Table:
        """Load a batch file with LRU caching."""
        if batch_id in self.batch_cache:
            # Move to end of access order (most recently used)
            self.cache_access_order.remove(batch_id)
            self.cache_access_order.append(batch_id)
            return self.batch_cache[batch_id]

        # Load batch file
        batch_path = self.data_root / self.split / f"batch_{batch_id}.parquet"
        batch_table = pq.read_table(batch_path)

        # Update cache
        self.batch_cache[batch_id] = batch_table
        self.cache_access_order.append(batch_id)

        # Evict oldest if cache is full
        if len(self.batch_cache) > self.cache_size:
            oldest_batch_id = self.cache_access_order.pop(0)
            del self.batch_cache[oldest_batch_id]

        return batch_table

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single event."""
        # Get event metadata
        batch_id = self.batch_ids[idx]
        event_id = self.event_ids[idx]
        first_pulse = self.first_pulse_idx[idx]
        last_pulse = self.last_pulse_idx[idx]
        n_pulses = last_pulse - first_pulse + 1

        # Load batch and extract event
        batch_table = self._load_batch(batch_id)
        event_table = batch_table.slice(first_pulse, n_pulses)

        # Extract features
        time = event_table.column("time").to_numpy()
        charge = event_table.column("charge").to_numpy()
        sensor_id = event_table.column("sensor_id").to_numpy()
        auxiliary = event_table.column("auxiliary").to_numpy()

        # Stack features: [time, charge, sensor_id, auxiliary]
        pulse_features = np.stack(
            [time, charge, sensor_id, auxiliary.astype(np.float32)], axis=1
        )

        # Convert to torch tensors
        result = {
            "pulse_features": torch.from_numpy(pulse_features).float(),
            "event_id": torch.tensor(event_id, dtype=torch.long),
            "n_pulses": torch.tensor(n_pulses, dtype=torch.long),
        }

        # Add targets for training split
        if self.split == "train":
            target = np.array([self.azimuth[idx], self.zenith[idx]], dtype=np.float32)
            result["target"] = torch.from_numpy(target)

        return result


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
            n_pulses_in_dom = dom_pulses.shape[0]

            # Add to batch
            all_pulse_features.append(dom_pulses)
            dom_pulse_counts.append(n_pulses_in_dom)
            dom_to_event_idx.append(event_idx)
            dom_ids.append(dom_id.item())

            # Track which DOM index each pulse belongs to
            pulse_to_dom_idx_list.extend([current_dom_idx] * n_pulses_in_dom)

            current_dom_idx += 1

    # Stack everything
    result = {
        # Pulse-level (for T1)
        'pulse_features': torch.cat(all_pulse_features, dim=0),  # (total_pulses, 4)
        'pulse_to_dom_idx': torch.tensor(pulse_to_dom_idx_list, dtype=torch.long),
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


def collate_deepsets(
    batch: List[Dict[str, torch.Tensor]],
    geometry_table: torch.Tensor | None = None
) -> Dict[str, torch.Tensor]:
    """
    Optimized collate function for DeepSets DOM encoder.

    More efficient than collate_with_dom_grouping by using vectorized operations
    where possible and avoiding redundant sorting.

    IMPORTANT: Replaces sensor_id with geometry (x, y, z) in pulse features!

    Args:
        batch: List of event dicts from IceCubeDataset
        geometry_table: (5160, 3) tensor of sensor positions, or None to load from config

    Returns:
        Dict with:
            - pulse_features: (total_pulses, 6) - [time, charge, x, y, z, auxiliary]
            - pulse_to_dom_idx: (total_pulses,) - which DOM each pulse belongs to
            - num_doms: int - total number of unique DOMs in batch
            - dom_to_event_idx: (num_doms,) - which event each DOM belongs to
            - dom_ids: (num_doms,) - original sensor IDs for each DOM
            - event_dom_counts: (batch_size,) - number of DOMs per event
            - targets: (batch_size, 2) - azimuth, zenith (if available)
            - event_ids: (batch_size,) - event IDs
    """
    batch_size = len(batch)

    # Load geometry if not provided
    if geometry_table is None:
        from pathlib import Path
        import yaml
        config_path = Path(__file__).parent / "data_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        data_root = Path(config["data"]["root"])
        geometry_path = data_root / "sensor_geometry.csv"
        import pandas as pd
        geometry_df = pd.read_csv(geometry_path)
        if 'sensor_id' in geometry_df.columns:
            coords = geometry_df[['x', 'y', 'z']].values
        else:
            coords = geometry_df.values[:, :3]
        geometry_table = torch.from_numpy(coords).float()

    # Flatten all pulse features and create event membership
    pulse_features_list = []
    pulse_to_event_list = []

    for event_idx, event in enumerate(batch):
        pulse_features = event['pulse_features']  # (n_pulses, 4) [time, charge, sensor_id, auxiliary]
        n_pulses = pulse_features.shape[0]

        pulse_features_list.append(pulse_features)
        pulse_to_event_list.append(torch.full((n_pulses,), event_idx, dtype=torch.long))

    # Concatenate all pulses
    all_pulse_features = torch.cat(pulse_features_list, dim=0)  # (total_pulses, 4)
    pulse_to_event = torch.cat(pulse_to_event_list, dim=0)  # (total_pulses,)

    # Extract sensor IDs (column 2) and geometry
    sensor_ids = all_pulse_features[:, 2].long()  # (total_pulses,)
    pulse_geometry = geometry_table[sensor_ids]  # (total_pulses, 3) - lookup x,y,z

    # Replace sensor_id with geometry in pulse features
    # Old: [time, charge, sensor_id, auxiliary]
    # New: [time, charge, x, y, z, auxiliary]
    time = all_pulse_features[:, 0:1]  # (total_pulses, 1)
    charge = all_pulse_features[:, 1:2]  # (total_pulses, 1)
    auxiliary = all_pulse_features[:, 3:4]  # (total_pulses, 1)
    all_pulse_features = torch.cat([time, charge, pulse_geometry, auxiliary], dim=1)  # (total_pulses, 6)

    # Create unique (event, sensor_id) pairs for DOM identification
    # This is the key optimization: we create a unique DOM ID per event-sensor pair
    # using a hash: event_id * max_sensor_id + sensor_id
    max_sensor_id = 5160  # IceCube has 5160 DOMs
    dom_hash = pulse_to_event * max_sensor_id + sensor_ids  # (total_pulses,)

    # Find unique DOMs and create mapping
    unique_dom_hashes, inverse_indices = torch.unique(dom_hash, return_inverse=True, sorted=True)
    pulse_to_dom_idx = inverse_indices  # (total_pulses,) - which DOM each pulse belongs to
    num_doms = len(unique_dom_hashes)

    # Decode DOM metadata from hashes
    dom_to_event_idx = unique_dom_hashes // max_sensor_id  # (num_doms,)
    dom_ids = unique_dom_hashes % max_sensor_id  # (num_doms,)

    # Count DOMs per event
    event_dom_counts = torch.bincount(dom_to_event_idx, minlength=batch_size)

    # Build result dict
    result = {
        # Pulse-level (for DeepSets encoder)
        'pulse_features': all_pulse_features,  # (total_pulses, 6) [time, charge, x, y, z, auxiliary]
        'pulse_to_dom_idx': pulse_to_dom_idx,  # (total_pulses,)
        'num_doms': num_doms,

        # DOM-level metadata (for T2)
        'dom_to_event_idx': dom_to_event_idx,  # (num_doms,)
        'dom_ids': dom_ids,  # (num_doms,)
        'event_dom_counts': event_dom_counts,  # (batch_size,)

        # Event-level
        'event_ids': torch.stack([b['event_id'] for b in batch]),  # (batch_size,)
        'batch_size': batch_size
    }

    # Add targets if available
    if 'target' in batch[0]:
        result['targets'] = torch.stack([b['target'] for b in batch])  # (batch_size, 2)

    return result


def get_dataloader(
    config_path: Optional[str] = None,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_events: Optional[int] = None,
    collate_fn: str = "variable_length",
    use_batch_aware_sampler: bool = True,
    cache_size: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for IceCube events.

    Args:
        config_path: Path to data_config.yaml (defaults to src/iceaggr/data/data_config.yaml)
        split: 'train' or 'test'
        batch_size: Number of events per batch
        shuffle: Whether to shuffle events (ignored if use_batch_aware_sampler=True)
        num_workers: Number of worker processes (0 = single process)
        max_events: Optional limit on number of events
        collate_fn: "variable_length", "dom_grouping", or "deepsets"
        use_batch_aware_sampler: If True (default), use BatchAwareSampler for efficient I/O
        cache_size: Number of batch files to cache. Defaults to 1 if use_batch_aware_sampler=True, else 4

    Returns:
        DataLoader with continuous batching collate function
    """
    # Set optimal cache size based on sampler
    if cache_size is None:
        cache_size = 1 if use_batch_aware_sampler else 4

    dataset = IceCubeDataset(
        config_path=config_path, split=split, max_events=max_events, cache_size=cache_size
    )

    # Select collate function
    if collate_fn == "variable_length":
        collate_func = collate_variable_length
    elif collate_fn == "dom_grouping":
        collate_func = collate_with_dom_grouping
    elif collate_fn == "deepsets":
        collate_func = collate_deepsets
    else:
        raise ValueError(f"Unknown collate_fn: {collate_fn}. Use 'variable_length', 'dom_grouping', or 'deepsets'")

    # Create sampler for efficient file access
    if use_batch_aware_sampler:
        sampler = BatchAwareSampler(dataset.metadata)
        shuffle = False  # Sampler handles shuffling
        logger.info(f"Using BatchAwareSampler with cache_size={cache_size}")
    else:
        sampler = None
        logger.info(f"Using random sampling with cache_size={cache_size}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=True,  # Faster GPU transfer
    )

    return dataloader

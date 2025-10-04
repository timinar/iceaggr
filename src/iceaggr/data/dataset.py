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
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import yaml


class IceCubeDataset(Dataset):
    """
    PyTorch Dataset for IceCube neutrino events.

    Loads full events (variable length) from Parquet batch files with caching.
    Each event contains pulses with features: time, charge, sensor_id, auxiliary.

    Args:
        config_path: Path to config.yaml with data paths
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
        config_path: str = "config.yaml",
        split: str = "train",
        max_events: Optional[int] = None,
        cache_size: int = 1,
    ):
        assert split in ["train", "test"], f"split must be 'train' or 'test', got {split}"

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.data_root = Path(config["data"]["root"])
        self.split = split

        # Load metadata
        meta_path = self.data_root / f"{split}_meta.parquet"
        print(f"Loading metadata from {meta_path}...")
        self.metadata = pq.read_table(meta_path)

        if max_events is not None:
            self.metadata = self.metadata.slice(0, max_events)

        self.n_events = len(self.metadata)
        print(f"Loaded {self.n_events:,} events from {split} split")

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


def get_dataloader(
    config_path: str = "config.yaml",
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_events: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for IceCube events.

    Args:
        config_path: Path to config.yaml
        split: 'train' or 'test'
        batch_size: Number of events per batch
        shuffle: Whether to shuffle events
        num_workers: Number of worker processes (0 = single process)
        max_events: Optional limit on number of events

    Returns:
        DataLoader with continuous batching collate function
    """
    dataset = IceCubeDataset(
        config_path=config_path, split=split, max_events=max_events
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True,  # Faster GPU transfer
    )

    return dataloader

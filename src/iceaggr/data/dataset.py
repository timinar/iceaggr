"""
IceCube Dataset with PyArrow backend for efficient variable-length event loading.

This module provides:
- IceCubeDataset: Full event loading (no subsampling)
- IceCubeSubsampledDataset: Bucketed subsampled loading (future)
- get_dataloader: Convenience function for creating dataloaders
"""

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional
import yaml

from iceaggr.utils import get_logger
from .samplers import BatchAwareSampler, BucketBatchSampler
from .collators import collate_variable_length, collate_with_dom_grouping, collate_padded_subsampled

logger = get_logger(__name__)


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


class IceCubeSubsampledDataset(Dataset):
    """
    PyTorch Dataset for IceCube events with uniform random subsampling.

    This dataset loads events from a metadata file that includes bucket assignments.
    Events exceeding max_seq_len are uniformly subsampled.

    Args:
        metadata_path: Path to event_metadata.parquet (with bucket_id column)
        data_root: Root directory containing batch files
        split: 'train' or 'test' (determines subdirectory)
        max_seq_len: Maximum sequence length (events exceeding this are subsampled)
        max_events: Optional limit on number of events (for testing)
        cache_size: Number of batch files to keep in LRU cache (default: 1)

    Returns:
        Dict with keys:
            - 'pulse_features': (n_pulses, 4) array [time, charge, sensor_id, auxiliary]
            - 'target': (2,) array [azimuth, zenith] (if split=='train')
            - 'event_id': int
            - 'n_pulses': int
            - 'bucket_id': int
    """

    def __init__(
        self,
        metadata_path: str,
        data_root: str,
        split: str = "train",
        max_seq_len: int = 512,
        max_events: Optional[int] = None,
        cache_size: int = 1,
    ):
        assert split in ["train", "test"], f"split must be 'train' or 'test', got {split}"

        self.data_root = Path(data_root)
        self.split = split
        self.max_seq_len = max_seq_len

        # Load event metadata (includes bucket_id)
        logger.info(f"Loading event metadata from {metadata_path}...")
        self.metadata = pq.read_table(metadata_path)

        if max_events is not None:
            self.metadata = self.metadata.slice(0, max_events)

        self.n_events = len(self.metadata)
        logger.info(f"Loaded {self.n_events:,} events from {split} split")

        # Extract metadata columns as numpy arrays for fast access
        self.batch_ids = self.metadata.column("batch_id").to_numpy()
        self.event_ids = self.metadata.column("event_id").to_numpy()
        self.first_pulse_idx = self.metadata.column("first_pulse_index").to_numpy()
        self.last_pulse_idx = self.metadata.column("last_pulse_index").to_numpy()
        self.bucket_ids = self.metadata.column("bucket_id").to_numpy()
        self.pulse_counts = self.metadata.column("pulse_count").to_numpy()

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
        """Get a single event with optional subsampling."""
        # Get event metadata
        batch_id = self.batch_ids[idx]
        event_id = self.event_ids[idx]
        first_pulse = self.first_pulse_idx[idx]
        last_pulse = self.last_pulse_idx[idx]
        n_pulses = last_pulse - first_pulse + 1
        bucket_id = self.bucket_ids[idx]

        # Load batch and extract event
        batch_table = self._load_batch(batch_id)
        event_table = batch_table.slice(first_pulse, n_pulses)

        # Extract features
        time = event_table.column("time").to_numpy()
        charge = event_table.column("charge").to_numpy()
        sensor_id = event_table.column("sensor_id").to_numpy()
        auxiliary = event_table.column("auxiliary").to_numpy()

        # Perform uniform random subsampling if needed
        if n_pulses > self.max_seq_len:
            # Uniform random sampling
            indices = np.random.choice(n_pulses, size=self.max_seq_len, replace=False)
            indices = np.sort(indices)  # Keep temporal order
            time = time[indices]
            charge = charge[indices]
            sensor_id = sensor_id[indices]
            auxiliary = auxiliary[indices]
            n_pulses = self.max_seq_len

        # Stack features: [time, charge, sensor_id, auxiliary]
        pulse_features = np.stack(
            [time, charge, sensor_id, auxiliary.astype(np.float32)], axis=1
        )

        # Convert to torch tensors
        result = {
            "pulse_features": torch.from_numpy(pulse_features).float(),
            "event_id": torch.tensor(event_id, dtype=torch.long),
            "n_pulses": torch.tensor(n_pulses, dtype=torch.long),
            "bucket_id": torch.tensor(bucket_id, dtype=torch.long),
        }

        # Add targets for training split
        if self.split == "train":
            target = np.array([self.azimuth[idx], self.zenith[idx]], dtype=np.float32)
            result["target"] = torch.from_numpy(target)

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
        collate_fn: "variable_length" or "dom_grouping"
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
    else:
        raise ValueError(f"Unknown collate_fn: {collate_fn}. Use 'variable_length' or 'dom_grouping'")

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


def get_subsampled_dataloader(
    metadata_path: str,
    data_root: str,
    split: str = "train",
    batch_size: int = 4096,
    max_seq_len: int = 512,
    num_workers: int = 4,
    max_events: Optional[int] = None,
    drop_last: bool = False,
    cache_size: int = 1,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for subsampled IceCube events with bucketed batching.

    This creates a high-throughput dataloader optimized for padded transformers:
    - Uses IceCubeSubsampledDataset for uniform random subsampling
    - Uses BucketBatchSampler for batch-aware + length-bucketing
    - Uses collate_padded_subsampled for efficient padding

    Args:
        metadata_path: Path to event_metadata.parquet (with bucket_id column)
        data_root: Root directory containing batch files
        split: 'train' or 'test'
        batch_size: Number of events per batch (default: 4096 for high throughput)
        max_seq_len: Maximum sequence length (events exceeding this are subsampled, default: 512)
        num_workers: Number of worker processes (default: 4, tune for performance)
        max_events: Optional limit on number of events
        drop_last: If True, drop incomplete batches (default: False)
        cache_size: Number of batch files to cache (default: 1, optimal for BucketBatchSampler)

    Returns:
        DataLoader with padded batching for subsampled events
    """
    # Create dataset
    dataset = IceCubeSubsampledDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        split=split,
        max_seq_len=max_seq_len,
        max_events=max_events,
        cache_size=cache_size,
    )

    # Create batch sampler (combines batch-aware + bucketing)
    batch_sampler = BucketBatchSampler(
        metadata=dataset.metadata,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    # Create dataloader
    # Note: When using batch_sampler, batch_size and shuffle must not be specified
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_padded_subsampled,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )

    logger.info(
        f"Created subsampled dataloader: {len(dataset):,} events, "
        f"{len(batch_sampler):,} batches, batch_size={batch_size}, "
        f"num_workers={num_workers}, max_seq_len={max_seq_len}"
    )

    return dataloader

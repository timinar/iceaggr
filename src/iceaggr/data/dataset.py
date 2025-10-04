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

from iceaggr.utils import get_logger

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


def collate_dom_packing(
    batch: List[Dict[str, torch.Tensor]],
    max_seq_len: int = 512,
    max_pulses_per_batch: int = 32768,  # 64 * 512 default
) -> Dict[str, torch.Tensor]:
    """
    Collate function with DOM packing for memory-efficient T1 batching.

    Packs multiple sparse DOMs into fixed-length sequences to maximize GPU
    efficiency while controlling memory usage. Uses dynamic batch accumulation
    (Option A from design doc).

    Args:
        batch: List of event dicts from IceCubeDataset
        max_seq_len: Maximum sequence length for packing (default: 512)
        max_pulses_per_batch: Maximum total pulses to process (default: 32768)

    Returns:
        Dict with:
            - packed_sequences: (bsz, max_seq_len, 4) - packed pulse features
            - dom_boundaries: (bsz, max_seq_len) - local DOM ID at each position
            - dom_mask: (bsz, max_seq_len) - 1 for valid pulses, 0 for padding
            - metadata: Dict with:
                - global_dom_ids: (bsz, max_seq_len) - global DOM index for aggregation
                - total_doms: int - total number of unique DOMs
                - dom_to_event_idx: (total_doms,) - which event each DOM belongs to
                - event_ids: (n_events,) - original event IDs
                - targets: (n_events, 2) - azimuth, zenith (if available)
    """
    # Collect all (event, DOM) pairs with their pulses
    dom_data = []  # List of (event_idx, dom_id, pulses_tensor)
    event_ids = []
    targets = []

    for event_idx, event in enumerate(batch):
        pulse_features = event['pulse_features']  # (n_pulses, 4)
        event_ids.append(event['event_id'])
        if 'target' in event:
            targets.append(event['target'])

        # Extract sensor IDs (column 2 of pulse features)
        sensor_ids = pulse_features[:, 2].long()

        # Group by DOM
        unique_doms = torch.unique(sensor_ids, sorted=True)
        for dom_id in unique_doms:
            dom_mask = (sensor_ids == dom_id)
            dom_pulses = pulse_features[dom_mask]
            dom_data.append((event_idx, dom_id.item(), dom_pulses))

    total_doms = len(dom_data)

    # Pack DOMs into sequences
    packed_sequences = []
    dom_boundaries = []  # Local DOM ID within sequence
    dom_masks = []  # Valid pulse mask
    global_dom_ids = []  # Global DOM index for aggregation
    dom_to_event_idx = []
    sensor_ids = []  # Track actual sensor IDs for T2

    current_seq = []
    current_boundaries = []
    current_masks = []
    current_global_ids = []
    current_local_dom_id = 0

    for global_dom_idx, (event_idx, dom_id, pulses) in enumerate(dom_data):
        n_pulses = pulses.shape[0]

        # Check if adding this DOM would exceed sequence length
        current_len = len(current_boundaries)  # Number of pulses, not tensors
        if current_len + n_pulses > max_seq_len and current_len > 0:
            # Finalize current sequence and start new one
            packed_sequences.append(
                torch.nn.functional.pad(
                    torch.cat(current_seq, dim=0),
                    (0, 0, 0, max_seq_len - current_len),
                    value=0.0
                )
            )
            dom_boundaries.append(
                torch.nn.functional.pad(
                    torch.tensor(current_boundaries, dtype=torch.long),
                    (0, max_seq_len - current_len),
                    value=-1  # Padding marker
                )
            )
            dom_masks.append(
                torch.nn.functional.pad(
                    torch.tensor(current_masks, dtype=torch.float32),
                    (0, max_seq_len - current_len),
                    value=0.0
                )
            )
            global_dom_ids.append(
                torch.nn.functional.pad(
                    torch.tensor(current_global_ids, dtype=torch.long),
                    (0, max_seq_len - current_len),
                    value=-1  # Padding marker
                )
            )

            # Reset for next sequence
            current_seq = []
            current_boundaries = []
            current_masks = []
            current_global_ids = []
            current_local_dom_id = 0

        # Handle DOMs larger than max_seq_len (chunk them)
        if n_pulses > max_seq_len:
            # Take first max_seq_len pulses (could also use reservoir sampling)
            pulses = pulses[:max_seq_len]
            n_pulses = max_seq_len

        # Add DOM pulses to current sequence
        current_seq.append(pulses)
        current_boundaries.extend([current_local_dom_id] * n_pulses)
        current_masks.extend([1.0] * n_pulses)
        current_global_ids.extend([global_dom_idx] * n_pulses)
        current_local_dom_id += 1

        # Track DOM metadata
        dom_to_event_idx.append(event_idx)
        sensor_ids.append(dom_id)

    # Finalize last sequence
    if len(current_seq) > 0:
        current_len = len(current_boundaries)  # Number of pulses, not tensors
        packed_sequences.append(
            torch.nn.functional.pad(
                torch.cat(current_seq, dim=0),
                (0, 0, 0, max_seq_len - current_len),
                value=0.0
            )
        )
        dom_boundaries.append(
            torch.nn.functional.pad(
                torch.tensor(current_boundaries, dtype=torch.long),
                (0, max_seq_len - current_len),
                value=-1
            )
        )
        dom_masks.append(
            torch.nn.functional.pad(
                torch.tensor(current_masks, dtype=torch.float32),
                (0, max_seq_len - current_len),
                value=0.0
            )
        )
        global_dom_ids.append(
            torch.nn.functional.pad(
                torch.tensor(current_global_ids, dtype=torch.long),
                (0, max_seq_len - current_len),
                value=-1
            )
        )

    # Stack into batch
    result = {
        'packed_sequences': torch.stack(packed_sequences, dim=0),  # (bsz, max_seq_len, 4)
        'dom_boundaries': torch.stack(dom_boundaries, dim=0),  # (bsz, max_seq_len)
        'dom_mask': torch.stack(dom_masks, dim=0),  # (bsz, max_seq_len)
        'metadata': {
            'global_dom_ids': torch.stack(global_dom_ids, dim=0),  # (bsz, max_seq_len)
            'total_doms': total_doms,
            'dom_to_event_idx': torch.tensor(dom_to_event_idx, dtype=torch.long),  # (total_doms,)
            'sensor_ids': torch.tensor(sensor_ids, dtype=torch.long),  # (total_doms,) - actual sensor IDs
            'event_ids': torch.stack(event_ids) if event_ids else None,
            'targets': torch.stack(targets) if targets else None,
        }
    }

    return result


def get_dataloader(
    config_path: Optional[str] = None,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_events: Optional[int] = None,
    collate_fn: str = "variable_length",
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for IceCube events.

    Args:
        config_path: Path to data_config.yaml (defaults to src/iceaggr/data/data_config.yaml)
        split: 'train' or 'test'
        batch_size: Number of events per batch
        shuffle: Whether to shuffle events
        num_workers: Number of worker processes (0 = single process)
        max_events: Optional limit on number of events
        collate_fn: "variable_length" or "dom_grouping"

    Returns:
        DataLoader with continuous batching collate function
    """
    dataset = IceCubeDataset(
        config_path=config_path, split=split, max_events=max_events
    )

    # Select collate function
    if collate_fn == "variable_length":
        collate_func = collate_variable_length
    elif collate_fn == "dom_grouping":
        collate_func = collate_with_dom_grouping
    else:
        raise ValueError(f"Unknown collate_fn: {collate_fn}. Use 'variable_length' or 'dom_grouping'")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=True,  # Faster GPU transfer
    )

    return dataloader

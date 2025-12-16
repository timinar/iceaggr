"""
Sampling strategies for IceCube datasets.

This module provides various samplers for optimizing data loading:
- BatchAwareSampler: File-based sampling for cache efficiency
- BucketBatchSampler: Combined batch-aware + length-bucketing for padded datasets
"""

import random
from collections import defaultdict
from typing import Iterator, List, Optional
import numpy as np

from torch.utils.data import Sampler

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

    def __iter__(self) -> Iterator[int]:
        # 1. Shuffle the order of batch files (epoch-level randomness)
        random.shuffle(self.batch_keys)

        # 2. For each batch file, shuffle events and yield them
        for batch_key in self.batch_keys:
            indices_in_batch = self.grouped_indices[batch_key].copy()
            random.shuffle(indices_in_batch)
            yield from indices_in_batch

    def __len__(self) -> int:
        return self.n_events


class BucketBatchSampler(Sampler):
    """
    Combined batch-aware and bucket-based sampler for efficient padded batching.

    This sampler combines two strategies:
    1. File locality: Process one parquet file at a time (cache_size=1 sufficient)
    2. Length bucketing: Within each file, group events by bucket_id for minimal padding

    Strategy per epoch:
    - Shuffle order of batch files
    - For each file:
      - Load all events from that file
      - Group events by bucket_id
      - Shuffle within each bucket
      - Yield fixed-size batches from each bucket
    - Move to next file

    Args:
        metadata: PyArrow table with 'batch_id' and 'bucket_id' columns
        batch_size: Number of events per batch
        drop_last: If True, drop incomplete batches (default: False)
    """

    def __init__(
        self,
        metadata,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Extract metadata
        batch_ids = metadata.column("batch_id").to_numpy()
        bucket_ids = metadata.column("bucket_id").to_numpy()

        # Group event indices by (batch_id, bucket_id)
        # Structure: {batch_id: {bucket_id: [event_indices]}}
        self.batch_bucket_groups = defaultdict(lambda: defaultdict(list))
        for idx, (batch_id, bucket_id) in enumerate(zip(batch_ids, bucket_ids)):
            self.batch_bucket_groups[batch_id][bucket_id].append(idx)

        self.batch_keys = list(self.batch_bucket_groups.keys())
        self.n_events = len(metadata)

        # Count total batches
        self.n_batches = 0
        for batch_id in self.batch_keys:
            for bucket_id, indices in self.batch_bucket_groups[batch_id].items():
                n_full_batches = len(indices) // batch_size
                n_remainder = len(indices) % batch_size
                self.n_batches += n_full_batches
                if not drop_last and n_remainder > 0:
                    self.n_batches += 1

        logger.info(
            f"BucketBatchSampler: {self.n_events:,} events, "
            f"{len(self.batch_keys)} files, "
            f"{self.n_batches:,} batches (batch_size={batch_size}, drop_last={drop_last})"
        )

    def __iter__(self) -> Iterator[List[int]]:
        # 1. Shuffle the order of batch files (epoch-level randomness)
        random.shuffle(self.batch_keys)

        # 2. For each batch file, process all buckets
        for batch_key in self.batch_keys:
            bucket_dict = self.batch_bucket_groups[batch_key]

            # Process each bucket within this file
            for bucket_id, indices in bucket_dict.items():
                # Shuffle events within bucket
                indices_shuffled = indices.copy()
                random.shuffle(indices_shuffled)

                # Yield batches of size batch_size
                for i in range(0, len(indices_shuffled), self.batch_size):
                    batch_indices = indices_shuffled[i : i + self.batch_size]

                    # Skip incomplete batches if drop_last=True
                    if len(batch_indices) < self.batch_size and self.drop_last:
                        continue

                    yield batch_indices

    def __len__(self) -> int:
        return self.n_batches

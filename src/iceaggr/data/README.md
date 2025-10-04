# IceCube Data Loading

High-performance PyArrow-based dataset for variable-length IceCube neutrino events.

## Quick Start

```python
from iceaggr.data import get_dataloader

# Create DataLoader
dataloader = get_dataloader(
    config_path="config.yaml",
    split="train",
    batch_size=32,
    shuffle=True,
)

# Iterate over batches
for batch in dataloader:
    pulse_features = batch["pulse_features"]       # (total_pulses, 4)
    pulse_to_event = batch["pulse_to_event_idx"]   # (total_pulses,)
    event_lengths = batch["event_lengths"]         # (batch_size,)
    targets = batch["targets"]                     # (batch_size, 2)
```

## Features

- **Variable-length events**: Full events (no subsampling), 5-178K pulses
- **Continuous batching**: No padding waste, memory-efficient
- **High throughput**: 27K events/sec, 4.6M pulses/sec
- **LRU caching**: Automatic batch file caching (configurable)
- **Zero preprocessing**: Direct from Parquet files

## Batch Structure

Batches use continuous batching (flattened representation):

```python
batch = {
    'pulse_features': Tensor (total_pulses, 4),
        # Features: [time, charge, sensor_id, auxiliary]

    'pulse_to_event_idx': Tensor (total_pulses,),
        # Maps each pulse to its event index in batch

    'event_lengths': Tensor (batch_size,),
        # Number of pulses per event

    'targets': Tensor (batch_size, 2),
        # [azimuth, zenith] (train split only)

    'event_ids': Tensor (batch_size,),
        # Original IceCube event IDs
}
```

### Example

Batch of 3 events with [10, 5, 8] pulses:

```python
pulse_features.shape  # (23, 4) - flattened
pulse_to_event_idx    # [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2,2,2,2]
event_lengths         # [10, 5, 8]
```

## API Reference

### `IceCubeDataset`

```python
dataset = IceCubeDataset(
    config_path="config.yaml",  # Path to data config
    split="train",              # "train" or "test"
    max_events=None,            # Limit events (for testing)
    cache_size=1,               # Number of batch files to cache
)

event = dataset[idx]  # Get single event
```

### `get_dataloader`

```python
dataloader = get_dataloader(
    config_path="config.yaml",
    split="train",
    batch_size=32,
    shuffle=True,
    num_workers=0,       # Multi-process workers (0=single process)
    max_events=None,
)
```

### `collate_variable_length`

```python
# Custom collate function for variable-length events
# Automatically used by get_dataloader
batch = collate_variable_length(list_of_events)
```

## Performance

See `notes/02_dataset_implementation_results.md` for detailed benchmarks.

**Key metrics** (batch_size=32, single worker):
- **27K events/sec**
- **4.6M pulses/sec**
- **0.12s for 100 batches**

## Implementation Details

- **Backend**: PyArrow for fast Parquet access
- **Caching**: LRU cache for batch files
- **Memory**: ~110 KB per batch (median case)
- **I/O**: ~380 MB/s sustained on Lustre

For technical design decisions, see `notes/01_data_loading_strategy.md`.

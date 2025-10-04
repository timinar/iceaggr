# Hierarchical Transformer Architecture (T1 + T2)

**Date**: 2025-10-04
**Implementation**: Complete
**Status**: Ready for training

## Overview

The hierarchical transformer processes IceCube neutrino events in two stages:
1. **T1 (DOM-level)**: Aggregates pulses within each DOM → DOM embeddings
2. **T2 (Event-level)**: Aggregates DOM embeddings → Direction prediction

This architecture solves the scalability problem of processing events with 10K-150K pulses by exploiting the natural hierarchical structure of the detector.

## Architecture Diagram

```
Raw Event (up to 150K pulses, 2.5K DOMs)
    ↓
[Data Loading & DOM Grouping]
    ↓
Multiple Packed Sequences (bsz × 512 pulses each)
    ↓
[T1: DOM-level Transformer] ← Processes in mini-batches (max 64 seqs)
    ↓
DOM Embeddings (total_doms × d_model)
    ↓
[EventAccumulator] ← Collects all DOMs for each event
    ↓
Complete Events (batch_size events)
    ↓
[T2: Event-level Transformer] ← Geometry-aware attention
    ↓
Direction Predictions (batch_size × 2)
    ↓
(azimuth, zenith)
```

## Stage 1: T1 (DOM-level Transformer)

**Purpose**: Aggregate variable-length pulse sequences within each DOM into fixed-size embeddings.

### Architecture

```python
DOMTransformer(
    d_model=128,        # Embedding dimension
    n_heads=8,          # Multi-head attention
    n_layers=4,         # Transformer layers
    max_seq_len=512,    # Maximum pulses per packed sequence
)
```

### Key Features

1. **DOM Packing**: Multiple sparse DOMs packed into fixed-length sequences
   - Most DOMs have 1-10 pulses → pack many DOMs per sequence
   - Rare large DOMs (>512 pulses) → truncated to 512
   - Packing efficiency: ~88% at max_seq_len=512

2. **FlexAttention with DOM Boundaries**:
   ```python
   def dom_boundary_mask(score, b, h, q_idx, kv_idx):
       # Pulses only attend within same DOM
       same_dom = dom_boundaries[b, q_idx] == dom_boundaries[b, kv_idx]
       both_valid = dom_mask[b, q_idx] * dom_mask[b, kv_idx]
       return torch.where(same_dom & both_valid.bool(), score, float('-inf'))
   ```

3. **Mini-batch Processing**: Automatically splits large batches
   - `max_batch_size=64` sequences prevents OOM
   - Extreme events (245 sequences) → 4 mini-batches of 64
   - Memory: 1.8 GB vs 93 GB OOM (50× reduction)

4. **Aggregation**: Mean pooling of pulse embeddings → DOM embedding

### Input/Output

**Input** (from `collate_dom_packing`):
- `packed_sequences`: (bsz, 512, 4) - packed pulse features
- `dom_boundaries`: (bsz, 512) - which DOM each pulse belongs to
- `dom_mask`: (bsz, 512) - valid pulse mask (1=valid, 0=padding)
- `metadata`: DOM-to-event mappings, sensor IDs

**Output**:
- `dom_embeddings`: (total_doms, d_model) - one embedding per DOM
- `metadata`: Preserved for T2 (sensor IDs, event mappings)

### Memory Characteristics

| Event Type | Pulses | DOMs | T1 Sequences | GPU Memory |
|------------|--------|------|--------------|------------|
| Typical | ~60 | ~50 | 1 | 0.3 GB |
| Large | 10K | 600 | 20 | 1.2 GB |
| Extreme | 110K | 2,015 | 245 | 1.8 GB |

## Stage 2: T2 (Event-level Transformer)

**Purpose**: Aggregate DOM embeddings from across the detector to predict neutrino direction.

### Architecture

```python
EventTransformer(
    d_model=128,        # Must match T1 output
    n_heads=8,          # Multi-head attention
    n_layers=4,         # Transformer layers
    max_doms=2048,      # Maximum DOMs per event (99.9th percentile)
)
```

### Key Features

1. **Geometry-Aware Positional Encoding**:
   ```python
   # Load IceCube sensor geometry (x,y,z)
   dom_geometry = sensor_geometry[dom_ids]  # (total_doms, 3)

   # Encode geometry to d_model
   geo_encoding = geometry_encoder(dom_geometry)  # (total_doms, d_model)

   # Add to DOM embeddings
   dom_features = dom_embeddings + geo_encoding
   ```

2. **Standard Transformer Attention**: All DOMs attend to all DOMs within event

3. **Padding-aware Processing**: Variable DOMs per event handled via padding masks

4. **Global Aggregation**: Mean pooling over DOMs → event embedding

5. **Prediction Head**: `event_embedding → (azimuth, zenith)`

### Input/Output

**Input**:
- `dom_embeddings`: (total_doms, d_model) - from T1
- `dom_ids`: (total_doms,) - sensor IDs for geometry lookup
- `dom_to_event_idx`: (total_doms,) - which event each DOM belongs to
- `batch_size`: Number of events in batch

**Output**:
- `predictions`: (batch_size, 2) - (azimuth, zenith) angles

### Geometry Information

Uses real IceCube sensor geometry from `sensor_geometry.csv`:
- 5,160 sensors (DOMs)
- (x, y, z) coordinates in meters
- Geometry provides spatial context for direction reconstruction

## Event Accumulation (Handling Extreme Events)

### The Challenge

Extreme events don't fit in a single T1 batch:
- Event with 110K pulses, 2,015 DOMs → 245 sequences
- T1 processes in 4 mini-batches of 64 sequences
- But T2 needs **all DOMs from the event** together!

### Solution: EventAccumulator

**Purpose**: Collect DOM embeddings across multiple T1 batches until events are complete.

```python
accumulator = EventAccumulator()

# Process T1 batches (may split events)
for t1_batch in t1_dataloader:
    dom_embeddings, metadata = t1_model(t1_batch)
    accumulator.add_batch(dom_embeddings, metadata)

# Get complete events for T2
for t2_batch in accumulator.get_complete_events(batch_size=32):
    predictions = t2_model(**t2_batch)
```

### How It Works

1. **Accumulation**:
   - Maintains dict: `event_id → {'dom_embeddings': [...], 'dom_ids': [...], 'target': ...}`
   - Each T1 batch may contain partial events
   - Appends DOM embeddings to corresponding events

2. **Event Assembly**:
   - When requested, concatenates all chunks for each event
   - Batches complete events for efficient T2 processing

3. **Memory Management**:
   - Only stores embeddings (d_model floats per DOM)
   - Much smaller than original pulse data
   - Extreme event: 2K DOMs × 128 dims × 4 bytes = ~1 MB

### Example Flow for Extreme Event

```
Event 12345: 110,517 pulses, 2,015 DOMs

T1 Batch 1: [Event 12345 (partial), Event 12346, ...]
  → 64 sequences → 450 DOMs from event 12345
  → Accumulator stores: {12345: [450 DOM embeddings]}

T1 Batch 2: [Event 12345 (partial), ...]
  → 64 sequences → 500 DOMs from event 12345
  → Accumulator appends: {12345: [450 + 500 = 950 DOM embeddings]}

T1 Batch 3: [Event 12345 (partial), ...]
  → 64 sequences → 550 DOMs from event 12345
  → Accumulator appends: {12345: [950 + 550 = 1500 DOM embeddings]}

T1 Batch 4: [Event 12345 (rest), Event 12347, ...]
  → 53 sequences → 515 DOMs from event 12345
  → Accumulator completes: {12345: [1500 + 515 = 2015 DOM embeddings]} ✓

T2 Processing:
  → Event 12345 now complete with all 2,015 DOMs
  → Pass to T2 for direction prediction
```

## Data Flow

### Step 1: Data Loading

```python
dataset = IceCubeDataset(split="train")
loader = DataLoader(
    dataset,
    batch_size=32,  # Number of events to load
    collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=512)
)
```

### Step 2: Collation with DOM Packing

`collate_dom_packing` converts events to packed sequences:

**Input**: List of events (each with variable pulses)
**Output**: Packed sequences with metadata

```python
{
    'packed_sequences': (bsz, 512, 4),      # Pulse features
    'dom_boundaries': (bsz, 512),           # Which DOM each pulse belongs to
    'dom_mask': (bsz, 512),                 # Valid pulse mask
    'metadata': {
        'global_dom_ids': (bsz, 512),       # Pulse-to-DOM mapping
        'total_doms': int,                  # Total DOMs in batch
        'dom_to_event_idx': (total_doms,),  # DOM-to-event mapping
        'sensor_ids': (total_doms,),        # Actual sensor IDs
        'event_ids': (n_events,),           # Event IDs
        'targets': (n_events, 2),           # Azimuth, zenith
    }
}
```

### Step 3: T1 Forward Pass

```python
dom_embeddings, metadata = t1_model(batch)
# dom_embeddings: (total_doms, 128)
```

**What happens inside**:
1. Input projection: (bsz, 512, 4) → (bsz, 512, 128)
2. FlexAttention with DOM boundaries (4 layers)
3. Layer norm
4. Aggregate pulses → DOMs via scatter_add

### Step 4: Event Accumulation

```python
accumulator.add_batch(dom_embeddings, metadata)
```

**What happens**:
- Extract unique events in batch
- For each event, store DOM embeddings and sensor IDs
- If event already exists (from previous batch), append new DOMs

### Step 5: T2 Forward Pass

```python
for t2_batch in accumulator.get_complete_events(batch_size=32):
    predictions = t2_model(
        t2_batch['dom_embeddings'],
        t2_batch['dom_ids'],
        t2_batch['dom_to_event_idx'],
        t2_batch['batch_size']
    )
```

**What happens inside**:
1. Load geometry for sensor IDs
2. Encode geometry: (total_doms, 3) → (total_doms, 128)
3. Add to DOM embeddings: `dom_features = dom_embeddings + geo_encoding`
4. Pack into batched sequences with padding
5. Transformer attention (4 layers)
6. Mean pool over DOMs → event embedding
7. Prediction head: event embedding → (azimuth, zenith)

## Performance Characteristics

### Throughput (d_model=128, 4 layers)

| Configuration | Events/sec | Latency/event |
|--------------|------------|---------------|
| Small (64 events) | 136 | 7.4 ms |
| Large (128 events) | 520 | 1.9 ms |

### Time Breakdown

- **T1 (DOM-level)**: ~42% of total time
- **T2 (Event-level)**: ~11% of total time
- **Data loading/collation**: ~47% of total time

### Memory Usage (128 events, GPU)

- **Total**: 1.12 GB
- Mostly from attention computation in T1
- T2 memory negligible (typically <100 DOMs per event in batch)

## Design Decisions

### Why Two Stages?

**Problem**: Standard transformer on 100K pulses is O(n²) = infeasible

**Insight**: Pulses naturally group by DOM → hierarchical structure

**Solution**:
1. T1 handles high-cardinality (many pulses per DOM)
2. T2 handles moderate-cardinality (up to ~2K DOMs per event)

### Why DOM Packing?

**Observation**: 50-70% of DOMs have ≤10 pulses → very sparse

**Naive approach**: One sequence per DOM → waste GPU (many sequences with <10 pulses)

**Packing approach**: Multiple sparse DOMs per sequence → 88% efficiency

### Why Mini-batch Processing?

**Problem**: Extreme events create 245 sequences → 7.6 GB attention memory → OOM

**Solution**: Process in chunks of 64 sequences → 1.8 GB peak memory

**Trade-off**: Slight overhead from multiple forward passes, but enables extreme events

### Why EventAccumulator?

**Problem**: T1 batches by sequences, T2 needs complete events

**Alternative rejected**: Batch T1 by events → forces serialization for large events

**EventAccumulator**: Decouple T1 and T2 batching strategies

## Configuration Recommendations

### For Training

```python
t1_config = {
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 4,
    'max_seq_len': 512,      # Covers 99.9% of DOMs
    'max_batch_size': 64,    # Prevents OOM
}

t2_config = {
    'd_model': 128,          # Must match T1
    'n_heads': 8,
    'n_layers': 4,
    'max_doms': 2048,        # 99.9th percentile
}

data_config = {
    'batch_size': 32,        # Events per batch
    'max_seq_len': 512,      # For collation
}
```

### For Inference

Same config, but consider:
- Larger `batch_size` if memory allows
- Smaller models (d_model=64, n_layers=2) for faster inference

## Limitations and Future Work

### Current Limitations

1. **DOM truncation**: DOMs >512 pulses are truncated (affects <0.1% of DOMs)
   - Could use reservoir sampling or chunking

2. **Fixed max_seq_len**: Set at collation time
   - Could make dynamic based on batch statistics

3. **No sparse DOM fast path**: 1-pulse DOMs still go through attention
   - Could add lightweight bypass for single-pulse DOMs (~50% of DOMs)

### Potential Optimizations

1. **Flash Attention**: Use FlashAttention-2 for T2 (faster attention)

2. **Gradient Checkpointing**: Trade compute for memory in T1

3. **Mixed Precision**: FP16/BF16 training for 2× speedup

4. **DOM Subsampling**: For events >2K DOMs, subsample to fixed budget

## Testing

### Unit Tests

- `tests/unit/test_dom_packing.py`: T1 with packing (7 tests)
- `tests/unit/test_event_transformer.py`: T2 and EventAccumulator (10 tests)

### Integration Tests

- `tests/integration/test_e2e_pipeline.py`: Full T1→T2 pipeline (4 tests)
- Tests single-batch, multi-batch, gradient flow, extreme events

### Benchmarks

- `scripts/benchmark_dom_packing.py`: T1 memory on extreme events
- `scripts/benchmark_e2e_pipeline.py`: Full pipeline throughput

All tests passing ✅

## References

**Implementation**:
- T1: `src/iceaggr/models/dom_transformer.py`
- T2: `src/iceaggr/models/event_transformer.py`
- Collation: `src/iceaggr/data/dataset.py::collate_dom_packing`

**Design Docs**:
- DOM aggregation: `notes/dom_aggregation_architecture.md`
- FlexAttention benchmarks: `notes/flex_attention_benchmark_results.md`
- Memory constraints: `personal_notes/03_t1_memory_constraints.md`

# DOM Aggregation Architecture for Hierarchical Transformer

**Date**: 2025-10-04
**Status**: Design - Ready for Implementation
**Updated**: Post-benchmarking with real data

## Problem Statement

Build a hierarchical transformer to predict neutrino direction from IceCube detector pulses:
1. **T1 (DOM-level)**: Aggregate pulses within each DOM → DOM embeddings
2. **T2 (Event-level)**: Aggregate DOM embeddings → Direction prediction (azimuth, zenith)

## Architecture

```
Input: Batch of events (variable DOMs, variable pulses)
                    ↓
      [Group pulses by (event, DOM)]
                    ↓
         ┌─────────────────────┐
         │  T1: DOM-level      │
         │  FlexAttention      │
         │  → DOM embeddings   │
         └─────────────────────┘
                    ↓
      [DOM embeddings per event]
                    ↓
         ┌─────────────────────┐
         │  T2: Event-level    │
         │  Standard Attention │
         │  → Direction        │
         └─────────────────────┘
                    ↓
         Output: (azimuth, zenith)
```

## Data Characteristics (Real IceCube)

**Key insight from benchmarking**: Real data is extremely sparse!

- **Median pulses/DOM**: 1 (not 9!)
- **Max pulses/DOM**: 6-27 per batch (not 1000!)
- **Typical batch**: ~300-1300 pulses, ~200-900 DOMs
- **Distribution**: Highly uniform, not heavy-tailed

**Implication**: FlexAttention is perfect for this workload (5.86x faster than loops).

## Data Structure (Implemented)

Continuous batching with DOM grouping - see `src/iceaggr/data/dataset.py::collate_with_dom_grouping()`

```python
batch = {
    # Pulse-level (for T1)
    'pulse_features': (total_pulses, 4),        # Flattened pulses
    'pulse_to_dom_idx': (total_pulses,),        # Which DOM each pulse belongs to
    'dom_pulse_counts': (total_doms,),          # Pulses per DOM

    # DOM-level (for T2)
    'dom_to_event_idx': (total_doms,),          # Which event each DOM belongs to
    'dom_ids': (total_doms,),                   # Original sensor IDs
    'event_dom_counts': (batch_size,),          # DOMs per event

    # Event-level
    'targets': (batch_size, 2),                 # Azimuth, zenith
    'event_ids': (batch_size,)
}
```

**Critical**: `pulse_to_dom_idx` maps to unique (event, DOM) pairs, not just DOM IDs. This prevents incorrect aggregation across events.

## T1: DOM-level Transformer (Recommended)

**Use FlexAttention (dense)** - 5.86x faster than alternatives on real data.

See `notes/04_flex_attention_benchmark_results.md` for benchmarking details.

### Key Design Decisions

1. **Attention mechanism**: FlexAttention with document masking
   - No BlockMask (causes OOM)
   - Dense masking: `pulse_to_dom_idx[q] == pulse_to_dom_idx[kv]`

2. **Aggregation**: Mean pooling over pulses within each DOM

3. **Model size**: Start with d_model=128, 4 layers, 8 heads

### Implementation Status

- ✅ Collate function: `collate_with_dom_grouping()`
- ✅ Benchmark script: `scripts/benchmark_flex_attention.py`
- ⏸️ T1 model: Not yet implemented
- ⏸️ Aggregation helper: Not yet implemented

## T2: Event-level Transformer

Aggregate DOM embeddings to predict neutrino direction.

### Input
- DOM embeddings from T1: `(total_doms, d_model_t1)`
- DOM geometry: Learned position embeddings from `dom_ids`
- Event grouping: `dom_to_event_idx`, `event_dom_counts`

### Architecture
- Standard transformer (not FlexAttention)
- Max ~2000 DOMs per event → manageable sequence length
- Add geometry encoding (DOM x, y, z positions)
- Output head: 2D (azimuth, zenith)

### Model size
Start with d_model=256, 6 layers, 8 heads

### Implementation Status
- ⏸️ Not yet implemented

## Memory Estimates

### T1 (batch_size=16, real data)
```
Pulse embeddings: ~650 KB
FlexAttention: ~2-3 GB peak
DOM embeddings: ~450 KB
Total: ~3 GB ✅
```

### T2 (batch_size=16, ~900 DOMs)
```
DOM embeddings: ~450 KB
Position embeddings: ~150 KB
Attention computation: ~5 GB
Total: ~6 GB ✅
```

**Combined T1+T2**: ~10 GB per forward pass (easily fits on H100)

## Next Steps

1. **Implement T1**: DOM-level transformer with FlexAttention
   - Model class: `src/iceaggr/models/dom_transformer.py`
   - Aggregation helper: Extract from benchmark script

2. **Implement T2**: Event-level transformer
   - Model class: `src/iceaggr/models/event_transformer.py`
   - Geometry encoding

3. **Full model**: Combine T1 + T2
   - Model class: `src/iceaggr/models/hierarchical_transformer.py`

4. **Training**: Loss function, optimizer, training loop
   - Angular distance loss
   - W&B logging

## References

- FlexAttention benchmarks: `notes/04_flex_attention_benchmark_results.md`
- Collate function: `src/iceaggr/data/dataset.py::collate_with_dom_grouping()`
- Benchmark script: `scripts/benchmark_flex_attention.py`
- PyTorch FlexAttention: https://pytorch.org/blog/flexattention/

## Key Lessons

1. **Always benchmark on real data**: Synthetic assumptions were completely wrong
2. **Real data is sparse**: Design for sparsity, not worst-case
3. **FlexAttention wins on sparse data**: 5.86x faster than loops
4. **Skip BlockMask**: Dense masking is simpler and faster for our use case

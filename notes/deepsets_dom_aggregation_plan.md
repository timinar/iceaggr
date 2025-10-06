# DeepSets DOM Aggregation: Architecture Plan

**Date**: 2025-10-06
**Status**: Planning
**Branch**: `experiment/deepsets-dom-aggregation`

## Motivation

Current FlexAttention approach for T1 is inefficient:
- Pulse/DOM distribution: median=1, long tail to 100s
- FlexAttention block masks are mostly diagonal → wasted compute
- Python loops in collate function are a bottleneck
- T2 already works with first-pulse-only → suggests simple aggregation may suffice

## Core Idea: DeepSets with Relative Encodings

Use permutation-invariant DeepSets architecture for DOM-level pulse aggregation, enhanced with relative temporal/spatial features to capture physics (rescattering vs direct Cherenkov hits).

### Architecture

```
For each DOM:
  pulses (N x D_pulse)
    ↓
  Add relative encodings (time, charge, auxiliary)
    ↓
  MLP_encode (D_pulse → D_latent)
    ↓
  Pooling (mean + max + charge-weighted)
    ↓
  MLP_decode (D_pooled → D_dom_embedding)
    ↓
  DOM embedding → T2
```

### Relative Encodings

Enhance each pulse with intra-DOM context:

1. **Temporal**:
   - `Δt_first = time - first_pulse_time` (direct vs scattered)
   - `Δt_median = time - median_pulse_time` (position in sequence)
   - Normalized by time spread in DOM

2. **Charge**:
   - `charge / total_charge` (relative importance)
   - `charge / max_charge` (brightness ratio)
   - `log(charge_rank / N)` (ordering information)

3. **Auxiliary** (optional):
   - `pulse_index / N` (sequence position if sorted by time)
   - `is_first_pulse`, `is_brightest_pulse` (binary flags)

**Rationale**: Pooling loses ordering, but relative features preserve temporal/intensity relationships.

### Pooling Strategy

Multi-head pooling to preserve different types of information:

```python
# Three pooling heads
mean_pool = mean(encoded_pulses)           # Average signal
max_pool = max(encoded_pulses)             # Strongest feature
charge_weighted = Σ(charge_i * pulse_i) / Σ(charge_i)  # Physics-weighted

# Concatenate
pooled = concat([mean_pool, max_pool, charge_weighted])  # 3 * D_latent

# Final projection
dom_embedding = MLP_decode(pooled)  # → D_dom_embedding
```

### Benefits

✅ **Efficient**: Single forward pass, no attention overhead
✅ **Scalable**: Handles 1 to 1000s pulses with same code
✅ **Physics-informed**: Relative encodings capture temporal structure
✅ **Hardware-friendly**: Pure tensor ops, easy to JIT compile

## Implementation Plan

### Phase 1: Model Architecture

**Files to create**:
- `src/iceaggr/models/deepsets_dom_encoder.py`
  - `RelativeEncoder`: Computes relative features
  - `DeepSetsDOMEncoder`: Main module

**Key design**:
```python
class DeepSetsDOMEncoder(nn.Module):
    def __init__(self, d_pulse, d_latent, d_dom_embedding):
        self.relative_encoder = RelativeEncoder()
        self.mlp_encode = MLP(d_pulse + d_relative → d_latent)
        self.mlp_decode = MLP(3 * d_latent → d_dom_embedding)

    def forward(self, pulses, charges, times, dom_sizes):
        # Compute relative encodings
        rel_features = self.relative_encoder(pulses, charges, times, dom_sizes)

        # Encode pulses
        encoded = self.mlp_encode(concat([pulses, rel_features]))

        # Multi-head pooling (using scatter ops for batching)
        mean_pool = scatter_mean(encoded, dom_ids)
        max_pool = scatter_max(encoded, dom_ids)
        charge_pool = scatter_weighted(encoded, charges, dom_ids)

        # Decode to DOM embeddings
        dom_emb = self.mlp_decode(concat([mean_pool, max_pool, charge_pool]))
        return dom_emb
```

### Phase 2: Efficient Data Collation

**Challenges**:
- Variable pulses/DOM (1 to 1000s)
- Need to batch efficiently without Python loops

**Solution**: Flat pulse representation + scatter ops

```python
# Collate strategy
batch = {
    'pulse_features': [N_total_pulses, D_pulse],  # All pulses flattened
    'pulse_charges': [N_total_pulses],
    'pulse_times': [N_total_pulses],
    'dom_ids': [N_total_pulses],  # Which DOM each pulse belongs to
    'dom_sizes': [N_doms],  # Number of pulses per DOM
    'dom_geometry': [N_doms, 3],  # x,y,z for T2
    'event_ids': [N_doms],  # Which event each DOM belongs to
}
```

**Files to modify**:
- `src/iceaggr/data/icecube_dataset.py`
  - New `collate_fn_deepsets()`
  - Pure PyTorch/NumPy ops only

### Phase 3: Integration with T2

**Minimal changes needed**:
- T2 already expects `[N_doms, D_dom_embedding]`
- Just swap T1 → DeepSets encoder
- Keep T2 transformer unchanged

**Files to modify**:
- `src/iceaggr/models/hierarchical_model.py`
  - Replace `DOMTransformer` with `DeepSetsDOMEncoder`

### Phase 4: Training & Benchmarks

**Baselines to compare**:
1. First-pulse-only (existing)
2. DeepSets (no relative encodings)
3. DeepSets (with relative encodings) ← target
4. (Optional) FlexAttention (if we can fix it)

**Metrics**:
- Angular error (physics)
- Throughput (samples/sec)
- Memory usage
- Training speed (time to convergence)

## Implementation Checklist

- [ ] Create branch `experiment/deepsets-dom-aggregation` from main
- [ ] Implement `RelativeEncoder` module
- [ ] Implement `DeepSetsDOMEncoder` module
- [ ] Write unit tests for relative encodings
- [ ] Write unit tests for pooling operations
- [ ] Implement new `collate_fn_deepsets()`
- [ ] Benchmark collate function vs current implementation
- [ ] Integrate with existing T2 transformer
- [ ] Create experiment config `experiments/deepsets_baseline/config.yaml`
- [ ] Run ablation: first-pulse vs no-rel-enc vs full DeepSets
- [ ] Document results

## Expected Outcomes

**If successful**:
- 5-10x faster data loading (no Python loops)
- 3-5x faster forward pass (no FlexAttention overhead)
- Comparable or better angular error vs first-pulse baseline
- Simpler, more maintainable codebase

**If unsuccessful**:
- DeepSets too simple → need temporal modeling
- Fall back to: Temporal Conv1D or small GRU for multi-pulse DOMs
- Keep stratified approach (identity for single-pulse DOMs)

## Open Questions

1. **Relative encoding dimensionality**: How many features? (Start with 6-8)
2. **MLP depth**: How deep? (Start with 2-3 layers)
3. **Pooling heads**: Do we need all three? (Ablate)
4. **Normalization**: BatchNorm vs LayerNorm in MLPs? (Experiment)

## Next Steps

1. Create branch
2. Implement core modules with tests
3. Benchmark collate function in isolation
4. End-to-end integration test
5. Small-scale training run (10K events)
6. Full training run (900K events)

---

**References**:
- DeepSets paper: https://arxiv.org/abs/1703.06114
- Current architecture: `notes/t1_t2_data_flow_complete.md`
- First-pulse baseline: Works but suboptimal

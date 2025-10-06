# DeepSets Pipeline Analysis - Data Flow & Shapes

**Date**: 2025-10-06
**Status**: DEBUG - Training not working, investigating data flow

## Problem Statement

Training is:
1. **Very slow**: ~6 sec/batch on H100 (expected <1 sec)
2. **Not learning**: Loss flat at 1.566 after 200 batches (random baseline is π/2 ≈ 1.57)

## Complete Data Flow Analysis

### 1. Dataset: `IceCubeDataset.__getitem__(idx)`

**Input**: Event index `idx`

**Process**:
```python
# Load from parquet batch
event_table = batch_table.slice(first_pulse, n_pulses)
time = event_table.column("time").to_numpy()
charge = event_table.column("charge").to_numpy()
sensor_id = event_table.column("sensor_id").to_numpy()  # ⚠️ int16 (0-5159)
auxiliary = event_table.column("auxiliary").to_numpy()  # bool
```

**Output** (single event):
```python
{
    'pulse_features': (n_pulses, 4),  # [time, charge, sensor_id, auxiliary]
    'event_id': scalar,
    'n_pulses': scalar,
    'target': (2,)  # [azimuth, zenith]
}
```

**Shapes example**:
- Event with 100 pulses: `pulse_features` = (100, 4)
- Typical ranges:
  - time: ~0-30000
  - charge: ~0.1-100+
  - sensor_id: 0-5159 (integer)
  - auxiliary: 0 or 1

---

### 2. Collation: `collate_deepsets(batch)`

**Input**: List of `batch_size` event dicts

**Process**:
```python
# Flatten all pulses
all_pulse_features = torch.cat(pulse_features_list)  # (total_pulses, 4)
pulse_to_event = torch.cat([torch.full((n,), event_idx) for event_idx, n in enumerate(event_lengths)])

# Extract sensor IDs (column 2)
sensor_ids = all_pulse_features[:, 2].long()  # (total_pulses,)

# Create DOM hash: event_id * 5160 + sensor_id
max_sensor_id = 5160
dom_hash = pulse_to_event * max_sensor_id + sensor_ids

# Find unique DOMs
unique_dom_hashes, inverse_indices = torch.unique(dom_hash, return_inverse=True)
pulse_to_dom_idx = inverse_indices  # Maps each pulse to its DOM
```

**Output** (batch):
```python
{
    'pulse_features': (total_pulses, 4),        # All pulses flattened
    'pulse_to_dom_idx': (total_pulses,),        # Which DOM each pulse belongs to
    'num_doms': int,                             # Total unique DOMs in batch
    'dom_to_event_idx': (num_doms,),            # Which event each DOM belongs to
    'dom_ids': (num_doms,),                      # Original sensor IDs
    'event_dom_counts': (batch_size,),           # DOMs per event
    'event_ids': (batch_size,),
    'batch_size': int,
    'targets': (batch_size, 2)                   # [azimuth, zenith]
}
```

**Shapes example** (batch_size=256):
- total_pulses: ~50,000 (varies)
- num_doms: ~20,000 (varies)
- Typical event has ~80 DOMs with ~200 pulses total

**⚠️ POTENTIAL ISSUE**: DOM hash calculation
- `pulse_to_event * 5160 + sensor_ids`
- If `pulse_to_event` goes up to 255 (batch_size-1), hash goes up to 255 * 5160 + 5159 = 1,321,159
- This is fine for uniqueness, but might be inefficient

---

### 3. Model Forward: `HierarchicalIceCubeModel.forward(batch)`

#### 3.1 Input Normalization

```python
pulse_features_normalized = self._normalize_pulse_features(pulse_features)
```

**Normalization**:
```python
time_norm = (time - 1e4) / 3e4          # ⚠️ CHECK: time ranges 0-30K, so this gives negative values!
charge_norm = torch.log10(charge + 1e-8) / 3.0
sensor_id_norm = sensor_id / 5160.0
auxiliary_norm = auxiliary  # 0 or 1
```

**⚠️ CRITICAL ISSUE FOUND**: Time normalization!
- If time ranges from 0 to 30,000:
- `(0 - 10000) / 30000 = -0.33` (negative!)
- `(30000 - 10000) / 30000 = 0.67`
- This might not be the intended range. Should check actual time distribution in data.

**Output**: (total_pulses, 4) normalized features

---

#### 3.2 DeepSets DOM Encoder

**3.2.1 Relative Encoding**:
```python
rel_features = self.relative_encoder(times, charges, dom_idx, num_doms)
```

**Process**:
- Extracts time and charge from normalized features
- Computes per-DOM statistics using scatter ops:
  - `first_times = scatter_min_1d(times, dom_idx, ...)`
  - `mean_times = scatter_mean_1d(times, dom_idx, ...)`
  - `time_std = scatter_std_1d(times, dom_idx, ...)`
  - Similar for charges

**⚠️ POTENTIAL ISSUE**: Using normalized times/charges for relative encoding!
- We normalize features first, THEN extract times/charges
- But relative encoder expects raw values to compute meaningful deltas
- **BUG**: `times = pulse_features[:, 0]` uses NORMALIZED time!

**Output**: (total_pulses, 6) relative features

**3.2.2 Encoding**:
```python
pulse_with_rel = torch.cat([pulse_features_normalized, rel_features], dim=1)  # (total_pulses, 10)
encoded = self.mlp_encode(pulse_with_rel)  # (total_pulses, d_latent)
```

**3.2.3 Pooling**:
```python
mean_pool = scatter_mean(encoded, pulse_to_dom_idx, dim=0, dim_size=num_doms)
max_pool = scatter_max(encoded, pulse_to_dom_idx, dim=0, dim_size=num_doms)[0]
charge_pool = scatter_weighted(encoded, charges, pulse_to_dom_idx, ...)
```

**⚠️ ANOTHER BUG**: Charge-weighted pooling uses normalized charges!
- Should use raw charges for weighting

**Output**: (num_doms, d_dom_embedding)

---

#### 3.3 Geometry Lookup

```python
geometry = self._lookup_geometry(dom_ids)  # (num_doms, 3)
```

**Process**:
- Uses `dom_ids` (original sensor IDs 0-5159)
- Direct index: `geometry_table[dom_ids.long()]`
- Already normalized by /500.0

**Output**: (num_doms, 3) normalized geometry

---

#### 3.4 Event Transformer

```python
predictions = self.event_transformer(dom_embeddings, dom_to_event_idx, batch_size, geometry)
```

**Process**:
1. Project DOM embeddings: (num_doms, d_input) → (num_doms, d_model)
2. Add geometry encoding if enabled
3. **Pack into batch format** - THIS IS CRITICAL:

```python
# Count DOMs per event
dom_counts = torch.bincount(dom_to_event_idx, minlength=batch_size)
max_doms = dom_counts.max().item()

# Create padded tensor
batch_tensor = torch.zeros(batch_size, max_doms + 1, d_model)  # +1 for CLS
attention_mask = torch.ones(batch_size, max_doms + 1, dtype=torch.bool)

# Fill in DOMs
current_idx = torch.zeros(batch_size, dtype=torch.long)
for i in range(num_doms):
    event_id = dom_to_event_idx[i].item()
    pos = current_idx[event_id].item() + 1
    batch_tensor[event_id, pos, :] = x[i]
    attention_mask[event_id, pos] = False  # False = attend
    current_idx[event_id] += 1
```

**⚠️ PERFORMANCE ISSUE**: Python loop over `num_doms` (~20,000)!
- This is extremely slow
- Should be vectorized

4. Apply transformer with attention mask
5. Extract CLS token: `cls_output = transformed[:, 0, :]`
6. Predict: `predictions = self.output_head(cls_output)`

**Output**: (batch_size, 2) [azimuth, zenith]

---

### 4. Loss Computation

```python
loss = model.compute_loss(predictions, targets)
```

**Process**:
```python
# Extract angles
az_pred, zen_pred = predictions[:, 0], predictions[:, 1]
az_true, zen_true = targets[:, 0], targets[:, 1]

# Compute angular distance
sa1 = torch.sin(az_true)
ca1 = torch.cos(az_true)
sz1 = torch.sin(zen_true)
cz1 = torch.cos(zen_true)
# ... same for predictions

scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)
scalar_prod = torch.clip(scalar_prod, -1, 1)
loss = torch.mean(torch.abs(torch.arccos(scalar_prod)))
```

**⚠️ ISSUE**: Model outputs unbounded values!
- Azimuth should be in [0, 2π]
- Zenith should be in [0, π]
- But we're not constraining model outputs
- This means sin/cos of arbitrary values

---

## Critical Issues Found

### 🔴 HIGH PRIORITY

1. **Relative encoder uses normalized values** (line 164 in hierarchical_model.py)
   ```python
   # BUG: Uses normalized time/charge for relative encoding
   pulse_features_normalized = self._normalize_pulse_features(pulse_features)
   # Then RelativeEncoder extracts times/charges from normalized features!
   ```
   **Fix**: Pass raw pulse features to DeepSets, normalize only at the very end OR separate normalization

2. **Python loop in EventTransformer** (event_transformer.py:178-186)
   ```python
   for i in range(num_doms):  # ~20,000 iterations!
       event_id = dom_to_event_idx[i].item()
       ...
   ```
   **Fix**: Vectorize using scatter operations

3. **Model outputs unconstrained angles**
   - Network can output any float
   - Need to add activation: `tanh` → scale to [0, 2π] and [0, π]
   **Fix**: Add output activation or use von Mises distribution

### 🟡 MEDIUM PRIORITY

4. **Time normalization might be wrong**
   - Check actual time distribution in data
   - Current formula: `(time - 1e4) / 3e4`
   - If time < 10000, this is negative

5. **Charge-weighted pooling uses normalized charges**
   - Physics meaning is lost
   - Should weight by raw charge magnitudes

---

## Performance Analysis

**Current speed**: ~6 sec/batch (256 events)

**Bottlenecks**:
1. Python loop in EventTransformer: ~4-5 sec (estimated)
2. Data loading: ~0.5 sec
3. Actual computation: ~1 sec

**Expected speed after fixes**: <1 sec/batch

---

## Next Steps

1. **Immediate**: Run single-batch overfit test
2. **Fix relative encoding**: Use raw values, not normalized
3. **Fix EventTransformer loop**: Vectorize
4. **Fix output activation**: Constrain angles
5. **Re-test**: Should see learning + 5-6x speedup

# T2FirstPulseModel Architecture Documentation

## Overview
The T2FirstPulseModel is a **diagnostic transformer** that predicts neutrino direction by:
1. Extracting the first pulse from each DOM
2. Adding geometry (spatial position) information
3. Using transformer attention to aggregate across all DOMs in an event
4. Predicting a 3D unit vector representing neutrino direction

---

## Step-by-Step Data Flow

### **STEP 1: INPUT DATA** (from DataLoader)

**Batch Structure:**
```python
batch = {
    'packed_sequences': Tensor(bsz=5, max_seq_len=512, features=4),
    'dom_boundaries': Tensor(bsz=5, max_seq_len=512),
    'dom_mask': Tensor(bsz=5, max_seq_len=512),
    'metadata': {
        'total_doms': 1695,  # Total DOMs across all events in batch
        'global_dom_ids': Tensor(bsz=5, max_seq_len=512),  # Which DOM each pulse belongs to
        'sensor_ids': Tensor(total_doms=1695),  # Physical sensor ID (0-5159)
        'dom_to_event_idx': Tensor(total_doms=1695),  # Which event each DOM belongs to
        'event_ids': Tensor(batch_size=32),  # Actual event IDs
        'targets': Tensor(batch_size=32, 2),  # Ground truth (azimuth, zenith)
    }
}
```

**What is "packing"?**
- Multiple sparse DOMs are packed into fixed-length sequences (512)
- Example: Batch has 32 events → ~1695 DOMs → packed into 5 sequences of 512 positions each
- `dom_boundaries[i, j]` tells which DOM position `j` in sequence `i` belongs to
- `dom_mask[i, j] = 1` means valid pulse, `0` means padding

**Raw Features (per pulse):**
- `time`: Time of pulse arrival (~0-77,000 ns)
- `charge`: Photon charge (~0-2700)
- `sensor_id`: Physical DOM ID (0-5159)
- `auxiliary`: Binary flag (0 or 1)

---

### **STEP 2: EXTRACT FIRST PULSE PER DOM** (`_extract_first_pulses`)

**Input:**
- `packed_sequences`: (5, 512, 4) - packed pulse features
- `dom_boundaries`: (5, 512) - DOM assignments
- `dom_mask`: (5, 512) - valid pulse mask

**Normalization** (CRITICAL for stability!):
```python
time_normalized = (time - 1e4) / 3e4          # Scale to ~[-0.33, 2.23]
charge_normalized = log10(charge + 1e-8) / 3.0  # Scale to ~[-2.67, 1.14]
sensor_id_normalized = sensor_id / 5160.0     # Scale to [0, 1]
auxiliary = auxiliary                          # Already {0, 1}
```

**Extraction Logic:**
```python
# Flatten: (5, 512, 4) → (2560, 4)
# For each valid pulse in order:
for pulse in valid_pulses:
    dom_id = pulse's global_dom_id
    if not seen[dom_id]:
        first_pulses[dom_id] = pulse_features
        seen[dom_id] = True
```

**Output:**
- `first_pulse_features`: **(1695, 4)** - normalized first pulse per DOM
- `sensor_ids`: **(1695,)** - sensor IDs for each DOM

---

### **STEP 3: PROJECT PULSE FEATURES** (`pulse_projection`)

**Architecture:**
```python
self.pulse_projection = nn.Linear(4, 128)
```

**Computation:**
```python
dom_embeddings = pulse_projection(first_pulse_features)
# Input:  (1695, 4)
# Output: (1695, 128) - DOM embeddings from pulse features
```

---

### **STEP 4: ADD GEOMETRY ENCODING** (`geometry_encoder`)
***IT: Outdated: xyz positions should be included on the previous step instead of this separate encoding.***

**Load Geometry:**
```python
self.sensor_geometry: Tensor(5160, 3)  # (x, y, z) for all 5160 sensors
# Example: sensor 100 → position (-256.14, -521.08, -350.49) meters
```

**Get Geometry for Active DOMs:**
```python
dom_geometry = sensor_geometry[sensor_ids]  # (1695, 3)
dom_geometry_normalized = dom_geometry / 500.0  # Normalize by detector scale
```

**Encode Geometry:**
```python
self.geometry_encoder = nn.Sequential(
    nn.Linear(3, 64),    # (3 → 64)
    nn.GELU(),
    nn.Linear(64, 128),  # (64 → 128)
)

geo_encoding = geometry_encoder(dom_geometry_normalized)
# Input:  (1695, 3)
# Output: (1695, 128) - Positional encoding from geometry
```

**Combine:**
```python
dom_features_combined = dom_embeddings + geo_encoding
# (1695, 128) + (1695, 128) = (1695, 128)
```

Now each DOM has a **128-dim feature vector** that contains:
- Pulse information (time, charge, sensor_id, auxiliary)
- Spatial position (x, y, z)

---

### **STEP 5: PACK EVENTS** (`_pack_events`)

**Problem:**
- We have 1695 DOMs distributed across 32 events
- Transformer needs batched sequences (batch_size, seq_len, d_model)

**Solution - Pack by Event:**
```python
# Count DOMs per event
event 0: 50 DOMs
event 1: 48 DOMs
...
event 31: 55 DOMs

max_doms_in_batch = 60  # Max across these 32 events
```

**Create Padded Tensor:**
```python
event_sequences = zeros(32, 60, 128)  # (batch_size, max_doms, d_model)
padding_mask = ones(32, 60, dtype=bool)  # True = padding

for event_idx in range(32):
    event_doms = dom_features_combined[dom_to_event_idx == event_idx]  # e.g., (50, 128)
    n_doms = 50

    event_sequences[event_idx, :50, :] = event_doms  # Fill first 50 positions
    padding_mask[event_idx, :50] = False  # Mark as valid (not padding)
    # Positions 50-59 remain zeros with padding_mask=True
```

**Output:**
- `event_sequences`: **(32, 60, 128)** - padded DOM features per event
- `padding_mask`: **(32, 60)** - True for padding, False for real DOMs

---

### **STEP 6: TRANSFORMER** (`self.transformer`)

**Architecture:**
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,           # 4 attention heads
    dim_feedforward=512,  # FFN hidden size
    dropout=0.0,
    batch_first=True,
    norm_first=True,   # Pre-LayerNorm
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
```

**Forward Pass:**
```python
transformed = self.transformer(
    event_sequences,  # (32, 60, 128)
    src_key_padding_mask=padding_mask  # (32, 60) - mask out padding
)
# Output: (32, 60, 128)
```

**What Happens Inside (per layer):**

1. **Multi-Head Self-Attention:**
   ```python
   # For each head (4 heads, each 32-dim):
   Q = event_sequences @ W_q  # (32, 60, 32)
   K = event_sequences @ W_k  # (32, 60, 32)
   V = event_sequences @ W_v  # (32, 60, 32)

   scores = (Q @ K.T) / sqrt(32)  # (32, 60, 60) - attention scores
   scores = scores.masked_fill(padding_mask, -inf)  # Mask padding
   attention_weights = softmax(scores, dim=-1)  # (32, 60, 60)

   output = attention_weights @ V  # (32, 60, 32)
   # Concatenate 4 heads → (32, 60, 128)
   ```
   **Effect:** Each DOM attends to all other DOMs in the same event, learning spatial relationships

2. **Feed-Forward Network:**
   ```python
   ffn_output = GELU(Linear_1(output))  # (32, 60, 512)
   ffn_output = Linear_2(ffn_output)     # (32, 60, 128)
   ```

3. **Residual + LayerNorm** (pre-LN, so LN happens before attention)

**This repeats 4 times** (4 transformer layers)

**Memory Issue Here!**
- Attention matrix: (batch_size, max_doms, max_doms)
- For batch_size=512, max_doms=2500: **(512, 2500, 2500)** in fp16 = **~6.4GB**
- For extreme events (>2048 DOMs), this explodes → **OOM!**

---

### **STEP 7: GLOBAL AGGREGATION** (Mean Pooling)

```python
# Mask out padding
mask = ~padding_mask.unsqueeze(-1)  # (32, 60, 1)
masked_features = transformed * mask  # (32, 60, 128) - zeros out padding

# Sum across DOMs
event_embedding = masked_features.sum(dim=1)  # (32, 128)

# Count valid DOMs per event
valid_doms = mask.sum(dim=1).clamp(min=1)  # (32, 1)

# Mean pool
event_embedding = event_embedding / valid_doms  # (32, 128)
```

Each event is now represented by a **single 128-dim vector** (average of all DOM features)

---

### **STEP 8: PREDICTION HEAD**

```python
self.prediction_head = nn.Sequential(
    nn.Linear(128, 128),  # (32, 128) → (32, 128)
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(128, 3),    # (32, 128) → (32, 3)
)

vector = prediction_head(event_embedding)  # (32, 3) - raw 3D vector
```

**Normalize to Unit Sphere:**
```python
norm = sqrt(sum(vector**2, dim=1, keepdim=True))  # (32, 1)
unit_vector = vector / (norm + 1e-8)  # (32, 3)
```

**Output:**
- `unit_vector`: **(32, 3)** - normalized direction vectors (x, y, z)

---

### **STEP 9: LOSS COMPUTATION**

**Convert Targets to Unit Vectors:**
```python
target_angles = batch['metadata']['targets']  # (32, 2) - (azimuth, zenith)

# Convert angles to unit vectors:
target_vectors = [
    cos(azimuth) * sin(zenith),  # x
    sin(azimuth) * sin(zenith),  # y
    cos(zenith),                 # z
]  # (32, 3)
```

**Compute Angular Distance:**
```python
def angular_dist_score_unit_vectors(n_true, n_pred):
    # Dot product
    scalar_prod = sum(n_true * n_pred, dim=1)  # (32,)

    # Clip to [-1, 1] for numerical stability
    scalar_prod = clip(scalar_prod, -1, 1)

    # Angular distance = arccos(dot_product)
    angles = abs(arccos(scalar_prod))  # (32,) - radians

    return mean(angles)  # Scalar - mean angular error
```

**Loss Value:**
```python
loss = AngularDistanceLoss(use_unit_vectors=True)
loss_value = loss(unit_vector, target_vectors)  # Scalar, e.g., 1.2 radians (~69 degrees)
```

---

## Shape Summary Table

| Stage | Tensor Name | Shape | Description |
|-------|------------|-------|-------------|
| **Input** | `packed_sequences` | (5, 512, 4) | Packed pulse features |
| | `metadata['total_doms']` | 1695 | Total DOMs in batch |
| | `metadata['event_ids']` | (32,) | Batch size = 32 events |
| **Extract** | `first_pulse_features` | (1695, 4) | First pulse per DOM (normalized) |
| **Project** | `dom_embeddings` | (1695, 128) | Pulse features → d_model |
| **Geometry** | `geo_encoding` | (1695, 128) | Position encoding |
| **Combined** | `dom_features_combined` | (1695, 128) | Pulse + geometry |
| **Pack** | `event_sequences` | (32, 60, 128) | Padded by event |
| | `padding_mask` | (32, 60) | Mask for padding |
| **Transformer** | `transformed` | (32, 60, 128) | After 4 layers |
| **Pool** | `event_embedding` | (32, 128) | Mean-pooled per event |
| **Predict** | `vector` | (32, 3) | Raw 3D predictions |
| | `unit_vector` | (32, 3) | Normalized directions |
| **Loss** | `loss` | () | Scalar angular distance |

---

## Key Issues Identified

### 1. **Memory Explosion in Transformer**
- Attention matrix: O(max_doms²)
- Events with >2048 DOMs → OOM
- **Solution needed:** DOM subsampling or hierarchical attention

### 2. **Inefficient First Pulse Extraction**
- Python loop over all pulses
- **Solution needed:** Vectorized extraction using scatter/gather

### 3. **No Validation**
- Disabled due to OOM on extreme events
- **Solution needed:** Filter or subsample validation set

### 4. **Suboptimal Feature Encoding**
- Pulse features and geometry encoded separately then added
- Better approach: Concatenate raw features + geometry, then project together
- This allows the network to learn joint representations from the start

---

## Proposed Improvements

### Architecture Changes:
1. **Combine features upfront:** `[time, charge, auxiliary, x, y, z]` → 6 features, then `Linear(6, 128)`
2. **Remove separate geometry encoder** - let the network learn geometric relationships directly
3. **Add DOM subsampling** for events >2048 DOMs
4. **Vectorize first pulse extraction** - remove Python loop

### Training Optimizations:
1. **Dynamic batching** based on total DOMs, not just event count
2. **Gradient checkpointing** to reduce memory
3. **Flash Attention** for more efficient attention computation

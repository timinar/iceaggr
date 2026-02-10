# Flat Transformer v2: Architecture, Ablations & Scale-Up

## Summary

The flat transformer v2 with nanochat-style improvements achieves **56.3°** mean angular error on IceCube direction reconstruction at 5M params / 132M events, down from **62.0°** (v1 baseline, 554K params / 10M events). The architectural improvements (RMSNorm, zero-init projections, QK norm, residual scaling) account for most of the gain; input projection mode and pulse count K have minimal impact.

## Architecture (v2)

File: `src/iceaggr/models/flat_transformer_v2.py`

Differences from v1 (`flat_transformer.py`):
- **Functional RMSNorm** (no learnable params) instead of LayerNorm
- **Zero-init output projections** (attention c_proj, FFN c_proj) — model starts as identity
- **QK norm** in attention for stability
- **ReLU²** activation in FFN (sharper than GELU)
- **No bias** in transformer linear layers
- **Per-layer residual scaling**: `x = λ_resid[i] * x + λ_x0[i] * x0` (x0 = initial embedding)
- **Post-embedding RMSNorm** after input projection

Input projection modes tested:
- `none`: identity (raw features in residual stream, zero-padded to d_model)
- `linear`: single Linear(input_dim → d_model)
- `mlp`: Linear → GELU → Linear

---

## Results

### Small Model Ablations (10M events, 10 epochs, lr=1e-4, BSZ=256)

| Run | K | Input | Head | Params | Best Val | Notes |
|-----|---|-------|------|--------|----------|-------|
| v1 baseline | 16 | linear | 128 | 554K | 62.0° | 3-epoch plateau at 87.9° |
| **v2 linear** | **16** | **linear** | **128** | **548K** | **57.0°** | Best at K=16 |
| v2 MLP | 16 | mlp | 128 | 565K | 57.6° | MLP doesn't help |
| v2 linear K41 | 41 | linear | 512 | 608K | 57.8° | More pulses don't help |
| v2 none K41 | 41 | none | 512 | 592K | 57.5° | No projection works |

### Scale-Up (5M params, v2-none-K41, BSZ=4096, lr=3e-4)

| Dataset | Epochs | Best Val | W&B |
|---------|--------|----------|-----|
| 10M events | 10 | 57.5° | hmtocuhp |
| **132M events (full)** | **4** | **56.3°** | hmtocuhp |
| 132M events (full) | 10 | *running* | 1xg0k2c3 |

### Key Findings

1. **v2 architecture >> v1**: 5° improvement from norms/init alone
2. **Input mode doesn't matter**: none ≈ linear ≈ MLP (all 57-58° at small scale)
3. **K=41 ≈ K=16**: 78% of DOMs have 1 pulse; extra slots are mostly zeros
4. **Full dataset helps**: 57.5° → 56.3° going from 10M to 132M events
5. **Bigger head (1024 vs 128) utilized at scale**: effective rank 201/256

---

## Detailed Model Analysis

### Small Model (548K params, v2-linear K=16, 57.0°)

#### Feature importance (input projection column norms)
- **t1 dominates**: col norm 6.26 (5× next feature)
- By type: time 40%, charge 26%, aux 25%, xyz 6%, n_pulses 2%
- Per-slot: pulse 1 norm = 6.42, pulses 2-16 all ≈ 1.4

#### Residual scaling (learned)
```
Layer 0: resid=1.11, x0=0.37  (strong x0 bleed)
Layer 1: resid=0.90, x0=0.22
Layer 2: resid=0.63, x0=0.10  (heavy shrinkage)
Layer 3: resid=0.52, x0=0.13  (x0 rebounds)
```

#### Attention head specialization (layer 3, 4 layers total)
- **Heads 2, 4, 6**: "timing heads" — attend to earliest-hit DOMs (t1 ρ = -0.33 to -0.45)
- **Head 3**: "depth head" — z-position (ρ = +0.22)
- **All heads**: prefer multi-pulse DOMs
- Charge (q1) uncorrelated with attention routing (goes through FFN instead)
- CLS self-attention collapses: 0.08 → 0.006 across layers

#### Z-bias
- Target z-mean: 0.001 → predicted: 0.54 (**severe bias**)

---

### 5M Model (v2-none-K41, 56.3°)

#### Residual scaling — much more extreme

```
Layer  resid_λ  x0_λ     vs small model
  0     1.70    0.96     (was 1.11, 0.37)
  1     1.53    0.66     (was 0.90, 0.22)
  2     1.08    0.52     (was 0.63, 0.10)
  3     0.78    0.14
  4     0.54    0.17
  5     0.48    0.26     (x0 rebounds in final layer)
```

Key: x0_λ[0] = 0.96 means the model nearly doubles the initial embedding in layer 0.
With input_mode="none", this preserves raw physics features through the network.
resid_λ decays from 1.7 → 0.48 — early amplification, late attenuation.

#### Attention head specialization (6 layers, 8 heads)

**Layers 0-1 show strong specialization:**

| Layer.Head | Role | Correlation |
|-----------|------|-------------|
| L0.H3 | Z-depth (down) | z = **-0.81** |
| L0.H5 | Z-depth (up) | z = **+0.86** |
| L0.H1 | Early timing | t1 = **-0.45** |
| L0.H6 | Late timing | t1 = +0.46 |
| L1.H5 | Early timing (strong) | t1 = **-0.61** |
| L1.H0 | Z-depth | z = -0.79 |
| L1.H3 | Z + timing | z = -0.50, t1 = +0.46 |

**Late layers (4-5):** all heads uniformly prefer multi-pulse DOMs (n_pulses ρ = +0.15 to +0.24).

**CLS self-attention pattern:**
- Layer 0: very low (0.001-0.01) — CLS reads from DOMs
- Layers 1-2: some heads become "CLS sinks" (self-attn up to 0.58) — skip connections
- Layers 3-5: refocus on DOMs (self-attn drops to 0.005-0.08)

**Charge (q1) remains uncorrelated** across all layers/heads — routed through FFN.

#### Multi-pulse DOMs receive disproportionate attention

| DOM type | % of DOMs | Relative CLS attention (layer 5) |
|----------|-----------|----------------------------------|
| 1 pulse | 78% | 0.64× (under-attended) |
| 2-3 pulses | 16% | 1.2-1.8× |
| **4-10 pulses** | **4%** | **2.4×** |
| 10+ pulses | 2% | 1.2× |

Sweet spot is 4-10 pulses. Very high multiplicity (10+) gets average attention — possibly noisy.

#### Zero-padded dims become active immediately

Since input_mode="none", dims 0-126 have real features, dims 127-255 are zero-padded:

```
             DOM [0:126]  DOM [127:255]  Ratio
Input:          16.0          0.0         ∞
Block 0:        77.0         31.7         2.4×
Block 2:       239.3        179.7         1.3×
Final norm:     12.2         10.4         1.2×
```

The model fills the padded dimensions by layer 0 and uses the full 256-dim space.
For CLS, the split is even more balanced (ratio 1.06 at output).

#### Norm progression

```
             CLS norm    DOM mean    DOM max
Input:          1.3       16.0        16.0
Block 0:       40.0       88.8       437.7
Block 1:       88.5      183.2       602.2
Block 2:      107.3      320.9       909.9
Block 3:      100.6      401.6      1062.4   ← peak
Block 4:      144.5      273.0       611.7
Block 5:      234.3      194.1       336.4
Final norm:    16.0       16.0        16.0
```

Norms grow to ~1000 in mid-layers, then contract via resid_λ attenuation.
Final RMSNorm normalizes everything to √256 = 16.

#### Z-bias dramatically improved

```
Small model: predicted z = 0.54, target = 0.001  → bias 0.54
5M model:    predicted z = 0.15, target = 0.032  → bias 0.12  (5× reduction)
```

x,y biases are negligible (< 0.01). Still a slight upward z preference.

#### DirectionalHead (1024 hidden) — fully utilized

- **Effective rank: 201/256** (vs 57/128 for small model)
- Zero dead neurons (0/1024)
- Mean GELU activation rate: 87.7%
- fc2 output rows nearly orthogonal (cosine sim < 0.05)
- fc2 z-row has slightly higher norm (1.98 vs 1.69-1.71 for x,y) — consistent with residual z-bias

#### Angular error distribution

```
P10:    2.0°   (excellent on favorable events)
P25:    7.8°
Median: 49.8°  (well below mean)
Mean:   56.3°
P75:   94.7°
P90:  127.9°   (long tail of hard events)
```

Right-skewed: many events are predicted well, but a tail of hard events pulls up the mean.

---

## Scale-Up Configurations

| Config | d_model | hidden | Layers | Heads | Params |
|--------|---------|--------|--------|-------|--------|
| small | 128 | 256 | 4 | 8 | ~560K |
| **5M (current)** | **256** | **1024** | **6** | **8** | **~5.0M** |
| 7M | 256 | 1024 | 8 | 8 | ~6.6M |
| 8M | 320 | 1280 | 6 | 8 | ~7.8M |
| 10M-deep | 256 | 1024 | 12 | 8 | ~9.7M |

## W&B Links

- v2 MLP K16 (small): https://wandb.ai/polargeese/iceaggr/runs/o6y5fsaa
- v2 linear K16 (small): https://wandb.ai/polargeese/iceaggr/runs/uvr6cstr
- v2 none K41 5M (10M events): https://wandb.ai/polargeese/iceaggr/runs/hmtocuhp
- v2 none K41 5M (full, 10ep): https://wandb.ai/polargeese/iceaggr/runs/1xg0k2c3

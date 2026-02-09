# Flat Transformer v2: Architecture & Ablation Results

## Summary

The flat transformer v2 with nanochat-style improvements achieves **57.0-57.8°** mean angular error on IceCube direction reconstruction, a **5° improvement** over v1 (62.0°). The architectural improvements (RMSNorm, zero-init projections, QK norm, residual scaling) account for the gain. Input projection mode (MLP/linear/none) and number of pulse slots (K=16 vs K=41) have minimal impact.

## Architecture (v2)

Differences from v1:
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

## Results (10M events, 10 epochs, lr=1e-4)

| Run | K | Input | Head | Params | Best Val | Notes |
|-----|---|-------|------|--------|----------|-------|
| v1 baseline | 16 | linear | 128 | 554K | 62.0° | 3-epoch plateau at 87.9° |
| **v2 linear** | **16** | **linear** | **128** | **548K** | **57.0°** | Best at K=16 |
| v2 MLP | 16 | mlp | 128 | 565K | 57.6° | MLP doesn't help |
| v2 linear K41 | 41 | linear | 512 | 608K | 57.8° | More pulses don't help |
| **v2 none K41** | **41** | **none** | **512** | **592K** | **57.5°** | No projection works |

Key findings:
1. **v2 architecture >> v1**: 5° improvement from norms/init alone
2. **Input mode doesn't matter**: none ≈ linear ≈ MLP (all 57-58°)
3. **K=41 ≈ K=16**: 78% of DOMs have 1 pulse; extra slots are zeros
4. **Bigger head (512 vs 128) doesn't help**: head is not the bottleneck
5. **All models plateau ~57°** at this scale (600K params, 10M events)

## Model Analysis (v2-linear, K=16)

### Feature importance (input projection column norms)
- **t1 dominates**: col norm 6.26 (5× next feature)
- Time features: 40% of total, charge: 26%, aux: 25%, xyz: 6%
- Pulse 1 norm = 6.42, pulses 2-16 all ≈ 1.4

### Learned residual scaling
```
Layer 0: resid=1.11, x0=0.37  (strong x0 bleed)
Layer 1: resid=0.90, x0=0.22
Layer 2: resid=0.63, x0=0.10  (heavy shrinkage)
Layer 3: resid=0.52, x0=0.13  (x0 rebounds)
```

### Attention head specialization (layer 3)
- **Heads 2, 4, 6**: "timing heads" — attend to earliest-hit DOMs (t1 corr = -0.33 to -0.45)
- **Head 3**: "depth head" — correlates with z-position (+0.22)
- **All heads**: prefer DOMs with more pulses
- CLS self-attention collapses: 0.08 → 0.006 across layers

### Norm progression
```
Input proj:  CLS=0.48   DOMs=11.3
Block 0:     CLS=22.3   DOMs=24.0
Block 1:     CLS=22.6   DOMs=34.0  ← peak
Block 2:     CLS=16.6   DOMs=31.0
Block 3:     CLS=26.4   DOMs=25.3
Final norm:  CLS=11.3
```

### Z-bias problem
- Target z-mean: 0.001 (uniform on sphere)
- Predicted z-mean: 0.54 (strong downgoing bias)
- Source: CLS embedding, not head bias (fc2 z-bias = 0.028)

## Scale-Up Configurations

| Config | d_model | hidden | Layers | Heads | Params |
|--------|---------|--------|--------|-------|--------|
| current | 128 | 256 | 4 | 8 | ~560K |
| 5M | 256 | 1024 | 6 | 8 | ~5.0M |
| 7M | 256 | 1024 | 8 | 8 | ~6.6M |
| 8M | 320 | 1280 | 6 | 8 | ~7.8M |
| 10M-deep | 256 | 1024 | 12 | 8 | ~9.7M |

## W&B Links
- v2 MLP K16: https://wandb.ai/polargeese/iceaggr/runs/o6y5fsaa
- v2 linear K16: https://wandb.ai/polargeese/iceaggr/runs/uvr6cstr

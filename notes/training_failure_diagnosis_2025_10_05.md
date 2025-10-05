# Training Failure Diagnosis - 2025-10-05

## Problem Summary

Training run `baseline-1m-e2e-20251004_234040` failed after ~3 hours (epoch 3) with:
```
AssertionError: Non-finite values in predicted vectors
```

Additionally, the loss was **not decreasing** - stuck at ~1.5 radians (~86°), which is essentially random predictions (π/2 = 90° for orthogonal vectors).

## Root Cause Analysis

### 1. **Learning Rate Too High** ⚠️

**Finding**: LR=3e-4 causes gradient instability

**Evidence from diagnostics** (`scripts/diagnose_training_issue.py`):
| Learning Rate | Max Grad Norm | Loss Decreasing? | Stability |
|---------------|---------------|------------------|-----------|
| 1e-5 | 1.1 | No | ✓ Stable but too slow |
| 3e-5 | 312.8 | **Yes** | ✓ Stable |
| **1e-4** | **1.88** | **Yes** | **✓✓ Optimal** |
| 3e-4 (current) | 109.6 | Yes | ⚠️ Borderline |
| 1e-3 | 432M+ | No | ❌ Explodes |

**Gradient explosion with LR=1e-3:**
```
t1.layers.0.qkv_proj.weight: grad_norm=432,542,528  (432 MILLION!)
t1.layers.0.qkv_proj.weight: grad_norm=562,178,496  (562 MILLION!)
```

Even at LR=3e-4, gradients reach 100-300 range before clipping. After hours of training with aggressive gradient clipping (1.0), the model weights become unstable → NaN.

### 2. **Input Features Not Normalized** ⚠️⚠️

**Finding**: Raw pulse features have massive values

**Input feature ranges** (from `/groups/pheno/inar/icecube_kaggle/train/batch_1.parquet`):
| Feature | Min | Max | Mean |
|---------|-----|-----|------|
| **time** | **5,714** | **77,785** | **~13,130** |
| **charge** | 0.025 | 2,762 | ~3.9 |
| **auxiliary** | 0 | 1 | ~0.28 |
| **sensor_id** | 0 | 5,159 | - |

The T1 model receives these raw values directly:
```python
# In dataset.py collate function:
pulse_features = torch.cat([time, charge, sensor_id, auxiliary], dim=1)  # (n_pulses, 4)
# No normalization applied!
```

**Impact**:
- First layer weights multiply with values up to 77K
- Activations explode exponentially through transformer layers
- Gradients become unstable
- Eventually → NaN

### 3. **No Dropout** (Minor)

Current config: `dropout: 0.0`
- No regularization
- Makes training more sensitive to hyperparameters
- Increases risk of overfitting

### 4. **Loss Stuck at Random Baseline**

Initial loss: ~1.5 radians = ~86°
- π/2 = 90° is the expected angular error for random (orthogonal) predictions
- Model not learning anything meaningful
- Indicates fundamental optimization failure

## Validation Tests

### Test 1: Short Training Run (✓ Passed)
- **Script**: `scripts/debug_training_nan.py`
- **Setup**: 10K events, 40 batches
- **Result**: No NaN, loss decreased from 1.59 → 1.49
- **Conclusion**: Model architecture is sound; issue emerges during extended training

### Test 2: Learning Rate Sensitivity (✓ Passed)
- **Script**: `scripts/diagnose_training_issue.py`
- **Result**: Identified LR=1e-4 as optimal (gradients <2, loss decreases)

### Test 3: Fresh Model Initialization (✓ Passed)
- **Script**: `scripts/debug_nan_issue.py`
- **Result**: No NaN in forward pass or gradients for first 5 batches

## Recommended Fixes

### Fix 1: Lower Learning Rate ✓

**Change**: `learning_rate: 0.0003` → `learning_rate: 0.0001`

**Rationale**:
- LR=1e-4 shows stable gradients (<2) and consistent loss decrease
- 3x reduction from current value
- Still fast enough for reasonable convergence

**Alternative**: LR=3e-5 with warmup for maximum stability

### Fix 2: Normalize Input Features ✓✓ (CRITICAL)

**Implementation needed in** `/src/iceaggr/models/dom_transformer.py`:

```python
class DOMTransformer(nn.Module):
    def __init__(self, ...):
        ...
        # Add input normalization layer
        self.input_norm = nn.BatchNorm1d(4)
        # Or use manual normalization with fixed stats
        self.register_buffer('feature_mean', torch.tensor([13130.0, 3.9, 2580.0, 0.28]))
        self.register_buffer('feature_std', torch.tensor([10000.0, 10.0, 1500.0, 0.45]))

    def forward(self, batch):
        pulse_features = batch["packed_sequences"]  # (bsz, seq_len, 4)

        # Normalize
        pulse_features = (pulse_features - self.feature_mean) / self.feature_std
        # OR
        pulse_features = self.input_norm(pulse_features.transpose(1, 2)).transpose(1, 2)

        ...
```

**Rationale**:
- Brings all features to same scale (~0 mean, ~1 std)
- Prevents activation/gradient explosion
- Standard practice in deep learning
- Should have been done from the start!

### Fix 3: Add Dropout ✓

**Change**: `dropout: 0.0` → `dropout: 0.1`

**Rationale**:
- Provides regularization
- Stabilizes training
- Standard value for transformers

### Fix 4: Enable Shuffling (Follow-up)

**Current**: `shuffle=False` (from performance optimization)

**Follow-up task**: Implement efficient file-aware shuffling as described in `notes/performance_optimization_todo.md`
- Currently sacrificing model quality for speed
- Need proper solution

## Files Created

### Diagnostic Scripts
1. `scripts/debug_nan_issue.py` - Check for NaN in forward/backward pass
2. `scripts/debug_training_nan.py` - Simulate training to find NaN
3. `scripts/diagnose_training_issue.py` - LR sensitivity analysis
4. `scripts/find_nan_batch.py` - Search for problematic data batches

### Fixed Configuration
- `experiments/baseline_1m/config_fixed.yaml` - LR and dropout fixes applied

### Documentation
- This file

## Action Items

- [ ] **Implement input normalization in DOMTransformer** (CRITICAL)
- [ ] **Test training with `config_fixed.yaml`** (verify fixes work)
- [ ] **Add normalization layer to model architecture**
- [ ] **Document input feature statistics** (mean/std for normalization)
- [ ] **Implement proper shuffling** (performance optimization)
- [ ] **Add learning rate warmup** (optional, for extra stability)
- [ ] **Monitor gradient norms during training** (add to logging)

## Lessons Learned

1. **Always normalize inputs** - Raw features with large values (10K+) are a recipe for disaster
2. **Start with conservative hyperparameters** - LR=3e-4 might work elsewhere but not here
3. **Profile gradients early** - Would have caught this before 3-hour training run
4. **Test architectural choices** - Dropout=0 makes training brittle
5. **Monitor loss convergence** - Loss stuck at random baseline is a red flag

## References

- Performance optimization: `notes/performance_optimization_todo.md`
- Session summary (overly optimistic): `notes/training_session_2025_10_04_summary.md`
- Architecture docs: `notes/hierarchical_transformer_architecture.md`
- W&B run: https://wandb.ai/polargeese/iceaggr/runs/g03tiyau

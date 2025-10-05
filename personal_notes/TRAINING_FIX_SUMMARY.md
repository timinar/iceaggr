# Training Fix Summary - 2025-10-05

## Problem
Training failed after 3 hours with NaN predictions and loss stuck at ~1.5 radians (random baseline).

## Root Cause
**Missing input normalization** - pulse features had values up to 77,785 causing gradient explosion.

## Solution Implemented ✅

### 1. Input Feature Normalization (CRITICAL)
Added proper normalization in `src/iceaggr/models/dom_transformer.py`:

```python
# Pulse features (time, charge, sensor_id, auxiliary)
time_normalized = (time - 1e4) / 3e4
charge_normalized = torch.log10(charge + 1e-8) / 3.0
# sensor_id and auxiliary pass through
```

### 2. Geometry Normalization
Added in `src/iceaggr/models/event_transformer.py`:

```python
dom_geometry_normalized = dom_geometry / 500.0
```

### 3. Configuration Updates
Created `experiments/baseline_1m/config_fixed.yaml`:
- Added `dropout: 0.1` (was 0.0)
- LR can stay at 3e-4 with normalization!

## Results After Fix

| Learning Rate | Before Normalization | After Normalization |
|---------------|---------------------|---------------------|
| 1e-5 | ❌ Loss +0.023 | ✅ Loss -0.058 |
| 3e-5 | ✅ Loss -0.040 | ✅ Loss -0.067 |
| 1e-4 | ✅ Loss -0.024 | ✅ Loss -0.067 |
| **3e-4** | **⚠️ Loss -0.049** | **✅ Loss -0.093** |
| 1e-3 | ❌ EXPLODES | ✅ Loss -0.097 |

**Key Finding**: With normalization, LR=3e-4 is actually optimal (best loss decrease)!

## Files Modified

### Model Architecture
1. `/src/iceaggr/models/dom_transformer.py` - Added pulse feature normalization
2. `/src/iceaggr/models/event_transformer.py` - Added geometry normalization

### Configuration
3. `/experiments/baseline_1m/config_fixed.yaml` - Updated config with dropout

### Documentation
4. `/notes/training_failure_diagnosis_2025_10_05.md` - Detailed diagnosis
5. This file - Quick summary

## Next Steps

### Immediate
- [x] Implement normalization (DONE)
- [x] Test with fixed config (DONE)
- [ ] **Run full training with config_fixed.yaml**

### Command to run:
```bash
uv run python scripts/train_from_config.py experiments/baseline_1m/config_fixed.yaml
```

### Follow-up
- [ ] Enable shuffling (see `notes/performance_optimization_todo.md`)
- [ ] Monitor training for full convergence
- [ ] Compare to baseline metrics

## Key Lesson

**ALWAYS NORMALIZE INPUTS!**

Raw features with large values (10K+) will cause:
1. Gradient explosion → NaN
2. Poor convergence → stuck at random baseline
3. Wasted GPU time (3 hours of failed training)

This should have been done from the start. The reference implementation had:
```python
T_evt[:, 0] = (time - 1e4) / 3e4
T_evt[:, 1] = np.log10(charge) / 3.0
```

Always check your reference implementation carefully!

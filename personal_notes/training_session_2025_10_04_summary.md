# Training Session Summary - 2025-10-04

## What We Accomplished ✅

### 1. **Full E2E Training Pipeline Working**
- Implemented complete training loop for T1 (DOM-level) → T2 (Event-level) hierarchical transformer
- Model: 1.6M parameters (d_model=128, 4 layers each for T1 and T2)
- Training on 900K events with validation on 10K events
- W&B logging integrated with timestamped runs
- Loss function: Angular distance using unit vectors (stable, no NaN issues)

### 2. **Major Performance Fixes**
- ✅ Fixed `random_split` Subset overhead by creating separate datasets
- ✅ Fixed shuffle bottleneck: `shuffle=True` caused 140× slowdown (11.6 sec/batch → 84 ms/batch)
  - **Current workaround**: `shuffle=False` (NOT ideal for training!)
  - Root cause: PyTorch RandomSampler is extremely slow with 900K events
- ✅ Fixed NaN issues in unit vector → angles conversion
  - Changed from `arccos(z/norm)` to `arccos(z.clamp(-1, 1))`
- ✅ Switched to unit vector loss (no angle conversion in model or loss)
  - Model outputs unit vectors directly
  - Dataset converts angle targets to unit vectors in collate function
  - Loss uses `angular_dist_score_unit_vectors` (dot product + arccos)

### 3. **Infrastructure Improvements**
- ✅ Timestamped W&B run names: `baseline-1m-e2e-20251004_231058`
- ✅ Timestamped log files: `logs/baseline_1m/training_20251004_231058.log`
- ✅ Updated .gitignore for experiment artifacts (checkpoints, wandb)
- ✅ Batch size increased to 256 for better GPU utilization

### 4. **Performance Analysis**
- ✅ Created comprehensive benchmarks:
  - `scripts/debug_shuffle_overhead.py` - Quantified shuffle overhead (140× slowdown)
  - `scripts/test_gpu_utilization.py` - Overfit test to isolate bottleneck
- ✅ **Key finding**: I/O is NOT the bottleneck!
  - Same throughput (490 events/sec) with and without data loading
  - **Conclusion**: Model is too small to saturate GPU
  - GPU utilization: 10-60% (volatile, low)

---

## Current Training Status

**Running**: batch_size=256, shuffle=False, 900K train events

**Performance**:
- ~490-520 events/sec
- ~600ms per step (256 events)
- Loss starting around 1.5 radians (~86°)
- W&B: https://wandb.ai/polargeese/iceaggr/runs/ywrriixp

---

## Critical Issues to Address 🚨

### 1. **Shuffling Disabled (NOT A REAL SOLUTION)**
**Problem**: Training without shuffle → model sees events in same order every epoch → poor generalization

**Proper solutions**:
- [ ] **Option A**: Pre-shuffle indices once per epoch (fast random permutation)
- [ ] **Option B**: Custom fast sampler with cached shuffled indices
- [ ] **Option C**: Shuffle dataset files offline

**Why critical**: No shuffling significantly hurts model performance

### 2. **Low GPU Utilization (10-60%)**
**Root cause**: Model is too small (1.6M params) to saturate modern GPU

**Solutions to test**:
- [ ] Increase batch size: 512, 1024, 2048 (until OOM or util >80%)
- [ ] Increase model size: More layers (8-12), wider embeddings (256, 512)
- [ ] Use gradient accumulation if memory limited
- [ ] Profile with torch.profiler to find exact bottleneck

### 3. **Need Systematic Profiling**
- [ ] PyTorch profiler to identify slow operations
- [ ] Test different batch sizes vs GPU util
- [ ] Measure theoretical vs actual FLOPS utilization

---

## Files Created/Modified

### New Files
- `notes/performance_optimization_todo.md` - Detailed optimization roadmap
- `notes/training_session_2025_10_04_summary.md` - This summary
- `scripts/test_gpu_utilization.py` - GPU utilization test via overfitting
- `scripts/debug_shuffle_overhead.py` - Shuffle performance benchmark

### Modified Files
- `experiments/baseline_1m/config.yaml` - batch_size=256, num_workers=0
- `scripts/train_from_config.py` - Timestamped logs and W&B runs
- `src/iceaggr/data/dataset.py` - Targets converted to unit vectors in collate
- `src/iceaggr/models/event_transformer.py` - Outputs unit vectors (not angles)
- `src/iceaggr/training/losses.py` - Fixed `unit_vector_to_angles` NaN issue
- `.gitignore` - Added experiment artifacts

---

## Next Steps (Priority Order)

### Immediate (Next Session)
1. **Implement proper shuffling** (30 min)
   - Pre-shuffle indices at epoch start
   - Test that it doesn't slow training

2. **Optimize GPU utilization** (1 hour)
   - Test batch sizes: 512, 1024, 2048
   - Monitor GPU memory and utilization
   - Find optimal batch size for >80% GPU util

3. **Run torch.profiler** (15 min)
   - Identify exact bottlenecks
   - Check if any unexpected slow operations

### Medium Term
4. **Model size experiments**
   - Wider model: d_model=256, 512
   - Deeper model: 8-12 layers
   - Track performance vs accuracy tradeoff

5. **Baseline comparison**
   - Compare to spline-mpe baseline (from Kaggle)
   - Establish target angular error threshold

### Long Term
6. **Advanced optimizations** (if needed)
   - Mixed precision training (FP16)
   - Compiled model (torch.compile)
   - Custom CUDA kernels (if specific ops are slow)

---

## Benchmarking Results

### Shuffle Overhead Test
```
10K events:  shuffle overhead ~0ms (negligible)
100K events: shuffle overhead ~2.5ms
900K events: shuffle overhead ~11,600ms (11.6 sec!)
→ 140× slowdown with shuffle=True on large datasets
```

### GPU Utilization Test (Overfit Single Batch)
```
Batch: 256 events, 500 iterations
Throughput: 490 events/sec (522ms/iter)
Loss reduction: 92% (1.48 → 0.12 radians)
→ Same speed as training with data loading
→ Proves I/O is NOT the bottleneck
```

### Training Throughput
```
With shuffle=False, batch_size=256:
- ~490-520 events/sec
- ~600ms per step
- GPU util: 10-60% (volatile)
```

---

## Key Learnings

1. **PyTorch RandomSampler doesn't scale** to hundreds of thousands of samples
   - Need custom sampling strategy for large datasets

2. **Model size matters for GPU utilization**
   - 1.6M params too small for modern GPUs
   - Need bigger model or batch to saturate compute

3. **Unit vectors > angles for numerical stability**
   - No NaN issues from arccos domain errors
   - Simpler loss function (dot product)

4. **num_workers=0 is optimal for this dataset**
   - Multiprocessing overhead > benefit
   - Dataset __getitem__ is fast (~11ms)

---

## References

- Experiment config: `experiments/baseline_1m/config.yaml`
- Architecture docs: `notes/hierarchical_transformer_architecture.md`
- Performance TODOs: `notes/performance_optimization_todo.md`
- W&B project: https://wandb.ai/polargeese/iceaggr

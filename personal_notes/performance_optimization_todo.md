# Performance Optimization TODO

**Current Status**: Training works but GPU utilization is poor (10-60% fluctuating)

## Critical Issues to Fix

### 1. **Shuffling Disabled (NOT A REAL SOLUTION!)**
**Current workaround**: `shuffle=False` to avoid 11.6 sec/batch overhead
**Root cause**: PyTorch's RandomSampler is extremely slow with 900K events (140× slowdown)

**Proper solutions to implement**:
- [ ] **Option A**: File-aware block sampler (RECOMMENDED)
  - **Problem**: Data is in 660 parquet files. Random sampling across files requires loading many files per batch → slow I/O
  - **Solution**: Sample entire blocks from same parquet file, then shuffle blocks
  - Algorithm:
    1. Shuffle list of parquet files at epoch start
    2. Load batches sequentially from each file
    3. Result: Good shuffling without random file access
  - **Implementation**: Custom `BlockShuffleSampler` that groups indices by file

- [ ] **Option B**: Pre-shuffle event indices once per epoch
  - Pre-shuffle event indices at epoch boundaries
  - Use simple sequential batching from shuffled list
  - Still has issue: Random indices → random file access → slow

- [ ] **Option C**: Pre-shuffle dataset files offline
  - Reorganize all events into fewer large parquet files in shuffled order
  - Use sequential loading during training
  - Trade: Disk space and preprocessing time for training speed

**Why this matters**: Without shuffling, model sees events in same order every epoch → worse generalization

**Key insight**: The real bottleneck with shuffling is not the random permutation itself (which is fast), but the random file I/O when events are spread across 660 parquet files. Need file-aware sampling strategy.

---

### 2. **Systematic I/O Profiling Needed**
**Hypothesis**: Dataset/I/O is the bottleneck, NOT model computation

**Evidence**:
- GPU utilization: 10-60% (should be >80% if compute-bound)
- Volatile utilization pattern suggests waiting on data
- Model is small (1.6M params) but batch processing is slow

**Profiling tasks**:
- [x] **Test 1: Overfit single large batch** ✅ COMPLETED (2025-10-04)
  - Loaded batch of 256 events, trained for 500 steps
  - **Result**: 490 events/sec (same as training with data loading!)
  - **Conclusion**: I/O is NOT the bottleneck. Model is too small to saturate GPU.
  - **Evidence**: Same throughput with/without data loading
  - **Implication**: Need to increase model size or batch size, not optimize I/O

- [ ] **Test 2: PyTorch profiler**
  ```python
  from torch.profiler import profile, ProfilerActivity
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      # Training loop
  print(prof.key_averages().table(sort_by="cuda_time_total"))
  ```

- [ ] **Test 3: Data loading timing**
  - Time each step of data pipeline separately:
    - Parquet read time
    - Collate function time
    - Device transfer time
    - Forward pass time
    - Backward pass time

---

### 3. **Dataset Optimization Options**

If I/O is confirmed as bottleneck:

**Option A: Pre-processed HDF5/PyTorch tensor cache**
- [ ] Pre-process entire dataset into efficient format
- [ ] Store DOM-packed sequences ready to load
- [ ] Trade disk space for speed

**Option B: Multiprocessing with proper design**
- [ ] Currently num_workers>0 SLOWS DOWN (overhead > benefit)
- [ ] Design: Workers load+collate, main process only does GPU ops
- [ ] May need persistent workers + prefetching

**Option C: GPU-direct storage**
- [ ] Load data directly to GPU memory if supported
- [ ] Bypass CPU entirely for large tensors

---

### 4. **Batch Size Tuning**
**Current**: batch_size=256

**Next steps**:
- [ ] Find optimal batch size for GPU memory
- [ ] Test: 256, 512, 1024
- [ ] Monitor: GPU memory usage, utilization, throughput
- [ ] May need gradient accumulation for large effective batch sizes

---

## Immediate Action Plan

1. **Run overfit test** (5 min) - Verify I/O bottleneck hypothesis
2. **Profile with torch.profiler** (10 min) - Find exact bottleneck
3. **Implement fast shuffling** (30 min) - Fix training properly
4. **Optimize data pipeline** (1-2 hours) - Based on profiling results

---

## Related Files
- Shuffle benchmark: `scripts/debug_shuffle_overhead.py`
- Overfit test: `scripts/overfit_single_batch.py` (exists, needs update for large batch)
- Training config: `experiments/baseline_1m/config.yaml`
- Data loading: `src/iceaggr/data/dataset.py`, collate functions

---

## Notes
- Performance regression tests needed once optimized
- Document final throughput (events/sec) and GPU utilization
- Compare to theoretical maximum (GPU FLOPS vs model size)

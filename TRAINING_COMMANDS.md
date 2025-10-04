# Training Commands Reference

## Timing Estimates (900K events, batch_size=256)

**Current performance**:
- ~600ms per step (256 events)
- ~512 events/sec throughput
- 900K events = 3516 steps per epoch
- **1 epoch ≈ 35 minutes** (3516 steps × 0.6s = 2110s ≈ 35 min)

**For 6 hours of training**:
- 6 hours = 360 minutes
- **~10 epochs in 6 hours** (360 / 35 ≈ 10.3 epochs)

**Recommended config for overnight run**:
- Set `num_epochs: 10` for 6-hour run
- Set `num_epochs: 20` for 12-hour run (overnight)
- Current config has `num_epochs: 20` (will take ~12 hours)

---

## Using Screen (Recommended - Simpler)

### Start training in screen:
```bash
# Navigate to project
cd /lustre/hpc/pheno/inar/iceaggr

# Start new screen session named "training"
screen -S training

# Run training (will automatically log to timestamped file)
uv run python scripts/train_from_config.py experiments/baseline_1m/config.yaml

# Detach from screen: Press Ctrl+A, then D
```

### Reconnect to check progress:
```bash
# List screen sessions
screen -ls

# Reattach to training session
screen -r training

# Detach again: Ctrl+A, then D
```

### Kill training if needed:
```bash
# Reattach to session
screen -r training

# Kill process: Ctrl+C

# Exit screen: Ctrl+D or type 'exit'
```

---

## Using tmux (Alternative)

### Start training in tmux:
```bash
# Navigate to project
cd /lustre/hpc/pheno/inar/iceaggr

# Start new tmux session named "training"
tmux new -s training

# Run training
uv run python scripts/train_from_config.py experiments/baseline_1m/config.yaml

# Detach from tmux: Press Ctrl+B, then D
```

### Reconnect to check progress:
```bash
# List tmux sessions
tmux ls

# Reattach to training session
tmux attach -t training

# Detach again: Ctrl+B, then D
```

### Kill training if needed:
```bash
# Reattach to session
tmux attach -t training

# Kill process: Ctrl+C

# Exit tmux: Ctrl+D or type 'exit'
```

---

## Quick Config Changes

### Change number of epochs:
```bash
# Edit config file
nano experiments/baseline_1m/config.yaml

# Find line: num_epochs: 20
# Change to desired value (e.g., 10 for 6 hours)
```

### Change batch size (current optimal: 256):
```bash
# Edit same file
# Find line: batch_size: 512
# Change back to 256 (optimal)
```

---

## Monitoring Training

### Check W&B (in browser):
- Project: https://wandb.ai/polargeese/iceaggr
- Latest run will appear automatically

### Check log files:
```bash
# List recent log files
ls -lht logs/baseline_1m/training_*.log | head -5

# Tail latest log (find timestamp from above)
tail -f logs/baseline_1m/training_20251004_XXXXXX.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Quick status check:
```bash
# See if training is running
ps aux | grep train_from_config

# Check latest loss values
tail -20 logs/baseline_1m/training_*.log | grep "Loss:"
```

---

## Current Best Settings

**Optimal config** (in `experiments/baseline_1m/config.yaml`):
```yaml
batch_size: 256    # Best throughput (512 is slower!)
num_epochs: 20     # ~12 hours total
num_workers: 0     # Single process is fastest
shuffle: False     # Temporary workaround (TODO: fix)
```

**Expected timeline**:
- Epoch 0: 35 min
- Epoch 5: 3 hours
- Epoch 10: 6 hours
- Epoch 20: 12 hours

---

## Troubleshooting

### Training not starting:
```bash
# Check if another process is using GPU
nvidia-smi

# Kill all training processes
pkill -f train_from_config
```

### OOM (Out of Memory):
- Reduce batch_size to 128
- Check max_doms warnings in logs
- May need to increase max_doms=2048 to 4096

### Slow performance:
- Check shuffle is disabled (shuffle: False)
- Verify num_workers: 0
- Check GPU utilization with nvidia-smi

---

## Important Notes

⚠️ **Current limitations**:
- Shuffle is disabled (model sees same order every epoch)
- Need to implement file-aware block sampler
- GPU utilization is low (10-60%) - model too small

✅ **What's working**:
- Unit vector loss (no NaN issues)
- Timestamped logs and W&B runs
- Stable training at ~512 events/sec

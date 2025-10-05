# Training Quick Start Guide

## Start Training (One Command!)

```bash
./START_TRAINING.sh
```

That's it! The training will:
- Run in background (safe to logout)
- Save logs to `logs/baseline_1m_fixed/training_YYYYMMDD_HHMMSS.log`
- Save checkpoints every epoch
- Upload metrics to W&B

---

## Monitor Progress

### View live output:
```bash
screen -r training
```
(Press `Ctrl+A` then `D` to detach)

### Follow log file:
```bash
tail -f logs/baseline_1m_fixed/training_*.log
```

### W&B dashboard:
https://wandb.ai/polargeese/iceaggr

---

## Stop Training

```bash
screen -r training    # Reattach
# Press Ctrl+C         # Stop
# Type: exit           # Exit
```

---

## Expected Results

- **Training time**: ~12 hours
- **Loss**: Should decrease from ~1.5 → <1.0 rad
- **GPU util**: 50-80%
- **Checkpoints**: 20 files (one per epoch)

---

## If Something Goes Wrong

Check the log:
```bash
tail -100 logs/baseline_1m_fixed/training_*.log
```

Kill and restart:
```bash
screen -X -S training quit    # Kill
./START_TRAINING.sh           # Restart
```

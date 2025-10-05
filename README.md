# iceaggr

**Hierarchical Transformer for IceCube Neutrino Direction Reconstruction**

A research project developing transformer models for precise angular reconstruction of high-energy neutrino events in IceCube, using DOM-level pulse aggregation to handle events with thousands of pulses.

> **New to the project?** See setup instructions for [installing UV](#install-uv) and [downloading IceCube data](#download-icecube-kaggle-data) at the end of this README.

## 🚀 Quick Start

We use `uv` (the modern Python package manager) throughout the project for dependency management and running code (see [installing UV](#install-uv)).

**First time setup:**
```bash
# 1. Sync dependencies
uv sync

# 2. Configure data paths
cp src/iceaggr/data/data_config.template.yaml src/iceaggr/data/data_config.yaml
# Edit data_config.yaml with your IceCube data path

# 3. Run analysis scripts
uv run python scripts/2029_09_08_pulse_statistics.py
```

**After first setup:**
```bash
# Sync dependencies (run after pulling changes)
uv sync

# Run scripts
uv run python scripts/train_from_config.py experiments/baseline_1m/config_fixed.yaml
```

## 🎯 Common Commands

```bash
# Jupyter notebooks
uv run jupyter lab

# Testing
uv run pytest                    # All tests
uv run pytest --cov=src/iceaggr  # With coverage

# Code quality
uv run ruff format .             # Format code
uv run ruff check .              # Check for issues
uv run mypy src/                 # Type checking

# Dependencies
uv add package-name              # Add new package
uv add --dev dev-tool            # Add dev dependency
uv sync                          # Install/update all dependencies
```

## 🎯 Project Overview

### The Challenge

High-energy neutrino events in IceCube can produce tens of thousands of light pulses across the detector. Current maximum likelihood methods (like spline-mpe) perform well but are computationally expensive. Transformer models could offer competitive performance, but face scalability challenges:

- **Problem**: Events can have 10K+ pulses → standard transformers don't scale (O(n²) complexity)
- **Insight**: Pulses are naturally grouped by DOM (Digital Optical Module)
- **Solution**: Hierarchical transformer with DOM-level aggregation

### Architecture

```
Pulses → [DOM-level Transformer (T1)] → DOM embeddings → [Event-level Transformer (T2)] → Direction
         (per-DOM, batched)                              (across all DOMs)
```

**Two-stage processing:**
1. **T1 (DOM-level)**: Aggregates pulses within each DOM using transformer → produces DOM embedding
   - Most DOMs: 1-10 pulses (fast)
   - Some DOMs: 100s-1000s of pulses (handled via packing + FlexAttention)
   - Runs in parallel across all DOMs with smart batching

2. **T2 (Event-level)**: Aggregates DOM embeddings across the event → predicts direction
   - Max ~2000 active DOMs per event (manageable sequence length)
   - Incorporates DOM geometry (x,y,z positions)

### Key Technical Features

- **Input normalization**: Critical for stable training (time, charge, sensor_id, geometry)
- **DOM-level packing**: Multiple sparse DOMs packed into fixed-length sequences
- **FlexAttention**: Dynamic attention masking for variable-length sequences
- **Hierarchical batching**: Efficient GPU utilization via continuous batching

## 📁 Project Structure

```
iceaggr/
├── src/iceaggr/              # Main package (importable)
│   ├── data/                # Data loading with DOM-level batching
│   ├── models/              # T1 (DOM) & T2 (Event) transformers
│   ├── training/            # Training loops, losses, metrics
│   └── utils/               # Logging utilities
├── experiments/             # Experiment configurations
│   └── baseline_1m/         # 1M events baseline experiment
├── scripts/                 # Standalone scripts
│   ├── train_from_config.py # Main training script
│   ├── start_training.sh    # Training launcher
│   ├── archive/             # Benchmark & analysis scripts
│   └── debug/               # One-time debugging scripts
├── notes/                   # Architecture documentation
├── notebooks/               # Jupyter exploration
└── tests/                   # Unit & integration tests
```

**Note**: `data_config.yaml` lives in `src/iceaggr/data/` (gitignored, copy from template)

## 🧪 Current Progress

- [x] Data exploration and statistics
- [x] Dataloader with DOM-level batching and packing
- [x] DOM-level transformer (T1) with FlexAttention
- [x] Event-level transformer (T2) with geometry encoding
- [x] End-to-end training pipeline with W&B integration
- [ ] Baseline evaluation and comparison with spline-mpe
- [ ] Hyperparameter optimization

## 🚂 Training

Start training with one command:

```bash
./scripts/start_training.sh
```

This will:
- Run training in a background screen session
- Log to `logs/baseline_1m_fixed/training_YYYYMMDD_HHMMSS.log`
- Save checkpoints every epoch to `experiments/baseline_1m/checkpoints_fixed/`
- Track metrics on W&B

**Monitor progress:**
```bash
# Reattach to training session
screen -r training

# Follow log file
tail -f logs/baseline_1m_fixed/training_*.log

# View on W&B
# https://wandb.ai/polargeese/iceaggr
```

**Stop training:**
```bash
screen -r training    # Reattach
# Press Ctrl+C         # Stop
```

See [experiments/baseline_1m/config_fixed.yaml](experiments/baseline_1m/config_fixed.yaml) for training configuration.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete workflow guide.

**Quick workflow:**
```bash
git checkout main && git pull && uv sync    # Start with latest
git checkout -b experiment/your-idea         # Create branch
# ... make changes ...
uv run pytest && uv run ruff check .         # Test
git commit -m "Add feature"                  # Commit
git push origin experiment/your-idea         # Push & create PR
```

## 📊 Experiment Tracking

We use Weights & Biases:

```python
import wandb
wandb.init(project="iceaggr", name="dom-transformer-v1-yourname")
wandb.log({"loss": loss, "angular_error": error})
```

## 📚 Resources

- [IceCube Competition](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice)
- [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/)
- [Weights & Biases](https://docs.wandb.ai/)
- [UV Documentation](https://docs.astral.sh/uv/)

---

## 🔧 Setup Instructions

### Install UV

<details>
<summary>Click to expand UV installation instructions</summary>

```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart your terminal or run:
source $HOME/.local/bin/env
```

Then clone and setup:
```bash
git clone https://github.com/timinar/iceaggr.git
cd iceaggr
uv sync
```

</details>

### Download IceCube Kaggle Data

<details>
<summary>Click to expand data download instructions</summary>

Install Kaggle CLI and authenticate:
```bash
uv add kaggle

# Get API token from kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle
# Save credentials to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Download data:
```bash
kaggle competitions download -c icecube-neutrinos-in-deep-ice
unzip icecube-neutrinos-in-deep-ice.zip -d data/
```

</details>

---

Happy researching! 🔬⚡

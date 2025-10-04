# iceaggr

**Hierarchical Transformer for IceCube Neutrino Direction Reconstruction**

A research project developing transformer models for precise angular reconstruction of high-energy neutrino events in IceCube, using DOM-level pulse aggregation to handle events with thousands of pulses.

> **New to the project?** See setup instructions for [installing UV](#install-uv) and [downloading IceCube data](#download-icecube-kaggle-data) at the end of this README.

## 🚀 Quick Start
We use `uv` throughout the project for dependency management and running code (see [installing UV](#install-uv)).
```bash
# Sync dependencies (run after pulling changes)
uv sync

# Start exploring
uv run jupyter lab

# Run analysis scripts
uv run python scripts/2029_09_08_pulse_statistics.py
```

## 🎯 Common Commands

```bash
# Jupyter notebooks
uv run jupyter lab

# Run scripts
uv run python scripts/your_script.py

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
   - Some DOMs: 100s-1000s of pulses (need chunking/windowed attention)
   - Runs in parallel across all DOMs (with smart batching)

2. **T2 (Event-level)**: Aggregates DOM embeddings across the event → predicts direction
   - Max ~2000 active DOMs per event (manageable sequence length)
   - Can incorporate DOM geometry (x,y,z positions)

### Key Technical Challenges

1. **Variable sequence lengths**:
   - 1 pulse/DOM to 1000s of pulses/DOM
   - Need efficient batching strategy

2. **Parallel DOM processing**:
   - Continuous batching with attention masking
   - PyTorch FlexAttention for dynamic sequence lengths

3. **Single-pulse DOMs**:
   - ~50-70% of DOMs have ≤10 pulses
   - Need lightweight path or learned aggregation

4. **Data loading**:
   - Pre-group pulses by DOM
   - Efficient batching for variable-length sequences
   - Potential caching of DOM embeddings

## 🧪 Current Progress

- [x] Data exploration and statistics (see [scripts/2029_09_08_pulse_statistics.py](scripts/2029_09_08_pulse_statistics.py))
- [ ] Dataloader implementation with DOM grouping
- [ ] DOM-level transformer (T1) with FlexAttention
- [ ] Event-level transformer (T2)
- [ ] End-to-end training pipeline
- [ ] Comparison with spline-mpe baseline



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
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
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
# iceaggr

**Hierarchical Transformer for IceCube Neutrino Direction Reconstruction**

A research project developing transformer models for precise angular reconstruction of high-energy neutrino events in IceCube, using DOM-level pulse aggregation to handle events with thousands of pulses.

## 🚀 Quick Start

### Prerequisites
- Git installed on your system
- Terminal/command line access

### 1. Install UV (the modern Python package manager)
```bash
# Install UV - works on Windows, Mac, and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows, use PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart your terminal or run:
source $HOME/.local/bin/env
```

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/timinar/iceaggr.git
cd iceaggr

# UV automatically handles everything - Python version, virtual environment, dependencies!
uv sync
```

That's it! No need to manage virtual environments or install Python manually.

### 3. Download IceCube Kaggle Data

First, install the Kaggle CLI and authenticate:
```bash
# Install kaggle CLI (already in dependencies if you ran uv sync)
uv add kaggle

# Authenticate - get your API token from kaggle.com/USERNAME/account
# Create ~/.kaggle/kaggle.json with your credentials:
mkdir -p ~/.kaggle
# Download from: https://www.kaggle.com/settings → API → Create New Token
chmod 600 ~/.kaggle/kaggle.json
```

Download and extract the data:
```bash
# Download competition data (~20GB compressed)
kaggle competitions download -c icecube-neutrinos-in-deep-ice

# Unzip (creates train/ and test/ directories)
unzip icecube-neutrinos-in-deep-ice.zip -d data/
```


## 🎯 Usage

### Running Jupyter Notebooks
```bash
# Start Jupyter Lab for exploration
uv run jupyter lab

# Or traditional Jupyter notebook
uv run jupyter notebook
```

### Running Training Scripts
```bash
# Run a training script
uv run python scripts/train_model.py

# Run with specific config (once we add Hydra)
uv run python scripts/train_model.py --config configs/bert_baseline.yaml
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run only unit tests (fast)
uv run pytest tests/unit/

# Run only integration tests (slower, requires data)
uv run pytest tests/integration/

# Run with coverage report
uv run pytest --cov=src/iceaggr
```

### Adding New Dependencies
```bash
# Add a new package for everyone
uv add scikit-image  # Adds to main dependencies

# Add a development tool
uv add --dev black   # Adds to dev dependencies

# The changes are saved to pyproject.toml - commit this file!
# Other users (including on HPC clusters) get the new packages by running:
uv sync
```

## 📁 Project Structure

```
iceaggr/
├── src/iceaggr/           # Main code (importable package)
│   ├── config/           # YAML feature configurations
│   ├── data/             # Data loading code (DataLoaders, etc.)
│   ├── models/           # Model architectures 
│   ├── training/         # Training loops and utilities
│   └── utils/            # Utilities (logging, etc.)
├── notebooks/            # Jupyter notebooks for experiments
├── scripts/             # Standalone scripts (training, evaluation)
├── tests/              # Unit tests
└── pyproject.toml      # Project configuration (dependencies, tools)
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

**New to the project?** Check out [CONTRIBUTING.md](CONTRIBUTING.md) for our complete workflow guide.

**Quick contribution steps:**
1. Create a new branch: `git checkout -b experiment/your-idea`
2. Make your changes and test them
3. Commit: `git commit -m "Add transformer attention mechanism"`
4. Push: `git push origin experiment/your-idea`
5. Create a Pull Request on GitHub

## 📊 Weights & Biases Integration

We use Weights & Biases for experiment tracking:

```python
import wandb

# Login (first time only)
wandb.login()

# In your training script
wandb.init(project="iceaggr", name="dom-transformer-v1")
```

## 🔬 Development Tools

We use modern Python tooling for code quality:

```bash
# Format code automatically
uv run ruff format .

# Check for issues
uv run ruff check .

# Run type checking
uv run mypy src/
```

## 📚 Useful Resources

- **IceCube Kaggle Competition**: [Competition Page](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice)
- **PyTorch FlexAttention**: [Tutorial](https://pytorch.org/blog/flexattention/)
- **PyTorch Lightning**: [Documentation](https://lightning.ai/docs/pytorch/stable/)
- **Weights & Biases**: [Guides](https://docs.wandb.ai/)
- **UV Documentation**: [User Guide](https://docs.astral.sh/uv/)
- **IceCube Detector**: [Description](https://icecube.wisc.edu/science/icecube/)

## ❓ FAQ

### "UV isn't recognized as a command"
Make sure to restart your terminal after installation, or run:
```bash
source $HOME/.local/bin/env
```

### "I want to use my own Python installation"
UV manages Python for you, but if needed:
```bash
uv venv --python /path/to/your/python
```

### "How do I update dependencies?"
```bash
uv sync --upgrade  # Updates all packages to latest compatible versions
```

### "Can I still use pip?"
While UV handles everything, you can still use pip inside the UV environment:
```bash
uv run pip install some-package
```

## 🐛 Issues?

- Check [CONTRIBUTING.md](CONTRIBUTING.md) for common solutions
- Open an issue on GitHub
- Ask in our team chat

---

Happy researching! 🔬⚡
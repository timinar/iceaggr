# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**iceaggr** is a research project developing hierarchical transformer models for precise angular reconstruction of high-energy neutrino events in IceCube. The key challenge is handling events with thousands of pulses using DOM-level aggregation.

### Core Architecture

The project implements a two-stage hierarchical transformer:

1. **T1 (DOM-level transformer)**: Aggregates pulses within each Digital Optical Module (DOM) → produces DOM embeddings
   - Challenge: Variable sequence lengths (1 to 1000s of pulses per DOM)
   - Solution: Continuous batching with FlexAttention for dynamic sequences

2. **T2 (Event-level transformer)**: Aggregates DOM embeddings across the event → predicts neutrino direction
   - Input: DOM embeddings + geometry (x,y,z positions)
   - Max ~2000 active DOMs per event (manageable for standard attention)

### Key Technical Features

- **Input normalization**: CRITICAL for training stability (time, charge, sensor_id, geometry)
- **DOM-level packing**: Multiple sparse DOMs packed into fixed-length sequences  
- **FlexAttention**: Dynamic attention masking for variable-length sequences
- **Hierarchical batching**: Efficient GPU utilization via continuous batching

## Development Setup

This project uses **UV** (modern Python package manager) for all dependency management.

### Essential Commands

```bash
# Install/sync all dependencies (run after pulling changes)
uv sync

# Add new package
uv add package-name          # For main dependencies
uv add --dev package-name    # For dev tools only

# Update all packages
uv sync --upgrade
```

### Running Code

```bash
# Run Python scripts
uv run python scripts/script_name.py

# Start Jupyter Lab for notebooks
uv run jupyter lab

# Run tests
uv run pytest                    # All tests
uv run pytest tests/unit/        # Unit tests only
uv run pytest tests/integration/ # Integration tests only
uv run pytest --cov=src/iceaggr  # With coverage

# Code quality
uv run ruff format .             # Format code
uv run ruff check .              # Check for issues
uv run ruff check . --fix        # Auto-fix issues
uv run mypy src/                 # Type checking
```

## Project Structure

```
iceaggr/
├── src/iceaggr/              # Main package (importable)
│   ├── data/                # Data loading with DOM packing
│   ├── models/              # T1 (DOM) & T2 (Event) transformers
│   ├── training/            # Training loops, losses, metrics
│   └── utils/               # Logging utilities
├── experiments/             # Experiment configurations
│   └── baseline_1m/         # 1M events baseline (config, checkpoints)
├── scripts/                 # Standalone scripts
│   ├── train_from_config.py # Main training script
│   ├── archive/             # Benchmark & analysis scripts
│   └── debug/               # One-time debugging scripts
├── notes/                   # Architecture documentation
├── personal_notes/          # Session notes (gitignored)
├── notebooks/               # Jupyter exploration
├── tests/                   # Unit & integration tests
└── START_TRAINING.sh        # Quick training launcher
```

**Current state**: ✅ E2E pipeline complete! T1, T2, data loading, training all working.

## Configuration Management

The project uses a simple two-tier config approach:

1. **Data paths** (`src/iceaggr/data/data_config.yaml`):
   - Gitignored, local to each user
   - Contains **only data paths** (train, test directories)
   - Copy from `src/iceaggr/data/data_config.template.yaml` and modify
   - Example:
     ```yaml
     data:
       root: /path/to/icecube_kaggle
       train: /path/to/icecube_kaggle/train
       test: /path/to/icecube_kaggle/test
       batch_pattern: "batch_*.parquet"
     ```

2. **Experiment configs** (`experiments/` directory):
   - Committed to git
   - Model architectures, training params, logging settings
   - See `experiments/baseline_1m/config_fixed.yaml` for example

Load data config in code:
```python
import yaml

with open("src/iceaggr/data/data_config.yaml") as f:
    config = yaml.safe_load(f)

train_path = config["data"]["train"]
```

**Important**: All scripts should use `src/iceaggr/data/data_config.yaml` for data paths, not hardcoded paths!

## Training

### Start Training

```bash
./START_TRAINING.sh
```

This runs training in a screen session with logging.

### Monitor Training

```bash
# Reattach to screen session
screen -r training

# Follow log file  
tail -f logs/baseline_1m_fixed/training_*.log

# View on W&B
# https://wandb.ai/polargeese/iceaggr
```

### Training Configuration

Current experiment: `experiments/baseline_1m/config_fixed.yaml`
- Model: 1.6M parameters (d_model=128, 4 layers each for T1 and T2)
- Training: 900K events, batch_size=256, LR=3e-4, dropout=0.1
- **Input normalization**: Enabled (CRITICAL!)
  - time: `(time - 1e4) / 3e4`
  - charge: `log10(charge + 1e-8) / 3.0`
  - sensor_id: `sensor_id / 5160.0`
  - geometry: `geometry / 500.0`

## Experiment Tracking

We use Weights & Biases:

```python
import wandb

wandb.init(
    project="iceaggr",
    name="component-description-version-yourname",
    tags=["component-type", "experiment-type"]
)

wandb.log({"loss": loss, "angular_error": angular_err})
```

### Naming Conventions

- **Branches**:
  - `experiment/transformer-attention`
  - `feature/data-loader-improvements`
  - `bugfix/memory-leak-training`

- **W&B runs**: `component-description-version-yourname`
- **Tags**: `["dom-level" | "event-level" | "e2e", experiment-type, yourname]`

## Logging

Use the project's color-coded logger for consistent output:

```python
from iceaggr.utils import get_logger

logger = get_logger(__name__)
logger.info("Loading batch 42")
logger.debug("Batch shape: (32, 128, 4)")
logger.warning("Cache miss for batch 99")
logger.error("Failed to load data")
```

**Log levels**: DEBUG (blue), INFO (green), WARNING (yellow), ERROR (red)

**Change level**: `get_logger(__name__, level=logging.DEBUG)`

## Data Analysis Best Practices

**IMPORTANT**: Avoid pandas for large-scale data analysis. Use Polars or PyArrow instead.

```python
# ✅ GOOD: Use Polars for dataframes
import polars as pl
df = pl.read_parquet("data.parquet")
df = df.filter(pl.col("value") > 10)

# ✅ GOOD: Use PyArrow for columnar data
import pyarrow.parquet as pq
table = pq.read_table("data.parquet")

# ❌ BAD: Avoid pandas (10-100x slower)
import pandas as pd  # Don't use this!
```

**Why?**
- Polars is 10-100x faster than pandas
- PyArrow has zero-copy operations
- IceCube data is large (~20GB) - pandas will be painfully slow

## Development Workflow

1. **Always start with**: `git checkout main && git pull && uv sync`
2. **Create descriptive branches**: Use prefixes experiment/feature/bugfix
3. **Test before committing**: Run `uv run pytest && uv run ruff check .`
4. **Commit with context**: Explain why, not just what
5. **Track experiments**: Use W&B for all training runs
6. **Use logging**: Replace `print()` with `logger.info()` in all code

## Key Technologies

- **PyTorch**: Deep learning framework
- **PyTorch FlexAttention**: For variable-length sequence handling
- **Polars / PyArrow**: Fast dataframe operations (NOT pandas!)
- **UV**: Python package and environment manager
- **Weights & Biases**: Experiment tracking
- **Ruff**: Code formatting and linting

## Important Notes

### Input Normalization (CRITICAL!)

**ALWAYS normalize inputs!** This was the cause of a major training failure (NaN after 3 hours).

Raw features (time: 77K, charge: 2.7K, sensor_id: 5.1K) caused gradient explosion.

Normalization is implemented in:
- `src/iceaggr/models/dom_transformer.py` - pulse features
- `src/iceaggr/models/event_transformer.py` - geometry

### Scripts Organization

- **scripts/**: Main training script (`train_from_config.py`)
- **scripts/archive/**: Benchmarks and analysis (useful but not daily use)
- **scripts/debug/**: One-time debugging scripts (for reference only)

### Notes Organization

- **notes/**: Architecture docs (committed, important)
- **personal_notes/**: Session notes, TODOs (gitignored)

## Git Configuration Notes

SSH access is configured for this repository. If setting up on new system:

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub  # Add to GitHub Settings → SSH keys
ssh -T git@github.com      # Test connection
```

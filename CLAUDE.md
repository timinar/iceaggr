# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**iceaggr** is a research project developing transformer models for precise angular reconstruction of high-energy neutrino events in IceCube. The model predicts neutrino direction (azimuth, zenith) from photomultiplier pulse data across ~5000 Digital Optical Modules (DOMs).

### Core Architecture Concept

The project uses a **flat transformer** approach (`FlatTransformerV2`, ~5-10M params, achieves 55-56° angular error):

1. **DOM-level aggregation**: For each DOM, concatenate the first K pulse features (time, charge, auxiliary flag) into a fixed-length vector, prepend geometry (x,y,z) and pulse count
2. **Single transformer**: Process all DOM vectors with a standard transformer (RMSNorm, QK-norm attention, ReLU² FFN, residual scaling)
3. **CLS token**: Prepended learnable token; its output embedding is fed to a directional head that predicts unit vector → (azimuth, zenith)

Key design choices (nanochat/GPT-inspired):
- Functional RMSNorm, no learnable norm params
- Zero-init output projections (attention c_proj, FFN c_proj)
- Per-layer residual scaling + skip connection to initial embedding
- Configurable input projection: none / linear / MLP

### Key Technical Challenges

- **DOM subsampling**: Events can have up to ~2000 active DOMs; max_doms parameter limits sequence length (default: 128, selected by earliest arrival time)
- **Pulse truncation**: Each DOM keeps first K pulses (K=16-84); remaining are discarded
- **Loss function**: Angular distance loss on unit vectors (great-circle distance)

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

# Start Jupyter Lab for notebooks (automatically uses correct Python kernel)
uv run jupyter lab

# IMPORTANT: In notebooks, the iceaggr package is available after running `uv sync`
# The kernel "Python 3 (ipykernel)" uses the UV environment automatically

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
├── src/iceaggr/              # Main code (importable package)
│   ├── data/                # DataLoaders, collators, geometry ✓
│   ├── models/              # Flat transformer, losses, directional head ✓
│   │   ├── flat_transformer_v2.py  # Main model (FlatTransformerV2)
│   │   ├── flat_transformer.py     # V1 flat model (FlatTransformerModel)
│   │   ├── event_transformer.py    # TransformerBlock (used by v1)
│   │   ├── directional_head.py     # Unit vector → angles prediction
│   │   └── losses.py               # Angular distance loss
│   ├── training/            # (placeholder, training is in scripts/)
│   └── utils/               # Logging utilities ✓
├── configs/                 # Flat transformer training configs ✓
│   └── train_flat_v2_*.yaml # Various model size/input projection configs
├── archive/configs/         # Old hierarchical model configs (preserved)
├── notes/                   # Design documentation (committed) ✓
├── scripts/                 # Training and analysis scripts ✓
│   └── train_flat.py        # Main training script
├── notebooks/               # Jupyter notebooks for experiments
├── tests/                   # Unit and integration tests ✓
├── src/iceaggr/data/data_config.yaml         # Local data paths (gitignored) ✓
├── src/iceaggr/data/data_config.template.yaml # Template for data paths ✓
└── pyproject.toml           # Dependencies and project config ✓
```

**Current state**: Flat transformer trained and achieving 55-56° angular error. Data loading, model, and training pipeline all functional.

## Configuration Management

The project uses a simple two-tier config approach:

1. **Data paths** (`src/iceaggr/data/data_config.yaml` in root):
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

2. **Experiment configs** (`configs/` directory - to be created):
   - Committed to git
   - Model architectures, training params, logging settings
   - Organized by experiment type

Load data config in code:
```python
import yaml

with open("src/iceaggr/data/data_config.yaml") as f:
    config = yaml.safe_load(f)

train_path = config["data"]["train"]
```

**Important**: All scripts should use `src/iceaggr/data/data_config.yaml` for data paths, not hardcoded paths!

## Data Architecture

### IceCube Dataset Structure

- **Location**: `/groups/pheno/inar/icecube_kaggle/train/`
- **Format**: Parquet files (`batch_*.parquet`)
- **Size**: ~20GB compressed total

### Key Data Characteristics (from pulse analysis)

Based on analysis in [scripts/2029_09_08_pulse_statistics.py](scripts/2029_09_08_pulse_statistics.py):

- **DOMs per event**:
  - Median: varies by batch
  - 99th percentile: <2000 DOMs (manageable for T2)
  - Max: 5160 DOMs (full detector)

- **Pulses per DOM** (critical for T1 design):
  - 50-70% of DOMs have ≤10 pulses (sparse, lightweight path needed)
  - 99th percentile: determines max sequence length for T1
  - Heavy tail: Some DOMs have 1000s of pulses (need chunking/windowed attention)

- **Event sizes**:
  - Most events: <1K total pulses
  - Large events: 1K-10K pulses
  - Extreme outliers: >100K pulses (need special handling)

### Batching Strategy Considerations

The analysis script recommends either:
- **Option 2**: Grouped batch processing (if data is well-behaved)
- **Option 3**: Continuous/flattened batching with FlexAttention (for heavy-tailed distributions)

Choice depends on specific data distribution - run pulse statistics first.

## Experiment Tracking

### Weights & Biases Integration

```python
import wandb

# Initialize experiment
wandb.init(
    project="iceaggr",
    name="component-description-version-yourname",
    tags=["component-type", "experiment-type"]
)

# Example names:
# - "dom-transformer-baseline-v1-alice"
# - "event-transformer-geom-v2-bob"

# Log during training
wandb.log({"loss": loss, "angular_error": angular_err})
```

### Naming Conventions

- **Branches**:
  - `experiment/transformer-attention`
  - `feature/data-loader-improvements`
  - `bugfix/memory-leak-training`

- **W&B runs**: `component-description-version-yourname`
- **Tags**: `["dom-level" | "event-level" | "e2e", experiment-type, yourname]`

## Important Data Insights

From the data analysis, keep in mind:

1. **Most events are sparse**: 99% of events activate <5% of the detector
2. **DOM-level sparsity**: Median pulses per DOM is very low (often <5)
3. **Memory planning**: Batch size 32 at 99th percentile events is manageable, but worst-case can OOM
4. **Special handling needed**: Events >100K pulses may need reservoir sampling or splitting

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

See `src/iceaggr/utils/logger_config.py` for implementation. Original by [Midori Kato](https://github.com/pomidori).

## Development Workflow

1. **Always start with**: `git checkout main && git pull && uv sync`
2. **Create descriptive branches**: Use prefixes experiment/feature/bugfix/analysis
3. **Test before committing**: Run `uv run pytest && uv run ruff check .`
4. **Commit with context**: Explain why, not just what
5. **Track experiments**: Use W&B for all training runs
6. **Use logging**: Replace `print()` with `logger.info()` in all code

## Key Technologies

- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training framework
- **Polars / PyArrow**: Fast dataframe operations for analysis (NOT pandas - too slow!)
- **UV**: Python package and environment manager
- **Weights & Biases**: Experiment tracking
- **Ruff**: Code formatting and linting

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
- Polars has better memory efficiency

**When to use pandas**: Only for small results tables (<1000 rows) or final output formatting

## Git Configuration Notes

SSH access is configured for this repository. If setting up on new system:

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub  # Add to GitHub Settings → SSH keys
ssh -T git@github.com      # Test connection
```

## Next Development Steps

- [x] Dataloader implementation with DOM grouping
- [x] Flat transformer model (FlatTransformerV2)
- [x] End-to-end training pipeline (train_flat.py)
- [x] Scale-up experiments (5M, 10M params)
- [ ] Comparison with spline-mpe baseline
- [ ] Paper figures and analysis

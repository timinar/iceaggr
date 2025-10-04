# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**iceaggr** is a research project developing hierarchical transformer models for precise angular reconstruction of high-energy neutrino events in IceCube. The key challenge is handling events with thousands of pulses using DOM-level aggregation.

### Core Architecture Concept

The project implements a two-stage hierarchical transformer:

1. **T1 (DOM-level transformer)**: Aggregates pulses within each Digital Optical Module (DOM) → produces DOM embeddings
   - Challenge: Variable sequence lengths (1 to 1000s of pulses per DOM)
   - Solution: Continuous batching with FlexAttention for dynamic sequences

2. **T2 (Event-level transformer)**: Aggregates DOM embeddings across the event → predicts neutrino direction
   - Input: DOM embeddings + geometry (x,y,z positions)
   - Max ~2000 active DOMs per event (manageable for standard attention)

### Key Technical Challenges

- **Variable-length sequences**: 1 pulse/DOM to 1000s of pulses/DOM requiring efficient batching
- **Parallel DOM processing**: Using continuous batching with attention masking
- **Single-pulse DOMs**: ~50-70% of DOMs have ≤10 pulses, need lightweight path
- **Heavy-tailed distribution**: Some events have 10K+ pulses requiring special handling

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
│   ├── config/              # Configuration utilities (to be created)
│   ├── data/                # DataLoaders with DOM grouping ✓
│   ├── models/              # T1 and T2 transformer architectures (to be created)
│   ├── training/            # Training loops and utilities (to be created)
│   └── utils/               # Logging utilities ✓
├── configs/                 # Experiment configurations (to be created)
│   ├── experiment/          # Full experiment configs
│   └── model/               # Model-specific configs
├── notes/                   # Design documentation (committed) ✓
│   ├── 03_dom_aggregation_architecture.md    # Architecture design
│   └── 04_flex_attention_benchmark_results.md # Benchmarking results
├── personal_notes/          # Personal notes, experiments (gitignored)
├── scripts/                 # Standalone analysis/training scripts ✓
├── notebooks/               # Jupyter notebooks for experiments
├── tests/                   # Unit and integration tests ✓
├── src/iceaggr/data/data_config.yaml         # Local data paths (gitignored) ✓
├── src/iceaggr/data/data_config.template.yaml # Template for data paths ✓
└── pyproject.toml           # Dependencies and project config ✓
```

**Current state**: Data loading complete (✓). Design docs complete (✓). Next: Model implementation.

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
- **PyTorch FlexAttention**: For variable-length sequence handling in transformers
- **PyTorch Lightning**: Training framework (to be added)
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

## Next Development Steps (as per README)

- [ ] Dataloader implementation with DOM grouping
- [ ] DOM-level transformer (T1) with FlexAttention
- [ ] Event-level transformer (T2)
- [ ] End-to-end training pipeline
- [ ] Comparison with spline-mpe baseline

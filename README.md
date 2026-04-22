# iceaggr

**Transformer-based angular reconstruction for IceCube neutrino events.**

A research project predicting neutrino direction (azimuth, zenith) from photomultiplier pulse data across ~5,000 Digital Optical Modules (DOMs). Our best model reaches **top-5 Kaggle leaderboard performance** (0.9681 rad mean angular error on held-out validation) at a fraction of the parameters used by the competition winners, and **outperforms the top published models on physically interesting high-energy events** (≥1000 pulses). Paper in preparation.

> New to the project? See [Setup](#setup) at the end of this README.

## Approach

Events can produce thousands of pulses across thousands of DOMs, and a naive transformer over raw pulses is O(n²) with n up to 100K+ (max observed: 178K pulses in a single event). We collapse the inner axis *before* attention by building a **fixed-length DOM vector** for every active DOM:

```
  per DOM:  [x, y, z, n_pulses,  t_1, q_1, a_1,  t_2, q_2, a_2,  ...,  t_K, q_K, a_K]
            └─ geometry + count ┘└────── first K pulses (K=84) ──────┘

  event = [CLS, dom_1, dom_2, ..., dom_M]  →  Transformer  →  CLS embedding  →  direction head
```

Each DOM keeps only its **first K=84 pulses** (time / charge / auxiliary-flag triplets), concatenated into one 256-dim vector. One transformer sees the whole event as a short sequence of DOM tokens — **no per-DOM inner transformer, no K× sequence blow-up**. For K=84 and input_mode=`none`, the DOM vector already has width 256 = `d_model`, so no projection is even needed and raw features live directly in the residual stream.

The backbone is nanochat-style: functional RMSNorm (no learned scale), QK-norm attention, ReLU² FFN, zero-init output projections, per-layer residual scaling with skip to the initial embedding. The CLS token's final embedding is fed to a directional head (unit vector → angular-distance loss) or, more recently, a **vMF mixture head** trained on negative log-likelihood.

See [notes/05_flat_transformer_v2_results.md](notes/05_flat_transformer_v2_results.md) for ablations and scaling data.

## Quick Start

We use [`uv`](https://docs.astral.sh/uv/) for everything.

```bash
# 1. Install deps
uv sync

# 2. Point at your data
cp src/iceaggr/data/data_config.template.yaml src/iceaggr/data/data_config.yaml
# edit data_config.yaml with your IceCube path

# 3. Train
uv run python scripts/train_flat.py --config configs/train_flat_v2_none_K84_5M.yaml
```

Data paths are kept out of git in [src/iceaggr/data/data_config.yaml](src/iceaggr/data/data_config.yaml). Experiment configs live in [configs/](configs/) and *are* committed.

## Common Commands

```bash
# Environment
uv sync                          # Install/update deps after pulling
uv add package-name              # Add a dependency
uv add --dev tool                # Add a dev dependency

# Running
uv run python scripts/train_flat.py --config configs/<name>.yaml
uv run jupyter lab               # Notebooks (kernel picks up the uv env)

# Testing & quality
uv run pytest                    # All tests
uv run pytest --cov=src/iceaggr  # Coverage
uv run ruff format .             # Format
uv run ruff check . --fix        # Lint
uv run mypy src/                 # Type-check
```

## Project Layout

```
iceaggr/
├── src/iceaggr/
│   ├── data/              # Dataset, collators, geometry, BatchAwareSampler
│   ├── models/
│   │   ├── flat_transformer_v2.py   # Main model (nanochat-style)
│   │   ├── flat_transformer.py      # v1 (kept for reference)
│   │   ├── directional_head.py      # Unit-vector head → (az, zen)
│   │   ├── vmf_loss.py              # vMF mixture head + NLL loss
│   │   └── losses.py                # Angular-distance loss, vector ↔ angles
│   └── utils/             # Color-coded logger
├── configs/               # Experiment configs (committed)
├── scripts/
│   ├── train_flat.py      # Main training entry point
│   ├── inference.py       # Run a trained checkpoint on held-out batches
│   └── …                  # Analysis / profiling scripts
├── notes/                 # Experiment write-ups (committed)
└── tests/                 # Unit + integration tests
```

Archived hierarchical-model configs live under `archive/configs/`.

## Experiment Tracking

```python
import wandb
wandb.init(project="iceaggr", name="v2-none-K84-5M-yourname")
wandb.log({"val/loss": loss, "val/angular_error_deg": err_deg})
```

The training script logs `val/angular_error_rad` and `val/angular_error_deg` for every validation pass — use those for apples-to-apples comparison across runs even when the training loss differs (angular-distance vs. vMF NLL).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

```bash
git checkout main && git pull && uv sync
git checkout -b experiment/your-idea
# ... code ...
uv run pytest && uv run ruff check .
git push origin experiment/your-idea
```

## Resources

- [IceCube Kaggle competition](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice)
- [UV docs](https://docs.astral.sh/uv/)
- [Weights & Biases docs](https://docs.wandb.ai/)

---

## Setup

### Install UV

<details>
<summary>Click to expand</summary>

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Then
source $HOME/.local/bin/env   # or restart your terminal
git clone https://github.com/timinar/iceaggr.git
cd iceaggr
uv sync
```

</details>

### Download IceCube Kaggle data

<details>
<summary>Click to expand</summary>

```bash
uv add kaggle
# Get an API token at kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c icecube-neutrinos-in-deep-ice
unzip icecube-neutrinos-in-deep-ice.zip -d data/
```

</details>

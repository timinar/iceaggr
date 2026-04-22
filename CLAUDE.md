# CLAUDE.md

Project-specific context for working in this repository. The [README](README.md) covers what the project is and how to run it — this file captures conventions, data layout, and gotchas that aren't obvious from the code alone.

## Architecture in one paragraph

One flat transformer over **DOM vectors**. For every active DOM, the first K=84 pulses (time/charge/aux triplets) are concatenated into a single fixed-length vector and prepended with `[x, y, z, n_pulses]`. The event becomes a short sequence of such vectors plus a CLS token, processed by a single nanochat-style transformer (RMSNorm, QK-norm, ReLU², zero-init proj). The CLS embedding drives either a directional head (angular-distance loss) or a **vMF mixture head** (NLL, κ-weighted mean direction for point estimates).

This is intentionally *not* hierarchical — there is no per-DOM inner transformer. Concatenation collapses the pulse axis before attention, saving ~K× in sequence length vs. a naive two-stage model. Any references to "T1 / T2 / hierarchical" in old notes or `archive/configs/` describe a deprecated architecture.

Main files: [src/iceaggr/models/flat_transformer_v2.py](src/iceaggr/models/flat_transformer_v2.py), [src/iceaggr/models/vmf_loss.py](src/iceaggr/models/vmf_loss.py), [src/iceaggr/models/losses.py](src/iceaggr/models/losses.py), [scripts/train_flat.py](scripts/train_flat.py).

## Data paths

All data lives at `/groups/pheno/inar/icecube_kaggle/`. User-local paths belong in [src/iceaggr/data/data_config.yaml](src/iceaggr/data/data_config.yaml) (gitignored) — **do not hard-code paths in scripts**.

| Path | What |
|---|---|
| `train/batch_*.parquet` | Pulse data, ~20GB compressed |
| `train_meta/train_meta_*.parquet` | `event_id, first_pulse_index, last_pulse_index, azimuth, zenith` |
| `sensor_geometry_normalized.csv` | DOM positions divided by 500 (use this one) |
| `sensor_geometry.csv` | Raw DOM positions |
| `ice_transparency.txt` | Ice optical properties |

### Train/val split convention

- Ours: train = batches **1–650**, val = batches **651–655**
- 2nd-place (DrHB) used batches 655–659 as their val
- Batches **656–659** are unseen by both → best set for head-to-head comparison

## Data quirks worth remembering

- 50–70% of DOMs have ≤10 pulses (sparse tail). Truncating to K=84 is cheap.
- 99th percentile: <2000 active DOMs per event; `max_doms=128` covers the vast majority (selection is by earliest pulse time).
- Heavy tail: rare events have 10K+ pulses. The flat architecture handles these via DOM truncation, no special path needed.
- **Use Polars or PyArrow, not pandas.** IceCube parquets are large; pandas is 10–100× slower and eats memory. Exception: tiny result tables (<1K rows) are fine in pandas.

## Configuration pattern

Two tiers:

1. **Data paths** → [src/iceaggr/data/data_config.yaml](src/iceaggr/data/data_config.yaml), gitignored, local per user.
2. **Experiments** → [configs/](configs/), committed. Naming: `train_flat_v2_<input_mode>_K<K>_<size>[_<variant>].yaml`.

Load data config:
```python
import yaml
with open("src/iceaggr/data/data_config.yaml") as f:
    paths = yaml.safe_load(f)["data"]
```

## Checkpoint format

Training wraps the model with `torch.compile`, so `state_dict` keys have an `_orig_mod.` prefix. Inference scripts must strip it:

```python
ckpt = torch.load(path)
state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
```

Current best: `checkpoints/v2-none-K84-5M-proper-split-10ep/best.pt` (val loss 0.968 rad ≈ 55.5° mean, 48° median). vMF run in progress as of 2026-04-22.

Keys in checkpoint dicts: `model`, `optimizer`, `scheduler`, `epoch`, `batch_idx` (nullable — mid-epoch resume), `train_loss`, `val_loss`, `val_angular_error_rad`, `best_loss`, `config`.

## Logging

Use the project's color-coded logger rather than `print`:

```python
from iceaggr.utils import get_logger
logger = get_logger(__name__)
logger.info("Loading batch 42")
```

Levels: DEBUG (blue), INFO (green), WARNING (yellow), ERROR (red). See [src/iceaggr/utils/logger_config.py](src/iceaggr/utils/logger_config.py).

## wandb

Always log `val/angular_error_rad` and `val/angular_error_deg` — these are comparable across loss types (angular-distance vs. vMF NLL), and `val/loss` alone is not.

```python
wandb.init(project="iceaggr", name="v2-<variant>-<size>-<yourname>")
```

## Dev workflow

```bash
git checkout main && git pull && uv sync      # always start here
git checkout -b experiment/<short-name>        # or feature/, bugfix/, analysis/
# work…
uv run pytest && uv run ruff check .           # before pushing
```

Commit messages explain *why*, not *what* (the diff shows the what).

## Next steps

- [x] Dataloader with DOM grouping, flat collator
- [x] FlatTransformerV2 + training pipeline
- [x] 5M / 10M scale-up runs
- [x] vMF mixture head + NLL training path
- [ ] spline-mpe baseline comparison
- [ ] Paper figures

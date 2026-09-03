#!/usr/bin/env python3
"""
Training script for flat transformer model.

QUICK START
-----------
# Train with config
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_flat.py --config configs/train_flat.yaml

# Override config params via CLI
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_flat.py --config configs/train_flat.yaml --lr 1e-3 --epochs 20

# Quick sanity check (real data, small scale)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_flat.py --config configs/train_flat.yaml --max-events 10000 --epochs 1 --no-wandb

# Run in background
screen -dmS flat_train bash -c 'CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_flat.py --config configs/train_flat.yaml 2>&1 | tee training_flat.log'

EXPECTED RESULTS
----------------
- Random baseline: ~90° angular error (1.57 rad)
- Loss should decrease within the first epoch if the model is learning
"""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from iceaggr.data import (
    IceCubeDataset,
    GeometryLoader,
    make_collate_flat,
    make_collate_npe15,
    make_collate_hybrid,
    BatchAwareSampler,
)
from iceaggr.models import (
    FlatTransformerModel,
    FlatTransformerV2,
    angular_distance_loss,
    angles_to_unit_vector,
)
from iceaggr.utils import get_logger
from iceaggr.utils.muon import Muon, MultiOptimizer, MultiScheduler, split_muon_params

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_flat.yaml", help="Config file path")

    # CLI overrides for common parameters
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--max-events", type=int, default=None, help="Override max training events")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint to resume")
    parser.add_argument("--workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--val-per-epoch", type=int, default=None, help="Override val_per_epoch")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument(
        "--ablate", type=str, default=None,
        choices=["none", "vanilla", "rmsnorm", "qknorm", "relu2", "zeroinit", "residscaling", "bias"],
        help="Per-change architecture ablation: disable one N2-T2 feature, "
             "'vanilla' (all off) or 'none' (full N2-T2).",
    )

    return parser.parse_args()


# Map an --ablate name to the model config flag it toggles off.
_ABLATE_FLAG = {
    "rmsnorm": "use_rmsnorm",
    "qknorm": "use_qknorm",
    "relu2": "use_relu2",
    "zeroinit": "use_zero_init",
    "residscaling": "use_resid_scaling",
}


def apply_ablation(config: dict, ablate: str) -> dict:
    """Translate --ablate into model-config flags (per-change ablation)."""
    if ablate in (None, "none"):
        return config
    m = config.setdefault('model', {})
    if ablate == "vanilla":
        m['use_rmsnorm'] = False
        m['use_qknorm'] = False
        m['use_relu2'] = False
        m['use_zero_init'] = False
        m['use_resid_scaling'] = False
        m['use_bias'] = True
    elif ablate == "bias":
        m['use_bias'] = True  # disable the no-bias feature (add biases back)
    else:
        m[_ABLATE_FLAG[ablate]] = False
    return config


def apply_cli_overrides(config: dict, args) -> dict:
    """Apply CLI argument overrides to config."""
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.max_events is not None:
        config['data']['max_events'] = args.max_events
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.checkpoint is not None:
        config['checkpoint']['resume'] = args.checkpoint
    if args.no_wandb:
        config['wandb']['enabled'] = False
    if args.workers is not None:
        config['data']['num_workers'] = args.workers
    if args.name is not None:
        config['wandb']['name'] = args.name
    if args.val_per_epoch is not None:
        config.setdefault('data', {})['val_per_epoch'] = args.val_per_epoch
    return config


def create_model(config: dict, device: str) -> nn.Module:
    """Create the flat transformer model from config."""
    model_config = {
        "max_pulses_per_dom": config['model']['max_pulses_per_dom'],
        "d_model": config['model']['d_model'],
        "max_doms": config['model']['max_doms'],
        "num_heads": config['model']['num_heads'],
        "num_layers": config['model']['num_layers'],
        "hidden_dim": config['model']['hidden_dim'],
        "head_hidden_dim": config['model']['head_hidden_dim'],
        "dropout": config['model']['dropout'],
    }

    version = config['model'].get('version', 'v1')
    if version == 'v2':
        model_config['input_mode'] = config['model'].get('input_mode', 'mlp')
        for key in (
            # explicit input dim override for non-flat tokenizations (e.g. npe15)
            'input_dim',
            'head_type',
            'vmf_components',
            'vmf_kappa_min',
            'vmf_kappa_max',
            'vmf_kappa_reg',
            # opt-in combined loss: NLL + vmf_angular_weight · angular-distance
            'vmf_angular_weight',
            # per-change architecture ablation flags (default to N2-T2 behavior)
            'use_rmsnorm',
            'use_qknorm',
            'use_relu2',
            'use_zero_init',
            'use_resid_scaling',
            'use_bias',
            # relative spacetime-interval attention bias (opt-in; default off)
            'use_spacetime_bias',
            'spacetime_bias_hidden_dim',
            # rotary position embedding (opt-in; default off)
            'use_rope',
            'rope_theta',
        ):
            if key in config['model']:
                model_config[key] = config['model'][key]
        model = FlatTransformerV2(model_config)
        head_type = model_config.get('head_type', 'directional')
        logger.info(
            f"Using FlatTransformerV2 (input_mode={model_config['input_mode']}, head={head_type})"
        )
    else:
        model = FlatTransformerModel(model_config)
        logger.info("Using FlatTransformerModel (v1)")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    return model


def _worker_init_fn(worker_id: int) -> None:
    """Pin each DataLoader worker to a single torch intra-op thread.

    Two problems, one fix:

    1. **Oversubscription.** The flat collator is torch-heavy (cat / unique /
       argsort / scatter, all multithreaded). By default every worker inherits
       torch's process-wide num_threads (≈ half the core count), so ``num_workers``
       workers spawn ``num_workers × threads`` intra-op threads on ``cores`` CPUs —
       e.g. 8 workers × 32 = 256 threads on 64 cores — which thrashes the scheduler
       and starves the GPU.
    2. **Fork-safety.** torch's intra-op (OpenMP) threadpool is not fork-safe: a
       worker that runs a *multi-threaded* torch op after fork can **deadlock** on
       the first `torch.unique`/`argsort` (observed here — it hangs intermittently
       depending on the parent's pool state at fork). Single-threaded ops never
       touch the parallel pool, so `set_num_threads(1)` removes the hazard
       entirely.

    One collate at 1 thread already runs faster per core than at 8/32 (the collate
    parallelizes only sub-linearly), so this costs nothing in aggregate throughput
    with a handful of workers — scale cores by adding *workers* (processes are
    fork-safe), not threads.
    """
    torch.set_num_threads(1)


def create_dataloader(
    config: dict,
    geometry: GeometryLoader,
    split: str = 'train',
    batch_range: tuple = None,
    max_events: int = None,
    num_workers: int = None,
) -> DataLoader:
    """Create dataloader from config.

    Args:
        config: Full config dict
        geometry: GeometryLoader instance
        split: 'train' or 'test'
        batch_range: Optional (min_batch, max_batch) inclusive for batch_id filtering
        max_events: Override max_events (default: from config)
        num_workers: Override num_workers (default: from config)
    """
    if max_events is None:
        max_events = config['data']['max_events'] if split == 'train' else config['data'].get('val_events', 50000)
    if num_workers is None:
        num_workers = config['data']['num_workers']

    dataset = IceCubeDataset(
        split=split,
        max_events=max_events,
        cache_size=1,
        batch_range=batch_range,
        min_pulses=config['data'].get('min_pulses'),
    )

    # training.seed (optional) also drives the shuffle order; unset/null keeps
    # every historical run's order byte-identical (sampler seed 42).
    _seed = config['training'].get('seed')
    sampler = BatchAwareSampler(dataset.metadata, seed=42 if _seed is None else int(_seed))
    # Tokenization switch: the default flat pulse-concat tokens, or the 15-dim
    # NPE summary-statistics tokens for the encoding comparison. Default keeps the
    # flat path byte-identical.
    tokenization = config['data'].get('tokenization', 'flat')
    # opt-in bf16 output + vectorized deterministic subsample (default False =
    # byte-identical). Not byte-identical when on — validate before flipping.
    fast_collate = config['data'].get('fast_collate', False)
    if tokenization == 'npe15':
        collate_fn = make_collate_npe15(
            geometry,
            max_doms=config['model']['max_doms'],
            # geometry is the /500-normalized file → positions already normalized
            normalize_positions=config['data'].get('npe_normalize_positions', False),
            correct_percentiles=config['data'].get('npe_correct_percentiles', False),
            fast_collate=fast_collate,
        )
    elif tokenization == 'hybrid':
        # 256-dim raw+aggregate DOM tokens (input_mode none/linear, no input_dim
        # needed: default 4+3*max_pulses_per_dom=256 with max_pulses_per_dom=84).
        collate_fn = make_collate_hybrid(
            geometry,
            max_doms=config['model']['max_doms'],
            mode=config['data'].get('hybrid_mode', 'full'),
            include_event_context=config['data'].get('hybrid_include_event_context', True),
            normalize_positions=config['data'].get('npe_normalize_positions', False),
            correct_percentiles=config['data'].get('npe_correct_percentiles', False),
            fast_collate=fast_collate,
        )
    elif tokenization == 'flat':
        collate_fn = make_collate_flat(
            geometry,
            max_pulses_per_dom=config['model']['max_pulses_per_dom'],
            max_doms=config['model']['max_doms'],
            # opt-in: order DOM tokens by earliest-hit time so RoPE sees hit rank
            order_doms_by_time=config['model'].get('order_doms_by_time', False),
            fast_collate=fast_collate,
        )
    else:
        raise ValueError(
            f"Unknown data.tokenization: {tokenization!r} (use 'flat', 'npe15', or 'hybrid')"
        )

    # Seeded runs give the loader its own RNG so building an iterator (which
    # draws the workers' base seed) never consumes the global torch RNG — that
    # keeps the dropout stream continuous across a checkpoint resume. Unseeded
    # runs keep the historical behaviour (global RNG).
    loader_gen = None
    if _seed is not None:
        loader_gen = torch.Generator()
        loader_gen.manual_seed(int(_seed))
    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=loader_gen,
    )

    return loader, sampler


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module through torch.compile's OptimizedModule."""
    return getattr(model, '_orig_mod', model)


def _rng_state() -> dict:
    """Snapshot every RNG a resume needs to continue dropout/shuffle streams."""
    import random
    import numpy as np
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    import random
    import numpy as np
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _ckpt_extra(scaler) -> dict:
    """RNG + GradScaler state stored in every checkpoint (restored on resume)."""
    return {'rng': _rng_state(), 'scaler': scaler.state_dict()}


def _amp_dtype(config: dict) -> torch.dtype:
    """Pick autocast dtype from config (default fp16; 'bf16' opt-in)."""
    name = str(config['training'].get('amp_dtype', 'fp16')).lower()
    return torch.bfloat16 if name in ('bf16', 'bfloat16') else torch.float16


def _one_cycle(optimizer, max_lr, total_steps, pct_start):
    """OneCycleLR with the project's fixed div-factor conventions."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        div_factor=25.0,          # Initial LR = max_lr / 25
        final_div_factor=1000.0,  # Final LR = max_lr / 1000
    )


def build_optimizer_and_scheduler(model, config, total_steps, pct_start):
    """Build the optimizer + OneCycleLR scheduler from config.

    ``training.optimizer`` selects the path:

    - ``adamw`` (default): a single AdamW over all parameters wrapped by one
      OneCycleLR — byte-identical to the historical training setup.
    - ``muon``: Muon on the 2D weight matrices inside ``model.blocks`` (attention
      + FFN), with an AdamW side-group on everything else (CLS token, input
      projection, head, per-layer scalars, norms/biases). Each optimizer gets its
      own OneCycleLR sharing the same schedule shape (``total_steps``/``pct_start``),
      driven together through MultiOptimizer / MultiScheduler. Muon uses
      ``training.muon_lr`` (default 0.02) / ``training.muon_momentum`` (default
      0.95); the AdamW side-group keeps ``training.lr``.
    """
    lr = float(config['training']['lr'])
    weight_decay = float(config['training']['weight_decay'])
    opt_name = str(config['training'].get('optimizer', 'adamw')).lower()

    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = _one_cycle(optimizer, lr, total_steps, pct_start)
        return optimizer, scheduler

    if opt_name == 'muon':
        # fp16 loss-scaling is not threaded through MultiOptimizer; Muon (like
        # nanochat) runs under bf16, so require it rather than train silently wrong.
        if config['training'].get('use_amp', False) and _amp_dtype(config) == torch.float16:
            raise ValueError(
                "training.optimizer: muon requires training.amp_dtype: bf16 "
                "(the fp16 GradScaler path is not wired through MultiOptimizer)."
            )
        muon_lr = float(config['training'].get('muon_lr', 0.02))
        muon_momentum = float(config['training'].get('muon_momentum', 0.95))
        muon_params, adamw_params, muon_names = split_muon_params(model)
        logger.info(
            f"Muon optimizer: {len(muon_params)} block weight matrices → Muon "
            f"(lr={muon_lr}, momentum={muon_momentum}); "
            f"{len(adamw_params)} params → AdamW (lr={lr})"
        )
        adamw_opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
        muon_opt = Muon(muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay)
        # AdamW first so scheduler.get_last_lr()[0] stays the base LR (as in every
        # AdamW run); the two schedulers move in lockstep on the same geometry.
        optimizer = MultiOptimizer([adamw_opt, muon_opt])
        scheduler = MultiScheduler([
            _one_cycle(adamw_opt, lr, total_steps, pct_start),
            _one_cycle(muon_opt, muon_lr, total_steps, pct_start),
        ])
        return optimizer, scheduler

    raise ValueError(f"Unknown training.optimizer: {opt_name!r} (use 'adamw' or 'muon')")


def compute_kappa_reg(config: dict, global_step: int, total_steps: int) -> float:
    """Linear schedule for vMF kappa_reg.

    Reads vmf_kappa_reg (start), vmf_kappa_reg_final (end), and
    vmf_kappa_reg_anneal_fraction from the model config. Default behavior
    (no anneal keys set) returns vmf_kappa_reg unchanged.
    """
    m = config['model']
    reg_start = float(m.get('vmf_kappa_reg', 1e-4))
    reg_final = float(m.get('vmf_kappa_reg_final', reg_start))
    frac = float(m.get('vmf_kappa_reg_anneal_fraction', 0.0))
    if reg_final == reg_start or frac <= 0.0:
        return reg_start
    anneal_steps = max(1, int(frac * total_steps))
    progress = min(1.0, global_step / anneal_steps)
    return reg_start + (reg_final - reg_start) * progress


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    epoch: int,
    config: dict,
    wandb_run=None,
    val_loader: DataLoader = None,
    val_interval: int = None,
    best_loss: float = float('inf'),
    checkpoint_dir: Path = None,
    start_batch: int = 0,
    grad_accum: int = 1,
) -> tuple:
    """Train for one epoch with optional mid-epoch validation.

    Returns:
        (train_loss, best_loss) tuple
    """
    model.train()

    total_loss = 0.0
    n_batches = len(loader)  # full epoch batch count (for progress display)
    n_trained = 0
    start_time = time.time()
    head_type = config['model'].get('head_type', 'directional')
    total_steps = n_batches * config['training']['epochs']
    current_kappa_reg = None  # set each step when head_type == 'vmf'
    window_gn_sum, window_clipped, window_steps = 0.0, 0, 0  # grad-norm window stats

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        actual_batch = start_batch + batch_idx
        dom_vectors = batch['dom_vectors'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        targets = batch['targets'].to(device)

        # vMF: apply kappa_reg schedule (in-place on buffer so torch.compile
        # sees the update without recompiling).
        if head_type == 'vmf':
            global_step = (epoch - 1) * n_batches + actual_batch
            current_kappa_reg = compute_kappa_reg(config, global_step, total_steps)
            _unwrap(model).vmf_loss.kappa_reg.fill_(current_kappa_reg)

        # Forward with AMP
        with torch.autocast(device_type='cuda', dtype=_amp_dtype(config), enabled=config['training']['use_amp']):
            if head_type == 'vmf':
                target_unit = angles_to_unit_vector(targets[:, 0], targets[:, 1])
                outputs = model(dom_vectors, padding_mask, target=target_unit)
                loss = outputs['loss']
            else:
                outputs = model(dom_vectors, padding_mask)
                y_pred = outputs['direction'] if isinstance(outputs, dict) else outputs
                loss = angular_distance_loss(y_pred, targets)

        # Backward with gradient scaling. Divide by the accumulation window size so
        # the summed gradient is a mean; the epoch's last window may be shorter
        # (n_batches % grad_accum micro-batches) and is divided by its own size.
        tail = n_batches % grad_accum
        denom = tail if (tail and actual_batch >= n_batches - tail) else grad_accum
        scaler.scale(loss / denom).backward()

        # Optimizer step every grad_accum micro-batches (and at epoch end). The
        # epoch-end flush is keyed on actual_batch: after a mid-epoch resume the
        # loader yields fewer batches than len(loader) reports.
        if (batch_idx + 1) % grad_accum == 0 or (actual_batch + 1) == n_batches:
            scaler.unscale_(optimizer)
            clip = float(config['training']['gradient_clip'])
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            # Pre-clip norm + clip rate since the last log line (scaling-ladder
            # diagnostic: a >10% clip rate means LR/clip need tuning, not the model).
            gn = float(grad_norm)
            if math.isfinite(gn):
                window_gn_sum += gn
                window_clipped += int(gn > clip)
                window_steps += 1
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # OneCycleLR steps per optimizer step
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n_trained += 1

        # Log to wandb
        if wandb_run is not None and (actual_batch + 1) % 25 == 0:
            step = (epoch - 1) * n_batches + actual_batch
            current_lr = scheduler.get_last_lr()[0]
            log_payload = {
                "train/loss": loss.item(),
                "train/lr": current_lr,
            }
            if window_steps > 0:
                log_payload["train/grad_norm"] = window_gn_sum / window_steps
                log_payload["train/clip_frac"] = window_clipped / window_steps
                window_gn_sum, window_clipped, window_steps = 0.0, 0, 0
            if head_type != 'vmf':
                log_payload["train/loss_deg"] = torch.rad2deg(torch.tensor(loss.item())).item()
            else:
                log_payload["train/kappa_reg"] = current_kappa_reg
            wandb_run.log(log_payload, step=step)

        # Progress every 25 batches
        if (actual_batch + 1) % 25 == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / n_trained
            speed = n_trained / elapsed
            if head_type == 'vmf':
                loss_str = f"NLL: {avg_loss:.4f}"
            else:
                loss_str = f"Loss: {avg_loss:.4f} ({torch.rad2deg(torch.tensor(avg_loss)):.1f} deg)"
            logger.info(
                f"Epoch {epoch:3d} | Batch {actual_batch+1:4d}/{n_batches} | "
                f"{loss_str} | {speed:.1f} b/s"
            )

        # Mid-epoch validation
        if val_loader is not None and val_interval is not None and (actual_batch + 1) % val_interval == 0:
            val_metrics = validate(model, val_loader, device, config)
            val_loss = val_metrics['loss']
            val_ang_err = val_metrics['angular_error_rad']
            val_ang_deg = torch.rad2deg(torch.tensor(val_ang_err)).item()
            logger.info(
                f"Epoch {epoch:3d} | Mid-epoch val @ batch {actual_batch+1}/{n_batches} | "
                f"Val loss: {val_loss:.4f} | Angular err: {val_ang_deg:.2f} deg"
                + _val_extra_str(val_metrics)
            )

            # Log to wandb
            if wandb_run is not None:
                step = (epoch - 1) * n_batches + actual_batch
                wandb_run.log({
                    "val/loss": val_loss,
                    "val/angular_error_rad": val_ang_err,
                    "val/angular_error_deg": val_ang_deg,
                    **_val_extra_payload(val_metrics),
                }, step=step)

            # Save best model checkpoint (by training-objective loss)
            if val_loss < best_loss and checkpoint_dir is not None:
                best_loss = val_loss
                checkpoint_path = checkpoint_dir / "best.pt"
                torch.save({
                    'epoch': epoch,
                    'batch_idx': actual_batch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_loss': total_loss / n_trained,
                    'val_loss': val_loss,
                    'val_angular_error_rad': val_ang_err,
                    'best_loss': best_loss,
                    'config': config,
                    **_ckpt_extra(scaler),
                }, checkpoint_path)
                logger.info(f"Saved best model (val angular err: {val_ang_deg:.2f} deg)")

            # Switch back to training mode
            model.train()

    return total_loss / n_trained if n_trained > 0 else 0.0, best_loss


# Pulse-count thresholds for the sliced validation metrics (val/angular_error_deg_geN).
VAL_SLICES = (200, 1000)


def validate(model: nn.Module, loader: DataLoader, device: str, config: dict) -> dict:
    """Validate on the full validation set.

    Returns a dict with at least
        'loss'              training-objective loss (batch-mean, as logged historically)
        'angular_error_rad' mean per-event angular error
    plus, for the scaling ladder,
        'angular_error_median_rad'
        'nll'               vMF only: loss minus the λ·angular term (≈ pure NLL)
        'angular_error_rad_ge<N>' / 'n_ge<N>'  for N in VAL_SLICES, when the collator
                            supplies 'n_pulses' (flat collator does; others skip).
    For the directional head 'loss' and 'angular_error_rad' coincide up to the
    batch-mean vs event-mean weighting of the last partial batch.
    """
    model.eval()

    total_loss = 0.0
    total_ang_err = 0.0
    n_batches = 0
    head_type = config['model'].get('head_type', 'directional')
    errs, nlls, n_pulses = [], [], []
    vmf_loss_mod = _unwrap(model).vmf_loss if head_type == 'vmf' else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            dom_vectors = batch['dom_vectors'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            targets = batch['targets'].to(device)

            with torch.autocast(device_type='cuda', dtype=_amp_dtype(config), enabled=config['training']['use_amp']):
                if head_type == 'vmf':
                    target_unit = angles_to_unit_vector(targets[:, 0], targets[:, 1])
                    outputs = model(dom_vectors, padding_mask, target=target_unit)
                    loss = outputs['loss']
                    y_pred = outputs['direction']
                    ang_err = angular_distance_loss(y_pred, targets)
                else:
                    outputs = model(dom_vectors, padding_mask)
                    y_pred = outputs['direction'] if isinstance(outputs, dict) else outputs
                    loss = angular_distance_loss(y_pred, targets)
                    ang_err = loss
            # Historical metric: batch-mean of the per-batch mean angular error,
            # computed under autocast exactly as every run before 2026-09 did, so
            # val/angular_error_rad stays comparable across the whole project.
            total_ang_err += ang_err.item()

            # Per-event angular error in fp32 (same epsilon as angular_distance_loss)
            y_true = angles_to_unit_vector(targets[:, 0], targets[:, 1]).float()
            dot = (torch.nn.functional.normalize(y_pred.float(), dim=-1) * y_true).sum(dim=-1)
            errs.append(torch.arccos(dot.clamp(-1 + 1e-4, 1 - 1e-4)).abs().cpu())
            if vmf_loss_mod is not None:
                # pure per-event mixture NLL: no κ penalty, no angular term
                nlls.append(vmf_loss_mod.per_event_nll(
                    outputs['mu'], outputs['raw_kappa'], outputs['log_weights'], y_true).cpu())
            if 'n_pulses' in batch:
                n_pulses.append(batch['n_pulses'].cpu())

            total_loss += loss.item()
            n_batches += 1

    if n_batches == 0:
        return {'loss': 0.0, 'angular_error_rad': 0.0}
    errs = torch.cat(errs)
    out = {
        'loss': total_loss / n_batches,
        'angular_error_rad': total_ang_err / n_batches,      # historical (batch-mean)
        'angular_error_event_rad': errs.mean().item(),       # event-weighted, fp32
        'angular_error_median_rad': errs.median().item(),
    }
    nlls = torch.cat(nlls) if nlls else None
    if nlls is not None:
        out['nll'] = nlls.mean().item()
    if n_pulses:
        n_pulses = torch.cat(n_pulses)
        for th in VAL_SLICES:
            m = n_pulses >= th
            if bool(m.any()):
                out[f'angular_error_rad_ge{th}'] = errs[m].mean().item()
                out[f'angular_error_median_rad_ge{th}'] = errs[m].median().item()
                out[f'n_ge{th}'] = int(m.sum())
                if nlls is not None:
                    out[f'nll_ge{th}'] = nlls[m].mean().item()
    return out


def _val_extra_payload(val_metrics: dict) -> dict:
    """wandb payload for the extra validate() keys (median, nll, pulse slices)."""
    payload = {}
    if 'angular_error_event_rad' in val_metrics:
        payload['val/angular_error_event_deg'] = math.degrees(val_metrics['angular_error_event_rad'])
    if 'angular_error_median_rad' in val_metrics:
        payload['val/angular_error_median_deg'] = math.degrees(val_metrics['angular_error_median_rad'])
    if 'nll' in val_metrics:
        payload['val/nll'] = val_metrics['nll']
    for th in VAL_SLICES:
        k = f'angular_error_rad_ge{th}'
        if k in val_metrics:
            payload[f'val/angular_error_deg_ge{th}'] = math.degrees(val_metrics[k])
            payload[f'val/angular_error_median_deg_ge{th}'] = math.degrees(
                val_metrics[f'angular_error_median_rad_ge{th}'])
            payload[f'val/n_ge{th}'] = val_metrics[f'n_ge{th}']
            if f'nll_ge{th}' in val_metrics:
                payload[f'val/nll_ge{th}'] = val_metrics[f'nll_ge{th}']
    return payload


def _val_extra_str(val_metrics: dict) -> str:
    """Compact log-line suffix for the extra validate() keys."""
    parts = []
    if 'angular_error_median_rad' in val_metrics:
        parts.append(f"med {math.degrees(val_metrics['angular_error_median_rad']):.2f}")
    if 'nll' in val_metrics:
        parts.append(f"nll {val_metrics['nll']:.4f}")
    for th in VAL_SLICES:
        k = f'angular_error_rad_ge{th}'
        if k in val_metrics:
            parts.append(f">={th}: {math.degrees(val_metrics[k]):.2f}/"
                         f"{math.degrees(val_metrics[f'angular_error_median_rad_ge{th}']):.2f}"
                         f" (n={val_metrics[f'n_ge{th}']})")
    return (" | " + " | ".join(parts)) if parts else ""


def main():
    args = parse_args()

    # Load and merge config
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    config = apply_ablation(config, args.ablate)
    if args.grad_accum is not None:
        config['training']['grad_accum_steps'] = args.grad_accum

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Optional cap on the MAIN process's intra-op threads (training.main_threads).
    # By default torch keeps a pool of ~half the cores that spin-waits while the
    # GPU runs; with several trainers per box (A100 fleet) those pools starve the
    # dataloader workers. 4–8 is plenty for the main process. Unset → unchanged.
    main_threads = config['training'].get('main_threads')
    if main_threads:
        torch.set_num_threads(int(main_threads))
        logger.info(f"Main-process torch threads capped at {int(main_threads)}")

    # Optional seed (training.seed): init + dropout + shuffle order. Unset →
    # historical behaviour (unseeded init, sampler seed 42).
    seed = config['training'].get('seed')
    if seed is not None:
        import random
        import numpy as np
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))  # also seeds CUDA
        logger.info(f"Seeded python/numpy/torch with training.seed={seed}")

    # Initialize wandb
    wandb_run = None
    if config['wandb']['enabled']:
        try:
            import wandb
            run_name = config['wandb']['name'] or f"flat-{datetime.now().strftime('%m%d-%H%M')}"
            wandb_run = wandb.init(
                project=config['wandb']['project'],
                name=run_name,
                config=config,
                tags=config['wandb'].get('tags', []),
            )
            logger.info(f"W&B run: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
        except Exception as e:
            logger.warning(f"Failed to init wandb: {e}")

    # Determine run_name for checkpoint directory
    if wandb_run is not None:
        pass  # run_name already set above
    elif config['wandb'].get('name'):
        run_name = config['wandb']['name']
    else:
        run_name = f"flat-{datetime.now().strftime('%m%d-%H%M')}"

    # Load geometry
    geometry = GeometryLoader(config['data']['geometry_path'])
    logger.info(f"Loaded geometry: {geometry}")

    # Create model
    model = create_model(config, device)
    model = torch.compile(model)
    logger.info("Model compiled with torch.compile")

    # Create dataloaders
    train_batch_range = None
    val_batch_range = None
    if 'train_batches' in config['data']:
        train_batch_range = tuple(config['data']['train_batches'])
    if 'val_batches' in config['data']:
        val_batch_range = tuple(config['data']['val_batches'])

    loader, train_sampler = create_dataloader(config, geometry, batch_range=train_batch_range)
    logger.info(f"Training: {len(loader.dataset):,} events, {len(loader):,} batches, bs={config['training']['batch_size']}")

    # Create validation dataloader
    val_loader = None
    if val_batch_range is not None:
        val_loader, _ = create_dataloader(
            config, geometry,
            batch_range=val_batch_range,
            max_events=config['data'].get('val_events'),
            num_workers=2,
        )
        logger.info(f"Validation: {len(val_loader.dataset):,} events, {len(val_loader):,} batches")

    # Optional fixed train-eval set (scaling ladder): the first
    # data.train_eval_events events of data.train_eval_batches (default: the first
    # training batch), scored in eval mode after every epoch so the generalization
    # gap (val − train-eval) is measured on identical events, dropout off.
    train_eval_loader = None
    n_train_eval = config['data'].get('train_eval_events')
    if n_train_eval and train_batch_range is not None:
        te_range = tuple(config['data'].get('train_eval_batches', (train_batch_range[0], train_batch_range[0])))
        train_eval_loader, _ = create_dataloader(
            config, geometry, batch_range=te_range, max_events=int(n_train_eval), num_workers=2,
        )
        logger.info(f"Train-eval: {len(train_eval_loader.dataset):,} events from batches {te_range}")

    # LR schedule geometry - OneCycleLR with warmup (count OPTIMIZER steps, not micro-batches)
    grad_accum = max(1, int(config['training'].get('grad_accum_steps', 1)))
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = steps_per_epoch * config['training']['epochs']
    warmup_steps = config['training'].get('warmup_steps', 1000)
    if grad_accum > 1:
        logger.info(f"Gradient accumulation: {grad_accum} micro-batches/step "
                    f"(effective batch size {config['training']['batch_size'] * grad_accum})")
    pct_start = min(warmup_steps / total_steps, 0.3)  # Cap at 30% of training

    # Optimizer + scheduler (AdamW default; Muon opt-in via training.optimizer)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps, pct_start)

    # AMP scaler
    # GradScaler is fp16-only; bf16 has fp32-equivalent exponent range and
    # doesn't need loss scaling. Disabling it under bf16 is the standard pattern.
    scaler = torch.amp.GradScaler(
        enabled=config['training']['use_amp'] and _amp_dtype(config) == torch.float16
    )

    # Resume from checkpoint if provided
    start_epoch = 1
    best_loss = float('inf')
    resume_batch = 0
    resume_path = config['checkpoint'].get('resume')
    if resume_path and Path(resume_path).exists():
        # Our own checkpoints: they carry RNG state (numpy arrays, python tuples)
        # that torch>=2.6's weights_only default refuses to unpickle.
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        state = checkpoint['model']
        # Back-compat: legacy vMF checkpoints saved before kappa_reg became a
        # registered buffer don't carry it. Inject from current config so the
        # buffer-aware model loads cleanly. Handles both compiled (_orig_mod.)
        # and non-compiled key prefixes.
        head_type = config['model'].get('head_type', 'directional')
        if head_type == 'vmf':
            has_buf = any(k.endswith('vmf_loss.kappa_reg') for k in state)
            if not has_buf:
                prefix = '_orig_mod.' if any(k.startswith('_orig_mod.') for k in state) else ''
                state[f'{prefix}vmf_loss.kappa_reg'] = torch.tensor(
                    float(config['model'].get('vmf_kappa_reg', 1e-4))
                )
                logger.info(f"Injected vmf_loss.kappa_reg into legacy checkpoint (key: {prefix}vmf_loss.kappa_reg)")
        model.load_state_dict(state)
        if config['checkpoint'].get('finetune', False):
            # Fine-tune mode: load model weights only, start fresh optimizer+scheduler
            logger.info("Fine-tune mode: loaded model weights only, fresh optimizer+scheduler")
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_loss = checkpoint.get('best_loss', checkpoint.get('val_loss', float('inf')))
            # Continue the RNG streams and loss-scaler state (checkpoints written
            # before these keys existed resume as they always did).
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if 'rng' in checkpoint:
                _restore_rng_state(checkpoint['rng'])
                logger.info("Restored RNG + GradScaler state from checkpoint")

            resume_batch_idx = checkpoint.get('batch_idx')
            if resume_batch_idx is not None:
                # Mid-epoch resume: restart same epoch, skip processed batches
                start_epoch = checkpoint['epoch']
                resume_batch = resume_batch_idx + 1
                logger.info(f"Resumed mid-epoch {start_epoch} from batch {resume_batch_idx}, skipping {resume_batch} batches")
            else:
                # End-of-epoch resume: start next epoch
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Resumed from epoch {start_epoch-1}")

    # Determine validation interval for mid-epoch validation
    val_per_epoch = config['data'].get('val_per_epoch', 1)
    val_interval = None
    if val_loader is not None and val_per_epoch > 1:
        val_interval = max(1, len(loader) // val_per_epoch)
        # Mid-epoch checkpoints must land right after an optimizer step, so the
        # validation cadence is a multiple of the accumulation window.
        if grad_accum > 1:
            val_interval = max(grad_accum, (val_interval // grad_accum) * grad_accum)
        logger.info(f"Mid-epoch validation: {val_per_epoch} times/epoch (every {val_interval} batches)")

    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(exist_ok=True)

    # Per-run checkpoint subdirectory
    run_checkpoint_dir = checkpoint_dir / run_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {run_checkpoint_dir}")

    # Training loop
    logger.info(f"Starting training for {config['training']['epochs']} epochs...")
    logger.info(f"LR: {config['training']['lr']}, d_model: {config['model']['d_model']}, layers: {config['model']['num_layers']}")

    # Choose validation loader: use val_loader if available, else fall back to train loader
    effective_val_loader = val_loader if val_loader is not None else loader

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # Set sampler epoch for deterministic shuffling + optional skip
        skip_batches = resume_batch if epoch == start_epoch and resume_batch > 0 else 0
        train_sampler.set_epoch(epoch, skip_batches=skip_batches, batch_size=config['training']['batch_size'])

        # Train (scheduler steps per-batch inside train_epoch)
        train_loss, best_loss = train_epoch(
            model, loader, optimizer, scaler, scheduler, device, epoch, config, wandb_run,
            val_loader=val_loader,
            val_interval=val_interval,
            best_loss=best_loss,
            checkpoint_dir=run_checkpoint_dir,
            start_batch=skip_batches,
            grad_accum=grad_accum,
        )

        # End-of-epoch validation
        val_metrics = validate(model, effective_val_loader, device, config)
        val_loss = val_metrics['loss']
        val_ang_err = val_metrics['angular_error_rad']

        # Get current LR (scheduler already stepped per-batch)
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch summary
        train_deg = torch.rad2deg(torch.tensor(train_loss)).item()
        val_ang_deg = torch.rad2deg(torch.tensor(val_ang_err)).item()

        logger.info(
            f"Epoch {epoch:3d} done | "
            f"Train: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | Angular err: {val_ang_deg:.2f} deg | "
            f"LR: {current_lr:.2e}"
            + _val_extra_str(val_metrics)
        )

        # Eval-mode score on the fixed train-eval events (generalization gap)
        te_payload = {}
        if train_eval_loader is not None:
            te = validate(model, train_eval_loader, device, config)
            te_deg = math.degrees(te['angular_error_rad'])
            te_payload = {
                "train_eval/loss": te['loss'],
                "train_eval/angular_error_deg": te_deg,
                "train_eval/gap_loss": val_loss - te['loss'],
                "train_eval/gap_deg": val_ang_deg - te_deg,
                **{k.replace("val/", "train_eval/"): v for k, v in _val_extra_payload(te).items()},
            }
            if 'nll' in te and 'nll' in val_metrics:
                te_payload["train_eval/gap_nll"] = val_metrics['nll'] - te['nll']
            logger.info(
                f"Epoch {epoch:3d} train-eval | loss {te['loss']:.4f} | Angular err: {te_deg:.2f} deg"
                + _val_extra_str(te) + f" | gap(val-train) {val_loss - te['loss']:+.4f}"
            )

        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_loss_deg": train_deg,
                "val/loss": val_loss,
                "val/angular_error_rad": val_ang_err,
                "val/angular_error_deg": val_ang_deg,
                "lr": current_lr,
                **_val_extra_payload(val_metrics),
                **te_payload,
            })

        # Save checkpoint if best
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = run_checkpoint_dir / "best.pt"
            torch.save({
                'epoch': epoch,
                'batch_idx': None,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_angular_error_rad': val_ang_err,
                'best_loss': best_loss,
                'config': config,
                **_ckpt_extra(scaler),
            }, checkpoint_path)
            logger.info(f"Saved best model (val angular err: {val_ang_deg:.2f} deg)")

        # Save periodic checkpoint
        save_every = config['checkpoint'].get('save_every', 5)
        if epoch % save_every == 0:
            latest_path = run_checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'batch_idx': None,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_angular_error_rad': val_ang_err,
                'best_loss': best_loss,
                'config': config,
                **_ckpt_extra(scaler),
            }, latest_path)

    head_type = config['model'].get('head_type', 'directional')
    if head_type == 'vmf':
        logger.info(f"Training complete. Best val loss (NLL): {best_loss:.4f}")
    else:
        logger.info(f"Training complete. Best val loss: {torch.rad2deg(torch.tensor(best_loss)):.1f} deg")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

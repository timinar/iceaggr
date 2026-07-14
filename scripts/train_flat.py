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
    BatchAwareSampler,
)
from iceaggr.models import (
    FlatTransformerModel,
    FlatTransformerV2,
    angular_distance_loss,
    angles_to_unit_vector,
)
from iceaggr.utils import get_logger

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

    return parser.parse_args()


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

    sampler = BatchAwareSampler(dataset.metadata)
    # Tokenization switch: the default flat pulse-concat tokens, or the 15-dim
    # NPE summary-statistics tokens for the encoding comparison. Default keeps the
    # flat path byte-identical.
    tokenization = config['data'].get('tokenization', 'flat')
    if tokenization == 'npe15':
        collate_fn = make_collate_npe15(
            geometry,
            max_doms=config['model']['max_doms'],
            # geometry is the /500-normalized file → positions already normalized
            normalize_positions=config['data'].get('npe_normalize_positions', False),
            correct_percentiles=config['data'].get('npe_correct_percentiles', False),
        )
    elif tokenization == 'flat':
        collate_fn = make_collate_flat(
            geometry,
            max_pulses_per_dom=config['model']['max_pulses_per_dom'],
            max_doms=config['model']['max_doms'],
            # opt-in: order DOM tokens by earliest-hit time so RoPE sees hit rank
            order_doms_by_time=config['model'].get('order_doms_by_time', False),
        )
    else:
        raise ValueError(f"Unknown data.tokenization: {tokenization!r} (use 'flat' or 'npe15')")

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return loader, sampler


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module through torch.compile's OptimizedModule."""
    return getattr(model, '_orig_mod', model)


def _amp_dtype(config: dict) -> torch.dtype:
    """Pick autocast dtype from config (default fp16; 'bf16' opt-in)."""
    name = str(config['training'].get('amp_dtype', 'fp16')).lower()
    return torch.bfloat16 if name in ('bf16', 'bfloat16') else torch.float16


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

    for batch_idx, batch in enumerate(loader):
        actual_batch = start_batch + batch_idx
        dom_vectors = batch['dom_vectors'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()

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

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()

        # Step LR scheduler (OneCycleLR needs per-step updates)
        scheduler.step()

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
            )

            # Log to wandb
            if wandb_run is not None:
                step = (epoch - 1) * n_batches + actual_batch
                wandb_run.log({
                    "val/loss": val_loss,
                    "val/angular_error_rad": val_ang_err,
                    "val/angular_error_deg": val_ang_deg,
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
                }, checkpoint_path)
                logger.info(f"Saved best model (val angular err: {val_ang_deg:.2f} deg)")

            # Switch back to training mode
            model.train()

    return total_loss / n_trained if n_trained > 0 else 0.0, best_loss


def validate(model: nn.Module, loader: DataLoader, device: str, config: dict) -> dict:
    """Validate on the full validation set.

    Returns:
        {'loss': training-objective loss, 'angular_error_rad': mean angular error}
        For the directional head these are identical; for vMF they differ.
    """
    model.eval()

    total_loss = 0.0
    total_ang_err = 0.0
    n_batches = 0
    head_type = config['model'].get('head_type', 'directional')

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
                    ang_err = angular_distance_loss(outputs['direction'], targets)
                else:
                    outputs = model(dom_vectors, padding_mask)
                    y_pred = outputs['direction'] if isinstance(outputs, dict) else outputs
                    loss = angular_distance_loss(y_pred, targets)
                    ang_err = loss

            total_loss += loss.item()
            total_ang_err += ang_err.item()
            n_batches += 1

    if n_batches == 0:
        return {'loss': 0.0, 'angular_error_rad': 0.0}
    return {
        'loss': total_loss / n_batches,
        'angular_error_rad': total_ang_err / n_batches,
    }


def main():
    args = parse_args()

    # Load and merge config
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay']),
    )

    # LR scheduler - OneCycleLR with warmup
    total_steps = len(loader) * config['training']['epochs']
    warmup_steps = config['training'].get('warmup_steps', 1000)
    pct_start = min(warmup_steps / total_steps, 0.3)  # Cap at 30% of training

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config['training']['lr']),
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        div_factor=25.0,        # Initial LR = max_lr / 25
        final_div_factor=1000.0,  # Final LR = max_lr / 1000
    )

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
        checkpoint = torch.load(resume_path)
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

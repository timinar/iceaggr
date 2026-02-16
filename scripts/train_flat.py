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
    BatchAwareSampler,
)
from iceaggr.models import FlatTransformerModel, FlatTransformerV2, angular_distance_loss
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
        model = FlatTransformerV2(model_config)
        logger.info(f"Using FlatTransformerV2 (input_mode={model_config['input_mode']})")
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
    )

    sampler = BatchAwareSampler(dataset.metadata)
    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=config['model']['max_pulses_per_dom'],
        max_doms=config['model']['max_doms'],
    )

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

    return loader


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
) -> tuple:
    """Train for one epoch with optional mid-epoch validation.

    Returns:
        (train_loss, best_loss) tuple
    """
    model.train()

    total_loss = 0.0
    n_batches = len(loader)
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        dom_vectors = batch['dom_vectors'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()

        # Forward with AMP
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config['training']['use_amp']):
            y_pred = model(dom_vectors, padding_mask)
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

        # Log to wandb
        if wandb_run is not None and (batch_idx + 1) % 25 == 0:
            step = (epoch - 1) * n_batches + batch_idx
            current_lr = scheduler.get_last_lr()[0]
            wandb_run.log({
                "train/loss": loss.item(),
                "train/loss_deg": torch.rad2deg(torch.tensor(loss.item())).item(),
                "train/lr": current_lr,
            }, step=step)

        # Progress every 25 batches
        if (batch_idx + 1) % 25 == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / (batch_idx + 1)
            speed = (batch_idx + 1) / elapsed
            logger.info(
                f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{n_batches} | "
                f"Loss: {avg_loss:.4f} ({torch.rad2deg(torch.tensor(avg_loss)):.1f} deg) | "
                f"{speed:.1f} b/s"
            )

        # Mid-epoch validation
        if val_loader is not None and val_interval is not None and (batch_idx + 1) % val_interval == 0:
            val_loss = validate(model, val_loader, device, config)
            val_deg = torch.rad2deg(torch.tensor(val_loss)).item()
            logger.info(
                f"Epoch {epoch:3d} | Mid-epoch val @ batch {batch_idx+1}/{n_batches} | "
                f"Val: {val_loss:.4f} ({val_deg:.1f} deg)"
            )

            # Log to wandb
            if wandb_run is not None:
                step = (epoch - 1) * n_batches + batch_idx
                wandb_run.log({
                    "val/loss": val_loss,
                    "val/loss_deg": val_deg,
                }, step=step)

            # Save best model checkpoint
            if val_loss < best_loss and checkpoint_dir is not None:
                best_loss = val_loss
                checkpoint_path = checkpoint_dir / "best_flat_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_loss': total_loss / (batch_idx + 1),
                    'val_loss': val_loss,
                    'config': config,
                }, checkpoint_path)
                logger.info(f"Saved best model (val_loss: {val_deg:.1f} deg)")

            # Switch back to training mode
            model.train()

    return total_loss / n_batches, best_loss


def validate(model: nn.Module, loader: DataLoader, device: str, config: dict) -> float:
    """Validate on the full validation set."""
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            dom_vectors = batch['dom_vectors'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            targets = batch['targets'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config['training']['use_amp']):
                y_pred = model(dom_vectors, padding_mask)
                loss = angular_distance_loss(y_pred, targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


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

    # Load geometry
    geometry = GeometryLoader(config['data']['geometry_path'])
    logger.info(f"Loaded geometry: {geometry}")

    # Create model
    model = create_model(config, device)

    # Create dataloaders
    train_batch_range = None
    val_batch_range = None
    if 'train_batches' in config['data']:
        train_batch_range = tuple(config['data']['train_batches'])
    if 'val_batches' in config['data']:
        val_batch_range = tuple(config['data']['val_batches'])

    loader = create_dataloader(config, geometry, batch_range=train_batch_range)
    logger.info(f"Training: {len(loader.dataset):,} events, {len(loader):,} batches, bs={config['training']['batch_size']}")

    # Create validation dataloader
    val_loader = None
    if val_batch_range is not None:
        val_loader = create_dataloader(
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
    scaler = torch.amp.GradScaler(enabled=config['training']['use_amp'])

    # Resume from checkpoint if provided
    start_epoch = 1
    best_loss = float('inf')
    resume_path = config['checkpoint'].get('resume')
    if resume_path and Path(resume_path).exists():
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch-1}")

    # Determine validation interval for mid-epoch validation
    val_per_epoch = config['data'].get('val_per_epoch', 1)
    val_interval = None
    if val_loader is not None and val_per_epoch > 1:
        val_interval = max(1, len(loader) // val_per_epoch)
        logger.info(f"Mid-epoch validation: {val_per_epoch} times/epoch (every {val_interval} batches)")

    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    logger.info(f"Starting training for {config['training']['epochs']} epochs...")
    logger.info(f"LR: {config['training']['lr']}, d_model: {config['model']['d_model']}, layers: {config['model']['num_layers']}")

    # Choose validation loader: use val_loader if available, else fall back to train loader
    effective_val_loader = val_loader if val_loader is not None else loader

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # Train (scheduler steps per-batch inside train_epoch)
        train_loss, best_loss = train_epoch(
            model, loader, optimizer, scaler, scheduler, device, epoch, config, wandb_run,
            val_loader=val_loader,
            val_interval=val_interval,
            best_loss=best_loss,
            checkpoint_dir=checkpoint_dir,
        )

        # End-of-epoch validation
        val_loss = validate(model, effective_val_loader, device, config)

        # Get current LR (scheduler already stepped per-batch)
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch summary
        train_deg = torch.rad2deg(torch.tensor(train_loss)).item()
        val_deg = torch.rad2deg(torch.tensor(val_loss)).item()

        logger.info(
            f"Epoch {epoch:3d} done | "
            f"Train: {train_loss:.4f} ({train_deg:.1f} deg) | "
            f"Val: {val_loss:.4f} ({val_deg:.1f} deg) | "
            f"LR: {current_lr:.2e}"
        )

        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_loss_deg": train_deg,
                "val/loss": val_loss,
                "val/loss_deg": val_deg,
                "lr": current_lr,
            })

        # Save checkpoint if best
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_flat_model.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            logger.info(f"Saved best model (val_loss: {val_deg:.1f} deg)")

        # Save periodic checkpoint
        save_every = config['checkpoint'].get('save_every', 5)
        if epoch % save_every == 0:
            latest_path = checkpoint_dir / f"flat_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, latest_path)

    logger.info(f"Training complete. Best val loss: {torch.rad2deg(torch.tensor(best_loss)):.1f} deg")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

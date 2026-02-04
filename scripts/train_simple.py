#!/usr/bin/env python3
"""
Training script for hierarchical DOM model.

QUICK START
-----------
# Train with config
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_simple.py --config configs/train_config.yaml

# Override config params via CLI
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_simple.py --config configs/train_config.yaml --lr 1e-3 --epochs 20

# Run in background
screen -dmS train bash -c 'CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_simple.py --config configs/train_config.yaml 2>&1 | tee training_output.log'

EXPECTED RESULTS
----------------
- Random baseline: ~90° angular error (1.57 rad)
- After 2 epochs (800k events): ~70° (1.22 rad)
- After 10 epochs: ~50-60° (0.87-1.05 rad)
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
    make_collate_with_geometry,
    BatchAwareSampler,
)
from iceaggr.models import HierarchicalDOMModel, angular_distance_loss
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Config file path")

    # CLI overrides for common parameters
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--max-events", type=int, default=None, help="Override max training events")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--pool-method", type=str, default=None, help="Override pool method")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint to resume")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--name", type=str, default=None, help="Run name for wandb")

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
    if args.pool_method is not None:
        config['model']['pool_method'] = args.pool_method
    if args.checkpoint is not None:
        config['checkpoint']['resume'] = args.checkpoint
    if args.no_wandb:
        config['wandb']['enabled'] = False
    if args.name is not None:
        config['wandb']['name'] = args.name
    return config


def create_model(config: dict, device: str) -> nn.Module:
    """Create the hierarchical model from config."""
    model_config = {
        "embed_dim": config['model']['embed_dim'],
        "max_doms": config['model']['max_doms'],
        "dom_encoder_type": config['model']['dom_encoder_type'],
        "pool_method": config['model']['pool_method'],
        "event_num_heads": config['model']['event_num_heads'],
        "event_num_layers": config['model']['event_num_layers'],
        "event_hidden_dim": config['model']['event_hidden_dim'],
        "head_hidden_dim": config['model']['head_hidden_dim'],
        "dropout": config['model']['dropout'],
    }

    model = HierarchicalDOMModel(model_config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    return model


def create_dataloader(config: dict, geometry: GeometryLoader, split: str = 'train') -> DataLoader:
    """Create dataloader from config."""
    max_events = config['data']['max_events'] if split == 'train' else config['data'].get('val_events', 50000)

    dataset = IceCubeDataset(
        split=split,
        max_events=max_events,
        cache_size=1,
    )

    sampler = BatchAwareSampler(dataset.metadata)
    collate_fn = make_collate_with_geometry(geometry)

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return loader


def move_batch_to_device(batch: dict, device: str) -> dict:
    """Move all tensors in batch to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


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
) -> float:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    n_batches = len(loader)
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        # Forward with AMP
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config['training']['use_amp']):
            y_pred = model(batch)
            loss = angular_distance_loss(y_pred, batch['targets'])

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

    return total_loss / n_batches


def validate(model: nn.Module, loader: DataLoader, device: str, config: dict) -> float:
    """Validate on a subset of data."""
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 20:  # Only validate on 20 batches for speed
                break

            batch = move_batch_to_device(batch, device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config['training']['use_amp']):
                y_pred = model(batch)
                loss = angular_distance_loss(y_pred, batch['targets'])

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
            run_name = config['wandb']['name'] or f"train-{datetime.now().strftime('%m%d-%H%M')}"
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

    # Create dataloader
    loader = create_dataloader(config, geometry)
    logger.info(f"Training: {config['data']['max_events']:,} events, {len(loader):,} batches, bs={config['training']['batch_size']}")

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

    # Training loop
    logger.info(f"Starting training for {config['training']['epochs']} epochs...")
    logger.info(f"LR: {config['training']['lr']}, Pool: {config['model']['pool_method']}")

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # Train (scheduler steps per-batch inside train_epoch)
        train_loss = train_epoch(model, loader, optimizer, scaler, scheduler, device, epoch, config, wandb_run)

        # Validate
        val_loss = validate(model, loader, device, config)

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
        checkpoint_dir = Path(config['checkpoint']['dir'])
        checkpoint_dir.mkdir(exist_ok=True)

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
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
        save_every = config['checkpoint'].get('save_every', 10)
        if epoch % save_every == 0:
            latest_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
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

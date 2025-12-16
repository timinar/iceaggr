#!/usr/bin/env python3
"""
Simple multi-epoch training script for hierarchical DOM model.

This script uses a simple training loop (no Lightning) for easier debugging.
It runs multiple epochs and tracks progress clearly.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_simple.py
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_simple.py --epochs 50 --max-events 50000
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--max-events", type=int, default=50000, help="Max training events")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--pool-method", type=str, default="mean", help="DOM pooling method")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def create_model(config: dict, device: str) -> nn.Module:
    """Create the hierarchical model."""
    model = HierarchicalDOMModel(config)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    return model


def create_dataloader(max_events: int, batch_size: int, geometry: GeometryLoader) -> DataLoader:
    """Create training dataloader with BatchAwareSampler."""
    dataset = IceCubeDataset(
        split='train',
        max_events=max_events,
        cache_size=1,
    )

    sampler = BatchAwareSampler(dataset.metadata)  # Already shuffles internally
    collate_fn = make_collate_with_geometry(geometry)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
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
    device: str,
    epoch: int,
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
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_pred = model(batch)
            loss = angular_distance_loss(y_pred, batch['targets'])

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

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


def validate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Validate on a subset of data."""
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 20:  # Only validate on 20 batches for speed
                break

            batch = move_batch_to_device(batch, device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred = model(batch)
                loss = angular_distance_loss(y_pred, batch['targets'])

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Model configuration
    model_config = {
        "embed_dim": 64,
        "max_doms": 128,
        "pulse_hidden_dims": [64, 64],
        "dom_encoder_type": "pooling",
        "pool_method": args.pool_method,
        "event_num_heads": 4,
        "event_num_layers": 2,
        "event_hidden_dim": 256,
        "head_hidden_dim": 128,
        "dropout": 0.1,
    }

    # Load geometry
    geometry = GeometryLoader('/groups/pheno/inar/icecube_kaggle/sensor_geometry_normalized.csv')
    logger.info(f"Loaded geometry: {geometry}")

    # Create model
    model = create_model(model_config, device)

    # Create dataloader
    loader = create_dataloader(args.max_events, args.batch_size, geometry)
    logger.info(f"Training: {args.max_events:,} events, {len(loader):,} batches, bs={args.batch_size}")

    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 100,  # Min LR is 1% of initial
    )

    # AMP scaler
    scaler = torch.amp.GradScaler()

    # Resume from checkpoint if provided
    start_epoch = 1
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch-1}")

    # Training loop
    best_loss = float('inf')
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"LR: {args.lr}, Pool: {args.pool_method}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, loader, optimizer, scaler, device, epoch)

        # Validate (quick validation on 20 batches)
        val_loss = validate(model, loader, device)

        # Step scheduler
        scheduler.step()
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

        # Save checkpoint if best
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = Path("checkpoints/best_model.pt")
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model (val_loss: {val_deg:.1f} deg)")

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            latest_path = Path(f"checkpoints/epoch_{epoch:03d}.pt")
            latest_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, latest_path)

    logger.info(f"Training complete. Best val loss: {torch.rad2deg(torch.tensor(best_loss)):.1f} deg")


if __name__ == "__main__":
    main()

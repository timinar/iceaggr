#!/usr/bin/env python
"""
Train DeepSets hierarchical model from config.

Usage:
    uv run python scripts/train_deepsets.py experiments/deepsets_baseline/config.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
import wandb

from iceaggr.data.dataset import get_dataloader, IceCubeDataset
from iceaggr.models import HierarchicalIceCubeModel
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {config_path}")
    return config


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_cfg = config["training"]

    # Create full dataset
    full_dataset = IceCubeDataset(
        split="train",
        max_events=train_cfg.get("max_events"),
    )

    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * train_cfg["val_split"])
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Dataset split: {len(train_dataset):,} train, {len(val_dataset):,} val")

    # Create dataloaders with DeepSets collate function
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,  # Data is already shuffled across batches
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=lambda batch: __import__('iceaggr.data.dataset', fromlist=['collate_deepsets']).collate_deepsets(batch),
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=lambda batch: __import__('iceaggr.data.dataset', fromlist=['collate_deepsets']).collate_deepsets(batch),
        pin_memory=True,
    )

    return train_loader, val_loader


def create_model(config: dict) -> HierarchicalIceCubeModel:
    """Create model from config."""
    model_cfg = config["model"]

    model = HierarchicalIceCubeModel(
        d_pulse=model_cfg["d_pulse"],
        d_dom_embedding=model_cfg["d_dom_embedding"],
        dom_latent_dim=model_cfg["dom_latent_dim"],
        dom_hidden_dim=model_cfg["dom_hidden_dim"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg.get("d_ff"),
        dropout=model_cfg["dropout"],
        use_geometry=model_cfg["use_geometry"],
    )

    return model


def train_epoch(
    model: HierarchicalIceCubeModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    train_cfg = config["training"]
    log_every = train_cfg["log_every_n_steps"]

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        predictions = model(batch)
        loss = model.compute_loss(predictions, batch['targets'])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if train_cfg.get("grad_clip_norm"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["grad_clip_norm"]
            )

        optimizer.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}"
            )

            if train_cfg.get("use_wandb"):
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/batch": batch_idx + epoch * len(train_loader),
                })

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: HierarchicalIceCubeModel,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: dict
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        predictions = model(batch)
        loss = model.compute_loss(predictions, batch['targets'])

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    logger.info(f"Validation | Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    if config["training"].get("use_wandb"):
        wandb.log({"val/loss": avg_loss, "epoch": epoch})

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train DeepSets model")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize W&B
    train_cfg = config["training"]
    if train_cfg.get("use_wandb"):
        wandb.init(
            project=train_cfg["wandb_project"],
            name=train_cfg["wandb_run_name"],
            tags=train_cfg.get("wandb_tags", []),
            config=config,
        )

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # Create checkpoint dir
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{train_cfg['num_epochs']}")
        logger.info("-" * 60)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, config)
        logger.info(f"Epoch {epoch} Training Loss: {train_loss:.4f}")

        # Validate
        if epoch % train_cfg.get("validate_every_n_epochs", 1) == 0:
            val_loss = validate(model, val_loader, device, epoch, config)

            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")

        # Save periodic checkpoint
        if epoch % train_cfg.get("save_every_n_epochs", 1) == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 60)

    if train_cfg.get("use_wandb"):
        wandb.finish()


if __name__ == "__main__":
    main()

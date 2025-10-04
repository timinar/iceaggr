#!/usr/bin/env python
"""
Train E2E model from experiment config file.

Usage:
    uv run python scripts/train_from_config.py experiments/baseline_1m/config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.models import HierarchicalTransformer
from iceaggr.training import AngularDistanceLoss, E2ETrainer, TrainingConfig
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
    """Create train and validation dataloaders from config."""
    train_cfg = config["training"]

    # Create dataset
    dataset = IceCubeDataset(
        split="train",
        max_events=train_cfg.get("max_events"),
    )

    # Split into train/val
    total_size = len(dataset)
    val_size = int(total_size * train_cfg["val_split"])
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    logger.info(f"Dataset split: {train_size} train, {val_size} val")

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=lambda batch: collate_dom_packing(
            batch, max_seq_len=config["model"]["t1_max_seq_len"]
        ),
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=lambda batch: collate_dom_packing(
            batch, max_seq_len=config["model"]["t1_max_seq_len"]
        ),
        pin_memory=True,
    )

    return train_loader, val_loader


def create_model(config: dict) -> HierarchicalTransformer:
    """Create model from config."""
    model_cfg = config["model"]

    model = HierarchicalTransformer(
        d_model=model_cfg["d_model"],
        t1_n_heads=model_cfg["t1_n_heads"],
        t1_n_layers=model_cfg["t1_n_layers"],
        t1_max_seq_len=model_cfg["t1_max_seq_len"],
        t1_max_batch_size=model_cfg["t1_max_batch_size"],
        t2_n_heads=model_cfg["t2_n_heads"],
        t2_n_layers=model_cfg["t2_n_layers"],
        t2_max_doms=model_cfg["t2_max_doms"],
        dropout=model_cfg["dropout"],
        sensor_geometry_path=None,  # Auto-detect
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Train from experiment config")
    parser.add_argument("config", type=str, help="Path to experiment config YAML")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create log directory
    log_dir = Path(config["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs will be saved to: {log_dir}")

    # Create checkpoint directory
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss and optimizer
    loss_fn = AngularDistanceLoss(use_unit_vectors=False)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Training config
    training_config = TrainingConfig(
        num_epochs=train_cfg["num_epochs"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        grad_clip_norm=train_cfg["grad_clip_norm"],
        accumulation_steps=train_cfg["accumulation_steps"],
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=train_cfg["save_every_n_epochs"],
        log_every_n_steps=train_cfg["log_every_n_steps"],
        validate_every_n_epochs=train_cfg["validate_every_n_epochs"],
        use_wandb=train_cfg.get("use_wandb", False),
        wandb_project=train_cfg.get("wandb_project"),
        wandb_run_name=train_cfg.get("wandb_run_name"),
        wandb_tags=train_cfg.get("wandb_tags", []),
    )

    # Create trainer
    trainer = E2ETrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=training_config,
    )

    # Resume from checkpoint if requested
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Log config to W&B
    if training_config.use_wandb and trainer.wandb_run:
        trainer.wandb_run.config.update({
            "model": config["model"],
            "training": config["training"],
        })

    # Train
    logger.info("Starting training...")
    history = trainer.fit(train_loader, val_loader)

    logger.info("Training complete!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()

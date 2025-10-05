#!/usr/bin/env python
"""
Train E2E model from experiment config file.

Usage:
    uv run python scripts/train_from_config.py experiments/baseline_1m/config.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.models import HierarchicalTransformer
from iceaggr.models.t2_first_pulse import T2FirstPulseModel
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
    model_cfg = config["model"]

    # Create full dataset
    full_dataset = IceCubeDataset(
        split="train",
        max_events=train_cfg.get("max_events"),
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * train_cfg["val_split"])
    train_size = total_size - val_size

    logger.info(f"Total events: {total_size}, will use first {train_size} for train, last {val_size} for val")

    # Create separate datasets (no Subset!) using slicing indices
    train_dataset = IceCubeDataset(
        split="train",
        max_events=train_size,  # Only load train_size events
    )

    # For validation, we'd need a different approach since IceCubeDataset doesn't support offset
    # For now, just use a small portion of the training data for validation
    # TODO: Implement proper train/val split in IceCubeDataset
    val_dataset = IceCubeDataset(
        split="train",
        max_events=min(10000, val_size),  # Use small val set for now
    )

    logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")

    # Determine max_seq_len for collate function
    # For t2_first_pulse model, we still need DOM packing (same collate function)
    max_seq_len = model_cfg.get("t1_max_seq_len", 512)

    # Create dataloaders
    # NOTE: shuffle=False because RandomSampler is extremely slow with 900K events (11 sec/batch!)
    # Data is already somewhat shuffled across batches since events span multiple parquet files
    # TODO: Implement efficient shuffling (e.g., shuffle batches or use custom sampler)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,  # Disabled due to performance (see shuffle_debug.log)
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=lambda batch: collate_dom_packing(
            batch, max_seq_len=max_seq_len
        ),
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=lambda batch: collate_dom_packing(
            batch, max_seq_len=max_seq_len
        ),
        pin_memory=True,
    )

    return train_loader, val_loader


def create_model(config: dict):
    """Create model from config."""
    model_cfg = config["model"]
    model_type = model_cfg.get("model_type", "hierarchical")

    if model_type == "t2_first_pulse":
        # T2-only model using first pulse features
        model = T2FirstPulseModel(
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            max_doms=model_cfg["max_doms"],
            dropout=model_cfg["dropout"],
            geometry_path=None,  # Auto-detect
        )
    else:
        # Standard hierarchical transformer (T1 + T2)
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

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load config
    config = load_config(args.config)

    # Create log directory with timestamped filename
    log_dir = Path(config["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging with timestamp
    import sys
    log_file = log_dir / f"training_{timestamp}.log"
    sys.stdout = open(log_file, 'w', buffering=1)  # Line buffered
    sys.stderr = sys.stdout

    logger.info(f"Logs will be saved to: {log_file}")

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
    loss_fn = AngularDistanceLoss(use_unit_vectors=True)

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
        wandb_run_name=f"{train_cfg.get('wandb_run_name')}-{timestamp}" if train_cfg.get('wandb_run_name') else timestamp,
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

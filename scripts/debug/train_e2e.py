#!/usr/bin/env python
"""
Training script for end-to-end hierarchical transformer (T1 + T2).

Usage:
    uv run python scripts/train_e2e.py
    uv run python scripts/train_e2e.py --use-wandb --wandb-run-name "e2e-baseline-yourname"
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.models import HierarchicalTransformer
from iceaggr.training import AngularDistanceLoss, E2ETrainer, TrainingConfig
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def load_data_config() -> dict:
    """Load data paths from config file."""
    config_path = Path("src/iceaggr/data/data_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Data config not found at {config_path}. "
            "Copy from data_config.template.yaml and modify."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def create_dataloaders(
    config: dict,
    batch_size: int = 32,
    max_seq_len: int = 512,
    val_split: float = 0.1,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        config: Data configuration
        batch_size: Number of events per batch
        max_seq_len: Maximum pulses per DOM sequence
        val_split: Validation split ratio
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader
    """
    # Create dataset
    dataset = IceCubeDataset(split="train")

    # Split into train/val
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    logger.info(f"Train size: {train_size} batches, Val size: {val_size} batches")

    # Create dataloaders with DOM packing collation
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=max_seq_len),
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=max_seq_len),
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train E2E hierarchical transformer")

    # Model hyperparameters
    parser.add_argument(
        "--d-model", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--t1-heads", type=int, default=8, help="T1 attention heads"
    )
    parser.add_argument(
        "--t1-layers", type=int, default=4, help="T1 transformer layers"
    )
    parser.add_argument(
        "--t2-heads", type=int, default=8, help="T2 attention heads"
    )
    parser.add_argument(
        "--t2-layers", type=int, default=4, help="T2 transformer layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate"
    )

    # Data hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (events)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=512, help="Max pulses per DOM sequence"
    )
    parser.add_argument(
        "--t1-max-batch", type=int, default=64, help="T1 max batch size (prevents OOM)"
    )
    parser.add_argument(
        "--t2-max-doms", type=int, default=2048, help="T2 max DOMs per event"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clip norm"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Data
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Dataloader workers"
    )

    # Geometry
    parser.add_argument(
        "--geometry-path",
        type=str,
        default=None,
        help="Path to sensor geometry CSV (default: auto-detect from data_config.yaml)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/e2e",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--save-every", type=int, default=1, help="Save checkpoint every N epochs"
    )

    # Logging
    parser.add_argument(
        "--log-every", type=int, default=10, help="Log every N steps"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="iceaggr", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="W&B run name"
    )

    args = parser.parse_args()

    # Load data config
    logger.info("Loading data configuration...")
    data_config = load_data_config()

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config=data_config,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    # Create E2E model
    logger.info("Creating hierarchical transformer model...")
    model = HierarchicalTransformer(
        d_model=args.d_model,
        t1_n_heads=args.t1_heads,
        t1_n_layers=args.t1_layers,
        t1_max_seq_len=args.max_seq_len,
        t1_max_batch_size=args.t1_max_batch,
        t2_n_heads=args.t2_heads,
        t2_n_layers=args.t2_layers,
        t2_max_doms=args.t2_max_doms,
        dropout=args.dropout,
        sensor_geometry_path=args.geometry_path,
    )

    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss function
    loss_fn = AngularDistanceLoss(use_unit_vectors=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training config
    config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epochs=args.save_every,
        log_every_n_steps=args.log_every,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
        or f"e2e-d{args.d_model}-t1L{args.t1_layers}-t2L{args.t2_layers}",
        wandb_tags=["e2e", "hierarchical", "t1+t2"],
    )

    # Create trainer
    trainer = E2ETrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
    )

    # Train
    logger.info("Starting training...")
    history = trainer.fit(train_loader, val_loader)

    logger.info("Training complete!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()

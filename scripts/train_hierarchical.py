#!/usr/bin/env python3
"""
Training script for hierarchical DOM aggregation model.

Usage:
    uv run python scripts/train_hierarchical.py --config configs/experiment/baseline_v1.yaml
    uv run python scripts/train_hierarchical.py --config configs/experiment/baseline_v1.yaml --max-events 10000
"""

import argparse
from pathlib import Path
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
import torch

from iceaggr.data import (
    IceCubeDataset,
    GeometryLoader,
    make_collate_with_geometry,
    BatchAwareSampler,
)
from iceaggr.training import HierarchicalModelModule
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train hierarchical DOM model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/baseline_v1.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Override max training events (for debugging)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast development run (1 batch train/val)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, max_events: int = None):
    """Create train and validation dataloaders."""
    data_config = config.get('data', {})
    training_config = config.get('training', {})

    # Load geometry
    geometry_path = data_config.get(
        'geometry_path',
        '/groups/pheno/inar/icecube_kaggle/sensor_geometry_normalized.csv'
    )
    geometry = GeometryLoader(geometry_path)
    logger.info(f"Loaded geometry: {geometry}")

    # Create collate function with geometry
    collate_fn = make_collate_with_geometry(geometry)

    # Training dataset
    max_train = max_events or data_config.get('max_train_events')
    train_dataset = IceCubeDataset(
        split='train',
        max_events=max_train,
        cache_size=1,
    )

    # Validation dataset (use last portion of train data for now)
    max_val = data_config.get('max_val_events', 50000)
    val_dataset = IceCubeDataset(
        split='train',
        max_events=max_val,
        cache_size=1,
    )

    batch_size = training_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)

    # Train sampler for efficient file access
    train_sampler = BatchAwareSampler(train_dataset.metadata)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    logger.info(f"Train: {len(train_dataset):,} events, {len(train_loader):,} batches")
    logger.info(f"Val: {len(val_dataset):,} events, {len(val_loader):,} batches")

    return train_loader, val_loader, geometry


def create_callbacks(config: dict) -> list:
    """Create training callbacks."""
    training_config = config.get('training', {})

    callbacks = [
        # Checkpoint best models
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="hierarchical-{epoch:02d}-{val/loss:.4f}",
            save_top_k=training_config.get('save_top_k', 3),
            monitor=training_config.get('monitor', 'val/loss'),
            mode=training_config.get('mode', 'min'),
        ),
        # Learning rate logging
        LearningRateMonitor(logging_interval='step'),
        # Early stopping
        EarlyStopping(
            monitor='val/loss',
            patience=10,
            mode='min',
        ),
    ]

    return callbacks


def create_logger(config: dict, use_wandb: bool = True):
    """Create experiment logger."""
    if not use_wandb:
        return None

    logging_config = config.get('logging', {})

    return WandbLogger(
        project=logging_config.get('project', 'iceaggr'),
        name=logging_config.get('name', 'hierarchical-dom'),
        tags=logging_config.get('tags', []),
        log_model=True,
    )


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Merge model and training configs
    model_config = {**config.get('model', {}), **config.get('training', {})}

    # Create dataloaders
    # Note: geometry stays on CPU for DataLoader workers, model handles GPU transfer
    train_loader, val_loader, geometry = create_dataloaders(
        config,
        max_events=args.max_events,
    )

    # Create model
    model = HierarchicalModelModule(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create callbacks and logger
    callbacks = create_callbacks(config)
    exp_logger = create_logger(config, use_wandb=not args.no_wandb)

    # Training config
    training_config = config.get('training', {})
    hardware_config = config.get('hardware', {})

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get('max_epochs', 100),
        accelerator=hardware_config.get('accelerator', 'gpu'),
        devices=hardware_config.get('devices', 1),
        precision=training_config.get('precision', '16-mixed'),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 1),
        val_check_interval=training_config.get('val_check_interval', 1.0),
        log_every_n_steps=config.get('logging', {}).get('log_every_n_steps', 50),
        callbacks=callbacks,
        logger=exp_logger,
        fast_dev_run=args.fast_dev_run,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Log final results
    if trainer.callback_metrics:
        final_loss = trainer.callback_metrics.get('val/loss', 0)
        final_error = trainer.callback_metrics.get('val/angular_error_deg', 0)
        logger.info(f"Final val loss: {final_loss:.4f}")
        logger.info(f"Final angular error: {final_error:.2f} degrees")


if __name__ == "__main__":
    main()

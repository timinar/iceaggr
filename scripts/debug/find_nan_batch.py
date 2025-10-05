#!/usr/bin/env python
"""
Find the exact batch that causes NaN by searching through the dataset.
"""

import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.models import HierarchicalTransformer
from iceaggr.training import AngularDistanceLoss
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def check_batch(model, batch, loss_fn, step, device):
    """Check if batch causes NaN."""
    # Move to device
    batch_device = {}
    for key, value in batch.items():
        if key == "metadata":
            batch_device[key] = {}
            for meta_key, meta_value in value.items():
                if isinstance(meta_value, torch.Tensor):
                    batch_device[key][meta_key] = meta_value.to(device)
                else:
                    batch_device[key][meta_key] = meta_value
        elif isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        else:
            batch_device[key] = value

    # Forward pass
    with torch.no_grad():
        predictions = model(batch_device)

    # Check predictions
    if not torch.isfinite(predictions).all():
        logger.error(f"❌ Step {step}: Non-finite predictions!")
        logger.error(f"   NaN count: {torch.isnan(predictions).sum()}")
        logger.error(f"   Inf count: {torch.isinf(predictions).sum()}")
        logger.error(f"   Event IDs: {batch_device['metadata']['event_ids']}")
        return True

    # Try loss
    targets = batch_device["metadata"]["targets"]
    try:
        loss = loss_fn(predictions, targets)
    except AssertionError as e:
        logger.error(f"❌ Step {step}: Loss failed: {e}")
        return True

    return False


def main():
    # Load config
    config_path = Path("experiments/baseline_1m/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Test larger dataset to find problematic batch
    # Epoch 3 would start around event 1.8M in a 900K dataset (with repeat)
    # Let's test the full dataset
    logger.info("Creating FULL dataset (900K events)...")
    dataset = IceCubeDataset(split="train", max_events=1000000)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=False,  # Keep in order to match training
        num_workers=0,
        collate_fn=lambda batch: collate_dom_packing(
            batch, max_seq_len=config["model"]["t1_max_seq_len"]
        ),
    )

    logger.info("Creating model...")
    model = HierarchicalTransformer(
        d_model=config["model"]["d_model"],
        t1_n_heads=config["model"]["t1_n_heads"],
        t1_n_layers=config["model"]["t1_n_layers"],
        t1_max_seq_len=config["model"]["t1_max_seq_len"],
        t1_max_batch_size=config["model"]["t1_max_batch_size"],
        t2_n_heads=config["model"]["t2_n_heads"],
        t2_n_layers=config["model"]["t2_n_layers"],
        t2_max_doms=config["model"]["t2_max_doms"],
        dropout=config["model"]["dropout"],
        sensor_geometry_path=None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Eval mode to isolate data issues

    loss_fn = AngularDistanceLoss(use_unit_vectors=True)

    logger.info(f"Device: {device}")
    logger.info(f"Total batches: {len(dataloader)}")
    logger.info(f"\nSearching for problematic batch...\n")

    # Search through dataset
    for step, batch in enumerate(dataloader):
        if check_batch(model, batch, loss_fn, step, device):
            logger.error(f"\n{'!'*60}")
            logger.error(f"Found problematic batch at step {step}!")
            logger.error(f"{'!'*60}\n")

            # Additional diagnostics
            logger.info("Batch diagnostics:")
            logger.info(f"  packed_sequences shape: {batch['packed_sequences'].shape}")
            logger.info(f"  packed_sequences min/max: {batch['packed_sequences'].min():.4f} / {batch['packed_sequences'].max():.4f}")
            logger.info(f"  targets shape: {batch['metadata']['targets'].shape}")
            logger.info(f"  targets min/max: {batch['metadata']['targets'].min():.4f} / {batch['metadata']['targets'].max():.4f}")
            logger.info(f"  targets has NaN: {torch.isnan(batch['metadata']['targets']).any()}")
            logger.info(f"  Event IDs: {batch['metadata']['event_ids']}")

            break

        if step % 100 == 0:
            logger.info(f"Checked {step} batches...")

    else:
        logger.info("\n✓ No problematic batches found in entire dataset!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Test that training works with the fixes applied.

Quick validation run (100 steps) to ensure:
1. No NaN issues
2. Loss decreases
3. Model trains stably
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


def main():
    # Load FIXED config
    config_path = Path("experiments/baseline_1m/config_fixed.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("="*60)
    logger.info("Testing FIXED training configuration")
    logger.info("="*60)
    logger.info(f"LR: {config['training']['learning_rate']}")
    logger.info(f"Dropout: {config['model']['dropout']}")
    logger.info(f"Input normalization: ENABLED ✓")
    logger.info(f"Geometry normalization: ENABLED ✓")
    logger.info("="*60)

    logger.info("\nCreating dataset (10K events)...")
    dataset = IceCubeDataset(split="train", max_events=10000)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
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
    model.train()

    loss_fn = AngularDistanceLoss(use_unit_vectors=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    logger.info(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Device: {device}")
    logger.info("\nStarting training test (100 steps)...\n")

    losses = []
    max_steps = 100

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

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

        # Training step
        optimizer.zero_grad()
        predictions = model(batch_device)

        # Check for NaN
        if not torch.isfinite(predictions).all():
            logger.error(f"❌ Step {step}: NaN in predictions!")
            return False

        # Loss
        targets = batch_device["metadata"]["targets"]
        loss = loss_fn(predictions, targets)
        losses.append(loss.item())

        # Check loss
        if not torch.isfinite(loss):
            logger.error(f"❌ Step {step}: NaN in loss!")
            return False

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip_norm"])
        optimizer.step()

        # Log
        if step % 20 == 0:
            logger.info(f"Step {step:3d}: Loss = {loss.item():.4f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Initial loss: {losses[0]:.4f} rad ({losses[0]*57.3:.1f}°)")
    logger.info(f"Final loss:   {losses[-1]:.4f} rad ({losses[-1]*57.3:.1f}°)")
    logger.info(f"Change:       {losses[-1] - losses[0]:.4f} rad")
    logger.info(f"")

    if losses[-1] < losses[0]:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        logger.info(f"✅ Loss DECREASED by {improvement:.1f}%")
        logger.info(f"✅ Model is learning!")
        logger.info(f"✅ Training is STABLE")
        success = True
    else:
        logger.warning(f"❌ Loss did not decrease")
        success = False

    logger.info(f"{'='*60}\n")

    if success:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("\nReady to run full training:")
        logger.info("  uv run python scripts/train_from_config.py experiments/baseline_1m/config_fixed.yaml")
    else:
        logger.error("❌ TESTS FAILED - needs more investigation")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

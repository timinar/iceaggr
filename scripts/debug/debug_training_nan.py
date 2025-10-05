#!/usr/bin/env python
"""
Debug NaN issue during training - simulate full training to find when NaN appears.
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


def check_for_nan(tensor, name):
    """Check if tensor has NaN/Inf."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.error(f"❌ {name} has NaN/Inf!")
        logger.error(f"   min={tensor.min():.4f}, max={tensor.max():.4f}")
        return True
    return False


def main():
    # Load config
    config_path = Path("experiments/baseline_1m/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Creating dataset...")
    dataset = IceCubeDataset(split="train", max_events=10000)  # 10K events

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=256,  # Same as config
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
    model.train()  # Training mode

    loss_fn = AngularDistanceLoss(use_unit_vectors=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    logger.info(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Device: {device}")
    logger.info(f"LR: {config['training']['learning_rate']}")
    logger.info(f"Weight decay: {config['training']['weight_decay']}")
    logger.info(f"\nStarting training simulation...\n")

    # Training loop
    for step, batch in enumerate(dataloader):
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
        optimizer.zero_grad()
        predictions = model(batch_device)

        # Check predictions
        if check_for_nan(predictions, f"Step {step} predictions"):
            logger.error("NaN appeared in forward pass!")
            break

        # Loss
        targets = batch_device["metadata"]["targets"]
        try:
            loss = loss_fn(predictions, targets)
        except AssertionError as e:
            logger.error(f"❌ Step {step}: Loss computation failed: {e}")
            logger.error(f"   Predictions stats: min={predictions.min():.4f}, max={predictions.max():.4f}")
            logger.error(f"   Predictions finite: {torch.isfinite(predictions).all()}")
            break

        # Check loss
        if check_for_nan(loss, f"Step {step} loss"):
            logger.error("NaN appeared in loss!")
            break

        # Backward
        loss.backward()

        # Check gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if check_for_nan(param.grad, f"Step {step} grad {name}"):
                    has_nan_grad = True
                    break

        if has_nan_grad:
            logger.error("NaN appeared in gradients!")
            break

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip_norm"])

        # Check after clipping
        for name, param in model.named_parameters():
            if param.grad is not None:
                if check_for_nan(param.grad, f"Step {step} grad (after clip) {name}"):
                    has_nan_grad = True
                    break

        if has_nan_grad:
            logger.error("NaN appeared after gradient clipping!")
            break

        # Optimizer step
        optimizer.step()

        # Check parameters after update
        has_nan_param = False
        for name, param in model.named_parameters():
            if check_for_nan(param, f"Step {step} param (after update) {name}"):
                has_nan_param = True
                break

        if has_nan_param:
            logger.error("NaN appeared in parameters after optimizer step!")
            break

        # Log progress
        if step % 10 == 0:
            logger.info(f"Step {step}: Loss = {loss.item():.4f}")

    logger.info(f"\nCompleted {step} steps without NaN issues")


if __name__ == "__main__":
    main()

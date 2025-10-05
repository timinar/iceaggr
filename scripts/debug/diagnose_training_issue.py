#!/usr/bin/env python
"""
Comprehensive diagnosis of training issues:
1. Check for gradient explosion/vanishing
2. Check learning rate sensitivity
3. Check model initialization
4. Monitor activation magnitudes
"""

import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
import numpy as np

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.models import HierarchicalTransformer
from iceaggr.training import AngularDistanceLoss
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def analyze_gradients(model):
    """Analyze gradient statistics."""
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 10.0 or grad_norm < 1e-6:
                logger.warning(f"  {name}: grad_norm={grad_norm:.6f}")

    return grad_norms


def main():
    # Load config
    config_path = Path("experiments/baseline_1m/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Creating dataset...")
    dataset = IceCubeDataset(split="train", max_events=5000)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=256,
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

    # Test different learning rates
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

    for lr in learning_rates:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing LR = {lr}")
        logger.info(f"{'='*60}")

        # Fresh model for each LR
        model_test = HierarchicalTransformer(
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
        model_test.to(device)
        model_test.train()

        optimizer = torch.optim.AdamW(
            model_test.parameters(),
            lr=lr,
            weight_decay=config["training"]["weight_decay"],
        )

        losses = []
        grad_norms_all = []

        for step, batch in enumerate(dataloader):
            if step >= 20:  # Test 20 steps
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

            # Forward
            optimizer.zero_grad()
            predictions = model_test(batch_device)

            # Check for NaN
            if not torch.isfinite(predictions).all():
                logger.error(f"  Step {step}: NaN in predictions! Stopping this LR test.")
                break

            # Loss
            targets = batch_device["metadata"]["targets"]
            loss = loss_fn(predictions, targets)
            losses.append(loss.item())

            # Backward
            loss.backward()

            # Analyze gradients
            grad_norms = analyze_gradients(model_test)
            max_grad = max(grad_norms) if grad_norms else 0
            grad_norms_all.append(max_grad)

            # Clip
            torch.nn.utils.clip_grad_norm_(model_test.parameters(), 1.0)

            # Update
            optimizer.step()

        # Summary
        if losses:
            logger.info(f"\nLR {lr} Summary:")
            logger.info(f"  Initial loss: {losses[0]:.4f}")
            logger.info(f"  Final loss: {losses[-1]:.4f}")
            logger.info(f"  Loss change: {losses[-1] - losses[0]:.4f}")
            logger.info(f"  Max grad norm: {max(grad_norms_all):.4f}")
            logger.info(f"  Loss stable: {np.std(losses) < 0.1}")

            if losses[-1] < losses[0]:
                logger.info(f"  ✓ Loss decreasing!")
            else:
                logger.warning(f"  ❌ Loss not decreasing")

    logger.info(f"\n{'='*60}")
    logger.info("Diagnosis complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

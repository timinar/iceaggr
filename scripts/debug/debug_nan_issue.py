#!/usr/bin/env python
"""
Debug NaN issue in model predictions.

Runs a few batches and checks where NaNs appear:
- Model outputs
- Loss computation
- Gradients
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


def check_tensor(name: str, tensor: torch.Tensor) -> bool:
    """Check if tensor contains NaN or Inf."""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        logger.error(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
        logger.error(f"   Stats: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        return True
    else:
        logger.info(f"✓ {name}: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        return False


def debug_batch(model, batch, loss_fn, batch_idx):
    """Debug a single batch."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Debugging Batch {batch_idx}")
    logger.info(f"{'='*60}")

    # Check inputs
    logger.info("\n1. Checking Input Data:")
    logger.info(f"   Batch keys: {batch.keys()}")
    logger.info(f"   Metadata keys: {batch['metadata'].keys()}")

    check_tensor("packed_sequences", batch["packed_sequences"])
    check_tensor("dom_mask", batch["dom_mask"])
    check_tensor("targets", batch["metadata"]["targets"])

    # Forward pass with intermediate checks
    logger.info("\n2. Forward Pass:")
    model.eval()  # Disable dropout for debugging
    has_issue = False

    with torch.no_grad():
        # Full forward pass
        logger.info("   Running full model (T1 → T2)...")
        predictions = model(batch)
        has_issue = check_tensor("   Model output (predictions)", predictions)

        # Check if predictions are valid unit vectors
        norms = torch.sqrt(torch.sum(predictions**2, dim=1))
        logger.info(f"   Prediction norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    # Loss computation
    logger.info("\n3. Loss Computation:")
    targets = batch["metadata"]["targets"]

    try:
        loss = loss_fn(predictions, targets)
        check_tensor("   Loss", loss)
        logger.info(f"   Loss value: {loss.item():.4f}")
    except AssertionError as e:
        logger.error(f"❌ Loss computation failed: {e}")
        return True

    # Gradient flow check (train mode)
    logger.info("\n4. Gradient Flow Check:")
    model.train()
    model.zero_grad()

    predictions_train = model(batch)
    loss_train = loss_fn(predictions_train, targets)

    try:
        loss_train.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_nan = torch.isnan(param.grad).any()
                grad_inf = torch.isinf(param.grad).any()
                if grad_nan or grad_inf:
                    logger.error(f"❌ Gradient {name}: NaN={grad_nan}, Inf={grad_inf}")
                    has_issue = True

        logger.info("   ✓ Gradients OK" if not has_issue else "   ❌ Gradient issues detected")

    except Exception as e:
        logger.error(f"❌ Backward pass failed: {e}")
        return True

    return has_issue


def main():
    # Load config
    config_path = Path("experiments/baseline_1m/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Creating dataset...")
    dataset = IceCubeDataset(split="train", max_events=1000)  # Just 1000 events

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,  # Smaller batch for debugging
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

    loss_fn = AngularDistanceLoss(use_unit_vectors=True)

    logger.info(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Device: {device}")
    logger.info(f"\nTesting {len(dataloader)} batches...\n")

    # Test first few batches
    num_batches_to_test = 5
    issues_found = False

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches_to_test:
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

        has_issue = debug_batch(model, batch_device, loss_fn, batch_idx)
        if has_issue:
            issues_found = True
            logger.error(f"\n{'!'*60}")
            logger.error(f"Issues found in batch {batch_idx}! Stopping...")
            logger.error(f"{'!'*60}\n")
            break

    if not issues_found:
        logger.info(f"\n{'='*60}")
        logger.info("✓ All tested batches OK - no NaN/Inf issues detected")
        logger.info(f"{'='*60}\n")

    # Additional diagnostics
    logger.info("\nModel Architecture Summary:")
    logger.info(f"T1 (DOM): {config['model']['t1_n_layers']} layers, {config['model']['t1_n_heads']} heads")
    logger.info(f"T2 (Event): {config['model']['t2_n_layers']} layers, {config['model']['t2_n_heads']} heads")
    logger.info(f"d_model: {config['model']['d_model']}")
    logger.info(f"dropout: {config['model']['dropout']}")


if __name__ == "__main__":
    main()

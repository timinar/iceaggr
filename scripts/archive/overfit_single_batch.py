#!/usr/bin/env python
"""
Overfit on a single batch to verify the training pipeline works.

This is a critical sanity check - if the model can't overfit a single batch,
there's a bug in the pipeline.

Usage:
    uv run python scripts/overfit_single_batch.py
    uv run python scripts/overfit_single_batch.py --iterations 500 --lr 1e-2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.models import HierarchicalTransformer
from iceaggr.training import AngularDistanceLoss
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


def main():
    parser = argparse.ArgumentParser(
        description="Overfit on single batch to verify pipeline"
    )

    # Model config
    parser.add_argument("--d-model", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--t1-layers", type=int, default=4, help="T1 layers")
    parser.add_argument("--t2-layers", type=int, default=4, help="T2 layers")

    # Training config
    parser.add_argument("--iterations", type=int, default=200, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (events)")

    # Data config
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")

    # Geometry
    parser.add_argument(
        "--geometry-path",
        type=str,
        default=None,
        help="Sensor geometry path (default: auto-detect from data_config.yaml)",
    )

    # Output
    parser.add_argument(
        "--save-plot", type=str, default="overfit_loss.png", help="Save loss plot"
    )

    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")

    dataset = IceCubeDataset(split="train")

    # Create dataloader (we'll just use the first batch)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=args.max_seq_len),
    )

    # Get single batch
    batch = next(iter(loader))
    logger.info(f"Loaded batch with {len(batch['metadata']['event_ids'])} events")
    logger.info(
        f"Total DOMs in batch: {batch['metadata']['total_doms']}, "
        f"Packed sequences: {batch['packed_sequences'].shape[0]}"
    )

    # Create model
    logger.info("Creating model...")
    model = HierarchicalTransformer(
        d_model=args.d_model,
        t1_n_heads=8,
        t1_n_layers=args.t1_layers,
        t1_max_seq_len=args.max_seq_len,
        t1_max_batch_size=64,
        t2_n_heads=8,
        t2_n_layers=args.t2_layers,
        t2_max_doms=2048,
        dropout=0.1,
        sensor_geometry_path=args.geometry_path,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Move batch to device
    def batch_to_device(batch, device):
        device_batch = {}
        for key, value in batch.items():
            if key == "metadata":
                device_batch[key] = {}
                for meta_key, meta_value in value.items():
                    if isinstance(meta_value, torch.Tensor):
                        device_batch[key][meta_key] = meta_value.to(device)
                    else:
                        device_batch[key][meta_key] = meta_value
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value
        return device_batch

    batch = batch_to_device(batch, device)
    targets = batch["metadata"]["targets"]

    # Loss and optimizer
    loss_fn = AngularDistanceLoss(use_unit_vectors=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    logger.info(f"Starting overfitting for {args.iterations} iterations...")
    losses = []

    for iteration in range(args.iterations):
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch)

        # Compute loss
        loss = loss_fn(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Log progress
        if (iteration + 1) % 20 == 0 or iteration == 0:
            logger.info(f"Iteration {iteration + 1}/{args.iterations} | Loss: {loss.item():.6f}")

    # Final evaluation
    logger.info("\n=== Final Results ===")
    logger.info(f"Initial loss: {losses[0]:.6f}")
    logger.info(f"Final loss: {losses[-1]:.6f}")
    logger.info(f"Loss reduction: {(1 - losses[-1] / losses[0]) * 100:.1f}%")

    # Check if overfitting succeeded
    if losses[-1] < losses[0] * 0.1:
        logger.info("✅ SUCCESS: Model successfully overfitted the batch!")
    elif losses[-1] < losses[0] * 0.5:
        logger.warning("⚠️  WARNING: Loss decreased but not enough. May need more iterations or higher LR.")
    else:
        logger.error("❌ FAILED: Model did not overfit. Check for bugs in the pipeline!")

    # Compute angular errors
    with torch.no_grad():
        predictions = model(batch)
        errors_rad = torch.abs(predictions - targets)
        errors_deg = errors_rad * 180 / 3.14159

        logger.info("\n=== Angular Errors (degrees) ===")
        logger.info(f"Azimuth error: {errors_deg[:, 0].mean():.2f} ± {errors_deg[:, 0].std():.2f}")
        logger.info(f"Zenith error: {errors_deg[:, 1].mean():.2f} ± {errors_deg[:, 1].std():.2f}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss (radians)", fontsize=12)
    plt.title("Single Batch Overfitting", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = Path(args.save_plot)
    plt.savefig(plot_path, dpi=150)
    logger.info(f"\nLoss plot saved to: {plot_path}")

    # Also log to stdout for easy viewing
    plt.show()


if __name__ == "__main__":
    main()

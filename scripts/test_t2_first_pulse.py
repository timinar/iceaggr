#!/usr/bin/env python
"""
Quick test script for T2FirstPulseModel.

Tests that the model can:
1. Load data
2. Process a batch
3. Compute loss
4. Run a backward pass
"""

import torch
from torch.utils.data import DataLoader

from iceaggr.data import IceCubeDataset, collate_dom_packing
from iceaggr.models.t2_first_pulse import T2FirstPulseModel
from iceaggr.training import AngularDistanceLoss
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("Testing T2FirstPulseModel")
    logger.info("=" * 80)

    # Create small dataset
    logger.info("Creating dataset...")
    dataset = IceCubeDataset(split="train", max_events=1000)
    logger.info(f"Dataset size: {len(dataset)} events")

    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=512),
    )

    # Create model
    logger.info("Creating model...")
    model = T2FirstPulseModel(
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.0,
        max_doms=2048,
    )
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Loss and optimizer
    loss_fn = AngularDistanceLoss(use_unit_vectors=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Test forward pass
    logger.info("\nTesting forward pass...")
    model.train()

    batch = next(iter(dataloader))

    # Move batch to device
    for key in ['packed_sequences', 'dom_boundaries', 'dom_mask']:
        batch[key] = batch[key].to(device)
    for key in batch['metadata']:
        if isinstance(batch['metadata'][key], torch.Tensor):
            batch['metadata'][key] = batch['metadata'][key].to(device)

    targets = batch['metadata']['targets'].to(device)

    logger.info(f"Batch size: {batch['metadata']['event_ids'].shape[0]} events")
    logger.info(f"Total DOMs: {batch['metadata']['total_doms']}")
    logger.info(f"Packed sequences shape: {batch['packed_sequences'].shape}")

    # Forward pass
    predictions = model(batch)
    logger.info(f"Predictions shape: {predictions.shape}")

    # Compute loss
    loss = loss_fn(predictions, targets)
    logger.info(f"Loss: {loss.item():.4f}")

    # Backward pass
    logger.info("\nTesting backward pass...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info("Backward pass successful!")

    # Test a few training steps
    logger.info("\nTesting 10 training steps...")
    for step, batch in enumerate(dataloader):
        if step >= 10:
            break

        # Move batch to device
        for key in ['packed_sequences', 'dom_boundaries', 'dom_mask']:
            batch[key] = batch[key].to(device)
        for key in batch['metadata']:
            if isinstance(batch['metadata'][key], torch.Tensor):
                batch['metadata'][key] = batch['metadata'][key].to(device)

        targets = batch['metadata']['targets'].to(device)

        # Forward
        predictions = model(batch)
        loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"Step {step + 1}/10: loss = {loss.item():.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS! T2FirstPulseModel works correctly")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

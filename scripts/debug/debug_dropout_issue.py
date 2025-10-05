#!/usr/bin/env python3
"""
Debug script to test if dropout is causing NaN issues during training.

Tests:
1. Training with dropout=0 (should work based on our tests)
2. Training with dropout=0.1 (current config)
3. Check when NaN first appears with dropout enabled
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceaggr.data import IceCubeDataset
from iceaggr.models import HierarchicalTransformer
from iceaggr.training.losses import AngularDistanceLoss

def test_training_with_dropout(dropout: float, max_steps: int = 100):
    """Test training with specific dropout value."""
    print(f"\n{'='*60}")
    print(f"Testing with dropout={dropout}")
    print(f"{'='*60}\n")

    # Load config
    with open("src/iceaggr/data/data_config.yaml") as f:
        data_config = yaml.safe_load(f)

    # Create dataset and loader
    dataset = IceCubeDataset(
        data_dir=data_config["data"]["train"],
        max_events=10000,
        split="train",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )

    # Create model with specified dropout
    model = HierarchicalTransformer(
        d_model=128,
        t1_n_heads=4,
        t1_n_layers=4,
        t1_max_seq_len=512,
        t1_max_batch_size=64,
        t2_n_heads=4,
        t2_n_layers=4,
        t2_max_doms=2048,
        dropout=dropout,
    ).cuda()

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = AngularDistanceLoss()

    # Training loop
    model.train()

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        # Move to GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch)

        # Check for NaN in predictions
        if not torch.isfinite(predictions).all():
            print(f"❌ NaN detected at step {step}!")
            print(f"   Predictions stats: min={predictions.min():.4f}, max={predictions.max():.4f}")
            print(f"   NaN count: {(~torch.isfinite(predictions)).sum().item()}")
            return False

        # Loss and backward
        targets = batch["direction"]
        loss = loss_fn(predictions, targets)

        if not torch.isfinite(loss):
            print(f"❌ NaN in loss at step {step}!")
            return False

        loss.backward()

        # Check gradients
        max_grad = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    print(f"❌ NaN in gradients at step {step}: {name}")
                    return False
                max_grad = max(max_grad, param.grad.abs().max().item())

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d} | Loss: {loss.item():.4f} | Max grad: {max_grad:.2f}")

    print(f"\n✅ Successfully completed {max_steps} steps with dropout={dropout}")
    return True

if __name__ == "__main__":
    # Test 1: dropout=0 (should work)
    success_no_dropout = test_training_with_dropout(dropout=0.0, max_steps=100)

    # Test 2: dropout=0.1 (current failing config)
    success_with_dropout = test_training_with_dropout(dropout=0.1, max_steps=100)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dropout=0.0:  {'✅ PASSED' if success_no_dropout else '❌ FAILED'}")
    print(f"Dropout=0.1:  {'✅ PASSED' if success_with_dropout else '❌ FAILED'}")
    print()

    if success_no_dropout and not success_with_dropout:
        print("⚠️  CONCLUSION: Dropout is causing the NaN issue!")
        print("    Recommendation: Start training with dropout=0, then gradually increase")
    elif success_with_dropout:
        print("✅ CONCLUSION: Dropout is NOT the issue!")
        print("    The problem must be elsewhere (batch composition, extreme events, etc.)")

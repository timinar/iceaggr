"""
Quick test script to verify DeepSets end-to-end pipeline.

Tests:
1. Data loading with collate_deepsets
2. Model forward pass
3. Loss computation
4. Basic gradient flow
"""

import torch
from iceaggr.data.dataset import get_dataloader
from iceaggr.models import HierarchicalIceCubeModel
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def test_e2e_pipeline():
    """Test end-to-end pipeline with small batch."""
    logger.info("=" * 60)
    logger.info("Testing DeepSets End-to-End Pipeline")
    logger.info("=" * 60)

    # Create small dataloader
    logger.info("\n1. Loading data...")
    dataloader = get_dataloader(
        split='train',
        batch_size=4,
        shuffle=False,
        max_events=16,  # Just load 16 events
        collate_fn='deepsets',
        num_workers=0
    )
    logger.info(f"   Created dataloader with {len(dataloader.dataset)} events")

    # Create model
    logger.info("\n2. Creating model...")
    model = HierarchicalIceCubeModel(
        d_pulse=4,
        d_dom_embedding=64,  # Smaller for testing
        dom_latent_dim=64,
        dom_hidden_dim=128,
        d_model=128,
        n_heads=4,
        n_layers=2,  # Fewer layers for testing
        dropout=0.1,
        use_geometry=False  # No geometry for now
    )
    logger.info(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    logger.info("\n3. Testing forward pass...")
    batch = next(iter(dataloader))
    logger.info(f"   Batch loaded:")
    logger.info(f"     - Total pulses: {batch['pulse_features'].shape[0]}")
    logger.info(f"     - Total DOMs: {batch['num_doms']}")
    logger.info(f"     - Batch size: {batch['batch_size']}")
    logger.info(f"     - Events: {batch['event_ids']}")

    # Forward pass
    with torch.no_grad():
        predictions = model(batch)

    logger.info(f"\n   Predictions shape: {predictions.shape}")
    logger.info(f"   Predictions (azimuth, zenith):")
    for i in range(batch['batch_size']):
        logger.info(f"     Event {i}: az={predictions[i, 0]:.3f}, zen={predictions[i, 1]:.3f}")

    # Test loss computation
    logger.info("\n4. Testing loss computation...")
    targets = batch['targets']
    loss = model.compute_loss(predictions, targets)
    logger.info(f"   Loss: {loss.item():.6f}")

    # Test gradient flow
    logger.info("\n5. Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    predictions = model(batch)
    loss = model.compute_loss(predictions, batch['targets'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info(f"   Backward pass successful!")
    logger.info(f"   Loss after backward: {loss.item():.6f}")

    # Check gradient magnitudes
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    logger.info(f"   Total gradient norm: {total_grad_norm:.6f}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_e2e_pipeline()

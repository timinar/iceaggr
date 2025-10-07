"""
Single-batch overfit test for DeepSets model.

This test checks if the model can overfit to a single batch, which validates:
1. Gradient flow is working
2. Loss function is correct
3. Model has enough capacity
4. Data pipeline is correct

If this fails, there's a fundamental bug in the model/pipeline.
"""

import torch
from iceaggr.data.dataset import get_dataloader
from iceaggr.models import HierarchicalIceCubeModel
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def test_single_batch_overfit():
    """Test if model can overfit to a single batch."""
    logger.info("=" * 70)
    logger.info("SINGLE BATCH OVERFIT TEST")
    logger.info("=" * 70)

    # Create dataloader
    logger.info("\n1. Loading single batch...")
    dataloader = get_dataloader(
        split='train',
        batch_size=32,  # Small batch
        shuffle=False,
        max_events=100,
        collate_fn='deepsets',
        num_workers=0
    )

    # Get single batch
    batch = next(iter(dataloader))
    logger.info(f"   Batch size: {batch['batch_size']}")
    logger.info(f"   Total pulses: {batch['pulse_features'].shape[0]}")
    logger.info(f"   Total DOMs: {batch['num_doms']}")

    # Create model
    logger.info("\n2. Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HierarchicalIceCubeModel(
        d_pulse=6,  # Changed to 6: [time, charge, x, y, z, auxiliary]
        d_dom_embedding=64,
        dom_latent_dim=64,
        dom_hidden_dim=128,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=0.0,  # NO DROPOUT for overfit test
        use_geometry=True,
    ).to(device)

    logger.info(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    logger.info("\n3. Training on single batch...")
    logger.info("-" * 70)
    logger.info(f"{'Step':<8} {'Loss':<12} {'Az MAE':<12} {'Zen MAE':<12}")
    logger.info("-" * 70)

    model.train()
    losses = []

    for step in range(500):  # 500 steps should be enough to overfit
        # Forward
        predictions = model(batch)
        loss = model.compute_loss(predictions, batch['targets'])

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check gradients
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        optimizer.step()

        # Compute errors for monitoring (convert unit vectors back to angles)
        with torch.no_grad():
            from iceaggr.training.losses import unit_vector_to_angles

            # Convert predictions (unit vectors) to angles
            pred_angles = unit_vector_to_angles(predictions)
            az_pred = pred_angles[:, 0]
            zen_pred = pred_angles[:, 1]

            az_true = batch['targets'][:, 0]
            zen_true = batch['targets'][:, 1]

            az_mae = torch.abs(az_pred - az_true).mean().item()
            zen_mae = torch.abs(zen_pred - zen_true).mean().item()

        losses.append(loss.item())

        # Log every 50 steps
        if (step + 1) % 50 == 0 or step < 5:
            logger.info(
                f"{step + 1:<8} {loss.item():<12.6f} {az_mae:<12.6f} {zen_mae:<12.6f} "
                f"(grad_norm: {total_grad_norm:.4f})"
            )

    # Check results
    logger.info("-" * 70)
    logger.info("\n4. Results:")
    logger.info(f"   Initial loss: {losses[0]:.6f}")
    logger.info(f"   Final loss:   {losses[-1]:.6f}")
    logger.info(f"   Reduction:    {losses[0] - losses[-1]:.6f} ({(losses[0] - losses[-1]) / losses[0] * 100:.1f}%)")

    # Success criteria
    success = losses[-1] < 0.5 * losses[0]  # Should reduce by at least 50%

    logger.info("\n5. Verdict:")
    if success:
        logger.info("   ✅ SUCCESS: Model can overfit to single batch")
        logger.info("   → Gradient flow is working")
        logger.info("   → Loss function is correct")
        logger.info("   → Data pipeline is correct")
    else:
        logger.info("   ❌ FAILURE: Model cannot overfit to single batch")
        logger.info("   → There is a fundamental bug in the model or pipeline")
        logger.info("   → Check:")
        logger.info("     - Are gradients flowing? (check grad norms)")
        logger.info("     - Is loss function correct?")
        logger.info("     - Are inputs/outputs in correct ranges?")
        logger.info("     - Is there a bug in data collation?")

    logger.info("=" * 70)

    return success


if __name__ == "__main__":
    test_single_batch_overfit()

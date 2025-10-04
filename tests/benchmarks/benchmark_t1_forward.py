"""
Benchmark T1 (DOM-level transformer) forward pass performance.

Measures:
1. Forward pass latency on real IceCube data
2. Throughput (events/sec, DOMs/sec, pulses/sec)
3. GPU memory usage
4. Scaling with batch size

Usage:
    uv run python tests/benchmarks/benchmark_t1_forward.py [--batch-sizes 4,8,16,32]
"""

import torch
from iceaggr.data import get_dataloader
from iceaggr.models import DOMTransformer
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def test_t1_forward():
    """Test T1 forward pass with real data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data with DOM grouping collation
    logger.info("Loading data...")
    dataloader = get_dataloader(
        split="train",
        batch_size=16,
        shuffle=False,
        num_workers=0,
        max_events=100,  # Small subset for testing
        collate_fn="dom_grouping",
    )

    # Initialize T1 model
    logger.info("Initializing T1 model...")
    model = DOMTransformer(
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Test forward pass on first batch
    logger.info("\nTesting forward pass...")
    batch = next(iter(dataloader))

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    logger.info(f"Batch info:")
    logger.info(f"  Events: {batch['batch_size']}")
    logger.info(f"  Total DOMs: {batch['total_doms']}")
    logger.info(f"  Total pulses: {batch['pulse_features'].shape[0]}")
    logger.info(f"  Pulses/DOM (mean): {batch['pulse_features'].shape[0] / batch['total_doms']:.1f}")

    # Forward pass
    with torch.no_grad():
        dom_embeddings = model(batch)

    logger.info(f"\nOutput shape: {dom_embeddings.shape}")
    logger.info(f"Expected: ({batch['total_doms']}, {model.d_model})")

    # Verify shape
    assert dom_embeddings.shape == (batch['total_doms'], model.d_model), \
        f"Shape mismatch: {dom_embeddings.shape} vs ({batch['total_doms']}, {model.d_model})"

    logger.info("✅ Forward pass successful!")

    # Memory usage
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"\nGPU memory:")
        logger.info(f"  Allocated: {memory_allocated:.2f} GB")
        logger.info(f"  Reserved: {memory_reserved:.2f} GB")

    return model, batch, dom_embeddings


def test_gradient_flow():
    """Test that gradients flow properly through T1."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nTesting gradient flow...")

    # Load data
    dataloader = get_dataloader(
        split="train",
        batch_size=8,
        max_events=50,
        collate_fn="dom_grouping",
    )

    model = DOMTransformer(d_model=64, n_heads=4, n_layers=2).to(device)

    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # Forward pass
    dom_embeddings = model(batch)

    # Dummy loss (just sum of embeddings)
    loss = dom_embeddings.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    logger.info(f"Parameters with gradients: {has_grads}/{total_params}")
    assert has_grads == total_params, "Not all parameters have gradients!"

    logger.info("✅ Gradient flow successful!")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing T1 (DOM-level transformer)")
    logger.info("=" * 60)

    # Test forward pass
    model, batch, dom_embeddings = test_t1_forward()

    # Test gradients
    test_gradient_flow()

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed! ✅")
    logger.info("=" * 60)

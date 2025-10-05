"""
Benchmark vectorized vs loop-based model implementations.

This script tests both models on the same data and measures:
1. Forward pass time
2. Backward pass time
3. Total training step time
4. Numerical equivalence (outputs should be identical)
"""

import torch
import time
import yaml
from pathlib import Path
import numpy as np

from iceaggr.data.dataset import IceCubeDataset, collate_dom_packing
from iceaggr.models.t2_first_pulse import T2FirstPulseModel
from iceaggr.models.t2_first_pulse_vectorized import T2FirstPulseModelVectorized
from iceaggr.training.losses import AngularDistanceLoss
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def benchmark_forward(model, batch, n_warmup=10, n_runs=100):
    """Benchmark forward pass only."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(batch)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(batch)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed / n_runs


def benchmark_backward(model, batch, loss_fn, n_warmup=10, n_runs=100):
    """Benchmark forward + backward pass."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    targets = batch['metadata']['targets']

    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        predictions = model(batch)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_runs):
        optimizer.zero_grad()
        predictions = model(batch)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed / n_runs


def check_equivalence(model1, model2, batch, loss_fn, tol=1e-4):
    """Check if two models produce identical outputs."""
    model1.eval()
    model2.eval()

    targets = batch['metadata']['targets']

    with torch.no_grad():
        pred1 = model1(batch)
        pred2 = model2(batch)

        loss1 = loss_fn(pred1, targets)
        loss2 = loss_fn(pred2, targets)

    # Check output equivalence
    max_diff = torch.max(torch.abs(pred1 - pred2)).item()
    mean_diff = torch.mean(torch.abs(pred1 - pred2)).item()
    loss_diff = abs(loss1.item() - loss2.item())

    logger.info(f"Output difference - Max: {max_diff:.6e}, Mean: {mean_diff:.6e}")
    logger.info(f"Loss difference: {loss_diff:.6e}")

    is_equivalent = (max_diff < tol) and (loss_diff < tol)

    return is_equivalent, max_diff, mean_diff, loss_diff


def main():
    logger.info("="*80)
    logger.info("Benchmark: Loop-based vs Vectorized Model")
    logger.info("="*80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Model config
    d_model = 128
    n_heads = 4
    n_layers = 4
    max_doms = 256
    dropout = 0.0

    # Load data
    logger.info("Loading dataset...")
    dataset = IceCubeDataset(
        split='train',
        max_events=1000,  # Small dataset for quick benchmarking
    )

    # Get a batch
    logger.info("Loading batch...")
    batch_size = 32  # Start with smaller batch for equivalence check
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dom_packing,
    )
    batch = next(iter(loader))

    # Move to GPU
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    batch['metadata'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch['metadata'].items()}

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Total DOMs in batch: {batch['metadata']['total_doms']}")

    # Create models
    logger.info("\nCreating models...")

    model_loop = T2FirstPulseModel(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_doms=max_doms,
        dropout=dropout,
    ).to(device)

    model_vectorized = T2FirstPulseModelVectorized(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_doms=max_doms,
        dropout=dropout,
    ).to(device)

    # Copy weights to ensure identical initialization
    model_vectorized.load_state_dict(model_loop.state_dict())

    # Loss function
    loss_fn = AngularDistanceLoss()

    # Check equivalence first
    logger.info("\n" + "="*80)
    logger.info("EQUIVALENCE CHECK")
    logger.info("="*80)

    is_equivalent, max_diff, mean_diff, loss_diff = check_equivalence(
        model_loop, model_vectorized, batch, loss_fn
    )

    if is_equivalent:
        logger.info("✓ Models are numerically equivalent!")
    else:
        logger.warning("✗ Models produce different outputs!")
        logger.warning("  This may indicate a bug in the vectorized implementation.")

    # Benchmark forward pass
    logger.info("\n" + "="*80)
    logger.info("FORWARD PASS BENCHMARK")
    logger.info("="*80)

    logger.info("Benchmarking loop-based model...")
    time_loop_fwd = benchmark_forward(model_loop, batch, n_warmup=10, n_runs=100)

    logger.info("Benchmarking vectorized model...")
    time_vec_fwd = benchmark_forward(model_vectorized, batch, n_warmup=10, n_runs=100)

    speedup_fwd = time_loop_fwd / time_vec_fwd

    logger.info(f"\nLoop-based:  {time_loop_fwd*1000:.2f} ms")
    logger.info(f"Vectorized:  {time_vec_fwd*1000:.2f} ms")
    logger.info(f"Speedup:     {speedup_fwd:.2f}x")

    # Benchmark full training step (forward + backward)
    logger.info("\n" + "="*80)
    logger.info("TRAINING STEP BENCHMARK (forward + backward)")
    logger.info("="*80)

    logger.info("Benchmarking loop-based model...")
    time_loop_step = benchmark_backward(model_loop, batch, loss_fn, n_warmup=10, n_runs=50)

    logger.info("Benchmarking vectorized model...")
    time_vec_step = benchmark_backward(model_vectorized, batch, loss_fn, n_warmup=10, n_runs=50)

    speedup_step = time_loop_step / time_vec_step

    logger.info(f"\nLoop-based:  {time_loop_step*1000:.2f} ms")
    logger.info(f"Vectorized:  {time_vec_step*1000:.2f} ms")
    logger.info(f"Speedup:     {speedup_step:.2f}x")

    # Test with larger batch size
    logger.info("\n" + "="*80)
    logger.info("LARGE BATCH BENCHMARK (batch_size=512)")
    logger.info("="*80)

    loader_large = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dom_packing,
    )
    batch_large = next(iter(loader_large))
    batch_large = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch_large.items()}
    batch_large['metadata'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch_large['metadata'].items()}

    logger.info(f"Total DOMs in batch: {batch_large['metadata']['total_doms']}")

    logger.info("Benchmarking loop-based model...")
    time_loop_large = benchmark_forward(model_loop, batch_large, n_warmup=5, n_runs=50)

    logger.info("Benchmarking vectorized model...")
    time_vec_large = benchmark_forward(model_vectorized, batch_large, n_warmup=5, n_runs=50)

    speedup_large = time_loop_large / time_vec_large

    logger.info(f"\nLoop-based:  {time_loop_large*1000:.2f} ms")
    logger.info(f"Vectorized:  {time_vec_large*1000:.2f} ms")
    logger.info(f"Speedup:     {speedup_large:.2f}x")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Equivalence:           {'PASS' if is_equivalent else 'FAIL'}")
    logger.info(f"Forward speedup:       {speedup_fwd:.2f}x")
    logger.info(f"Training step speedup: {speedup_step:.2f}x")
    logger.info(f"Large batch speedup:   {speedup_large:.2f}x")

    if speedup_step > 1.1:
        logger.info("\n✓ Vectorized version is FASTER! Consider switching.")
    elif speedup_step < 0.9:
        logger.warning("\n✗ Vectorized version is SLOWER. Keep loop-based version.")
    else:
        logger.info("\n≈ Performance is similar. Either version works.")


if __name__ == '__main__':
    main()

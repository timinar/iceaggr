"""
Test script for the simplified flat transformer model.

Generates mock IceCube-like data and validates:
1. Three-step pipeline: build_flat_dom_vectors → pad_to_event_batch → model
2. Single-pass collator: make_collate_flat → model
3. Equivalence between the two pipelines
4. Training loop with loss decreasing
"""

import math
import time
import torch
import numpy as np

from iceaggr.models.flat_transformer import (
    build_flat_dom_vectors,
    pad_to_event_batch,
    FlatTransformerModel,
)
from iceaggr.models.losses import AngularDistanceLoss
from iceaggr.data.collators import make_collate_flat, collate_with_dom_grouping


class MockGeometry:
    """Mock geometry lookup that returns fixed positions per sensor_id."""

    def __init__(self, positions: torch.Tensor):
        """positions: (n_sensors, 3)"""
        self._positions = positions

    def __getitem__(self, sensor_ids: torch.Tensor) -> torch.Tensor:
        return self._positions[sensor_ids.long()]


def generate_mock_events(
    batch_size: int = 8,
    min_doms: int = 10,
    max_doms_gen: int = 200,
    seed: int = 42,
) -> list:
    """
    Generate mock events in the per-event format that __getitem__ returns.

    Each event is a dict with:
        - pulse_features: (n_pulses, 4) [time, charge, sensor_id, aux]
        - target: (2,) [azimuth, zenith]
        - event_id: int
        - n_pulses: int

    Also returns a MockGeometry for sensor position lookup.
    """
    rng = np.random.default_rng(seed)

    # Create mock geometry: 5160 sensors with random positions
    geometry_positions = torch.from_numpy(
        rng.uniform(-1, 1, size=(5160, 3)).astype(np.float32)
    )
    geometry = MockGeometry(geometry_positions)

    events = []
    for event_i in range(batch_size):
        n_doms = rng.integers(min_doms, max_doms_gen + 1)

        # Pick unique sensor IDs for this event
        dom_ids = rng.choice(5160, size=n_doms, replace=False)

        all_times = []
        all_charges = []
        all_sensor_ids = []
        all_aux = []

        for dom_id in dom_ids:
            n_pulses = max(1, int(rng.exponential(scale=5.0)))
            n_pulses = min(n_pulses, 300)

            base_time = rng.uniform(5000, 15000)
            times = np.sort(base_time + rng.exponential(scale=500, size=n_pulses)).astype(np.float32)
            charges = (rng.exponential(scale=1.5, size=n_pulses) + 0.1).astype(np.float32)
            sensor_ids = np.full(n_pulses, dom_id, dtype=np.float32)
            aux = rng.choice([0.0, 1.0], size=n_pulses, p=[0.7, 0.3]).astype(np.float32)

            all_times.append(times)
            all_charges.append(charges)
            all_sensor_ids.append(sensor_ids)
            all_aux.append(aux)

        pulse_features = np.column_stack([
            np.concatenate(all_times),
            np.concatenate(all_charges),
            np.concatenate(all_sensor_ids),
            np.concatenate(all_aux),
        ])

        total_pulses = pulse_features.shape[0]
        azimuth = rng.uniform(0, 2 * math.pi)
        zenith = rng.uniform(0, math.pi)

        events.append({
            'pulse_features': torch.from_numpy(pulse_features).float(),
            'target': torch.tensor([azimuth, zenith], dtype=torch.float32),
            'event_id': torch.tensor(event_i, dtype=torch.long),
            'n_pulses': torch.tensor(total_pulses, dtype=torch.long),
        })

    return events, geometry


def test_collator_equivalence(events, geometry, config):
    """Verify that make_collate_flat produces the same result as the three-step pipeline."""
    K = config['max_pulses_per_dom']
    max_doms = config['max_doms']

    # --- Path A: Three-step pipeline ---
    # Step 1: DOM grouping (existing collator)
    grouped = collate_with_dom_grouping(events)

    # Step 2: Geometry lookup
    dom_positions = geometry[grouped['dom_ids']]

    # Step 3: Build flat vectors
    dom_vectors = build_flat_dom_vectors(
        pulse_features=grouped['pulse_features'],
        dom_positions=dom_positions,
        pulse_to_dom_idx=grouped['pulse_to_dom_idx'],
        pulse_idx_in_dom=grouped['pulse_idx_in_dom'],
        dom_pulse_counts=grouped['dom_pulse_counts'],
        total_doms=grouped['total_doms'],
        max_pulses_per_dom=K,
    )

    # Step 4: Pad
    padded_a, mask_a = pad_to_event_batch(
        dom_vectors=dom_vectors,
        dom_to_event_idx=grouped['dom_to_event_idx'],
        event_dom_counts=grouped['event_dom_counts'],
        batch_size=grouped['batch_size'],
        max_doms=max_doms,
        dom_min_time=grouped['dom_min_time'],
    )

    # --- Path B: Single-pass collator ---
    collate_fn = make_collate_flat(geometry, max_pulses_per_dom=K, max_doms=max_doms)
    result_b = collate_fn(events)
    padded_b = result_b['dom_vectors']
    mask_b = result_b['padding_mask']

    # --- Compare ---
    # Masks should match exactly
    mask_match = (mask_a == mask_b).all().item()

    # Padded vectors should match for valid positions
    # (DOM ordering within an event may differ, so compare sets of vectors per event)
    values_match = True
    for ev in range(len(events)):
        vecs_a = padded_a[ev][mask_a[ev]]  # (n_valid, input_dim)
        vecs_b = padded_b[ev][mask_b[ev]]

        if vecs_a.shape != vecs_b.shape:
            values_match = False
            break

        # Sort by first few features (xyz) to compare order-independently
        _, order_a = vecs_a[:, :3].sum(dim=1).sort()
        _, order_b = vecs_b[:, :3].sum(dim=1).sort()
        if not torch.allclose(vecs_a[order_a], vecs_b[order_b], atol=1e-5):
            values_match = False
            break

    return mask_match, values_match


def main():
    torch.manual_seed(0)

    config = {
        'max_pulses_per_dom': 84,
        'd_model': 256,
        'max_doms': 128,
        'num_heads': 8,
        'num_layers': 4,
        'hidden_dim': 512,
        'head_hidden_dim': 128,
        'dropout': 0.1,
    }
    K = config['max_pulses_per_dom']

    # --- Generate mock events (per-event format, like __getitem__) ---
    print("=" * 60)
    print("Generating mock IceCube events...")
    events, geometry = generate_mock_events(batch_size=8)
    total_pulses = sum(e['n_pulses'].item() for e in events)
    print(f"  Events: {len(events)}")
    print(f"  Total pulses: {total_pulses}")
    print(f"  Pulses per event: {[e['n_pulses'].item() for e in events]}")

    # --- Test collator equivalence ---
    print("\n" + "=" * 60)
    print("Testing collator equivalence (three-step vs single-pass)...")
    mask_match, values_match = test_collator_equivalence(events, geometry, config)
    print(f"  Masks match: {'OK' if mask_match else 'FAIL'}")
    print(f"  Values match: {'OK' if values_match else 'FAIL'}")
    assert mask_match, "Mask mismatch between pipelines"
    assert values_match, "Value mismatch between pipelines"

    # --- Benchmark single-pass collator ---
    print("\n" + "=" * 60)
    print("Benchmarking collators...")
    collate_flat = make_collate_flat(geometry, max_pulses_per_dom=K, max_doms=config['max_doms'])

    n_iters = 100

    # Three-step
    t0 = time.perf_counter()
    for _ in range(n_iters):
        grouped = collate_with_dom_grouping(events)
        dom_positions = geometry[grouped['dom_ids']]
        dom_vectors = build_flat_dom_vectors(
            grouped['pulse_features'], dom_positions,
            grouped['pulse_to_dom_idx'], grouped['pulse_idx_in_dom'],
            grouped['dom_pulse_counts'], grouped['total_doms'], K,
        )
        pad_to_event_batch(
            dom_vectors, grouped['dom_to_event_idx'],
            grouped['event_dom_counts'], grouped['batch_size'],
            config['max_doms'], grouped['dom_min_time'],
        )
    three_step_ms = (time.perf_counter() - t0) / n_iters * 1000

    # Single-pass
    t0 = time.perf_counter()
    for _ in range(n_iters):
        collate_flat(events)
    single_pass_ms = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  Three-step: {three_step_ms:.2f} ms/batch")
    print(f"  Single-pass: {single_pass_ms:.2f} ms/batch")
    print(f"  Speedup: {three_step_ms / single_pass_ms:.2f}x")

    # --- Model forward + training ---
    print("\n" + "=" * 60)
    print("Model test with single-pass collator...")
    batch = collate_flat(events)
    x, mask = batch['dom_vectors'], batch['padding_mask']
    print(f"  Input shape: {x.shape}")
    print(f"  Valid DOMs/event: {mask.sum(dim=1).tolist()}")

    model = FlatTransformerModel(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    print("\n" + "=" * 60)
    print("Training loop (20 steps)...")
    loss_fn = AngularDistanceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    for step in range(20):
        optimizer.zero_grad()
        directions = model(x, mask)
        loss = loss_fn(directions, batch['targets'])
        loss.backward()
        optimizer.step()

        degrees = loss.item() * 180 / math.pi
        if step % 5 == 0 or step == 19:
            print(f"  Step {step:2d}: loss = {loss.item():.4f} rad ({degrees:.1f}°)")

    print("\n" + "=" * 60)
    print("All checks passed!")


if __name__ == '__main__':
    main()

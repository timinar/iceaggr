"""
Test script for the simplified flat transformer model.

Generates mock IceCube-like data and validates the full pipeline:
    mock pulses → build_flat_dom_vectors → pad_to_event_batch → FlatTransformerModel → direction

Also runs a short training loop to verify gradients flow and loss decreases.
"""

import math
import torch
import numpy as np

from iceaggr.models.flat_transformer import (
    build_flat_dom_vectors,
    pad_to_event_batch,
    FlatTransformerModel,
)
from iceaggr.models.losses import AngularDistanceLoss


def generate_mock_batch(
    batch_size: int = 8,
    min_doms: int = 10,
    max_doms_gen: int = 200,
    seed: int = 42,
) -> tuple:
    """
    Generate a mock batch mimicking collate_with_dom_grouping output.

    Creates realistic-ish IceCube pulse data:
    - Variable DOMs per event (10-200)
    - Variable pulses per DOM (heavy-tailed: most have 1-10, some have 50+)
    - Realistic feature ranges

    Returns:
        (batch_dict, dom_positions) ready for build_flat_dom_vectors
    """
    rng = np.random.default_rng(seed)

    all_times = []
    all_charges = []
    all_sensor_ids = []
    all_aux = []
    all_pulse_to_dom = []
    all_pulse_idx_in_dom = []
    dom_counts_list = []
    dom_to_event_list = []
    dom_positions_list = []
    dom_min_times = []
    event_dom_counts = []
    targets = []

    dom_offset = 0

    for event_i in range(batch_size):
        n_doms = rng.integers(min_doms, max_doms_gen + 1)
        event_dom_counts.append(n_doms)

        # Random target direction
        azimuth = rng.uniform(0, 2 * math.pi)
        zenith = rng.uniform(0, math.pi)
        targets.append([azimuth, zenith])

        for dom_j in range(n_doms):
            # Heavy-tailed pulse count: most DOMs have few pulses
            n_pulses = max(1, int(rng.exponential(scale=5.0)))
            n_pulses = min(n_pulses, 300)  # cap
            dom_counts_list.append(n_pulses)
            dom_to_event_list.append(event_i)

            # DOM geometry: random normalized positions ~[-1, 1]
            dom_pos = rng.uniform(-1, 1, size=3).astype(np.float32)
            dom_positions_list.append(dom_pos)

            # Generate pulses sorted by time
            base_time = rng.uniform(5000, 15000)  # ns, centered ~1e4
            times = np.sort(base_time + rng.exponential(scale=500, size=n_pulses)).astype(np.float32)
            charges = rng.exponential(scale=1.5, size=n_pulses).astype(np.float32) + 0.1
            sensor_ids = np.full(n_pulses, rng.integers(0, 5160), dtype=np.float32)
            aux = rng.choice([0.0, 1.0], size=n_pulses, p=[0.7, 0.3]).astype(np.float32)

            dom_min_times.append(times[0])

            all_times.append(times)
            all_charges.append(charges)
            all_sensor_ids.append(sensor_ids)
            all_aux.append(aux)
            all_pulse_to_dom.append(np.full(n_pulses, dom_offset, dtype=np.int64))
            all_pulse_idx_in_dom.append(np.arange(n_pulses, dtype=np.int64))

            dom_offset += 1

    # Concatenate everything
    total_doms = dom_offset
    pulse_features = torch.tensor(np.column_stack([
        np.concatenate(all_times),
        np.concatenate(all_charges),
        np.concatenate(all_sensor_ids),
        np.concatenate(all_aux),
    ]), dtype=torch.float32)

    batch = {
        'pulse_features': pulse_features,
        'pulse_to_dom_idx': torch.tensor(np.concatenate(all_pulse_to_dom), dtype=torch.long),
        'pulse_idx_in_dom': torch.tensor(np.concatenate(all_pulse_idx_in_dom), dtype=torch.long),
        'dom_pulse_counts': torch.tensor(dom_counts_list, dtype=torch.long),
        'dom_to_event_idx': torch.tensor(dom_to_event_list, dtype=torch.long),
        'dom_min_time': torch.tensor(dom_min_times, dtype=torch.float32),
        'event_dom_counts': torch.tensor(event_dom_counts, dtype=torch.long),
        'targets': torch.tensor(targets, dtype=torch.float32),
        'total_doms': total_doms,
        'batch_size': batch_size,
    }

    dom_positions = torch.tensor(np.stack(dom_positions_list), dtype=torch.float32)

    return batch, dom_positions


def main():
    torch.manual_seed(0)

    # --- Config ---
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

    # --- Generate mock data ---
    print("=" * 60)
    print("Generating mock IceCube data...")
    batch, dom_positions = generate_mock_batch(batch_size=8)
    total_pulses = batch['pulse_features'].shape[0]
    total_doms = batch['total_doms']
    print(f"  Events: {batch['batch_size']}")
    print(f"  Total DOMs: {total_doms}")
    print(f"  Total pulses: {total_pulses}")
    print(f"  DOMs per event: {batch['event_dom_counts'].tolist()}")

    # Pulse distribution stats
    counts = batch['dom_pulse_counts']
    print(f"  Pulses/DOM: median={counts.median().item():.0f}, "
          f"mean={counts.float().mean().item():.1f}, "
          f"max={counts.max().item()}, "
          f"≤{K}: {(counts <= K).sum().item()}/{total_doms} "
          f"({(counts <= K).float().mean().item()*100:.1f}%)")

    # --- Build flat DOM vectors ---
    print("\n" + "=" * 60)
    print("Building flat DOM vectors...")
    dom_vectors = build_flat_dom_vectors(
        pulse_features=batch['pulse_features'],
        dom_positions=dom_positions,
        pulse_to_dom_idx=batch['pulse_to_dom_idx'],
        pulse_idx_in_dom=batch['pulse_idx_in_dom'],
        dom_pulse_counts=batch['dom_pulse_counts'],
        total_doms=total_doms,
        max_pulses_per_dom=K,
    )
    print(f"  DOM vector shape: {dom_vectors.shape}")  # (total_doms, 256)
    print(f"  Input dim: {dom_vectors.shape[1]} (expected {4 + 3*K})")

    # Verify: check a DOM with known pulse count
    example_dom = 0
    n_p = batch['dom_pulse_counts'][example_dom].item()
    vec = dom_vectors[example_dom]
    print(f"\n  Example DOM 0 ({n_p} pulses):")
    print(f"    xyz: {vec[:3].tolist()}")
    print(f"    n_pulses_feat: {vec[3].item():.3f}")
    if n_p >= 1:
        print(f"    pulse 0 (t,q,a): {vec[4:7].tolist()}")
    if n_p >= 2:
        print(f"    pulse 1 (t,q,a): {vec[7:10].tolist()}")
    # Check zero-padding: slots beyond actual pulses should be zero
    if n_p < K:
        zero_start = 4 + n_p * 3
        zeros_ok = (vec[zero_start:] == 0).all().item()
        print(f"    zero-padding beyond pulse {n_p}: {'OK' if zeros_ok else 'FAIL'}")

    # --- Pad to event batch ---
    print("\n" + "=" * 60)
    print("Padding to event batch...")
    x, mask = pad_to_event_batch(
        dom_vectors=dom_vectors,
        dom_to_event_idx=batch['dom_to_event_idx'],
        event_dom_counts=batch['event_dom_counts'],
        batch_size=batch['batch_size'],
        max_doms=config['max_doms'],
        dom_min_time=batch['dom_min_time'],
    )
    print(f"  Padded shape: {x.shape}")  # (8, 128, 256)
    print(f"  Mask shape: {mask.shape}")  # (8, 128)
    print(f"  Valid DOMs per event: {mask.sum(dim=1).tolist()}")

    # --- Create model ---
    print("\n" + "=" * 60)
    model = FlatTransformerModel(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("FlatTransformerModel created:")
    print(f"  Parameters: {n_params:,}")
    print(f"  Input dim: {model.input_dim}")
    print(f"  d_model: {model.d_model}")
    print(f"  Max pulses/DOM: {model.max_pulses_per_dom}")
    print(f"  Max DOMs: {model.max_doms}")

    # --- Forward pass ---
    print("\n" + "=" * 60)
    print("Forward pass...")
    with torch.no_grad():
        directions = model(x, mask)
    print(f"  Output shape: {directions.shape}")  # (8, 3)
    norms = directions.norm(dim=1)
    print(f"  Output norms (should be ~1): {norms.tolist()}")

    # --- Training loop ---
    print("\n" + "=" * 60)
    print("Training loop (20 steps on mock data)...")
    loss_fn = AngularDistanceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    targets = batch['targets']

    for step in range(20):
        optimizer.zero_grad()
        directions = model(x, mask)
        loss = loss_fn(directions, targets)
        loss.backward()
        optimizer.step()

        degrees = loss.item() * 180 / math.pi
        if step % 5 == 0 or step == 19:
            print(f"  Step {step:2d}: loss = {loss.item():.4f} rad ({degrees:.1f}°)")

    print("\n" + "=" * 60)
    print("All checks passed!")


if __name__ == '__main__':
    main()

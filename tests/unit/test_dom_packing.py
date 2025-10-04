"""
Unit tests for DOM packing collate function and updated DOMTransformer.
"""

import torch
import pytest
from iceaggr.data.dataset import collate_dom_packing
from iceaggr.models.dom_transformer import DOMTransformer


def create_mock_event(n_pulses: int, n_doms: int, event_id: int = 0):
    """Create a mock event with specified number of pulses and DOMs."""
    # Distribute pulses across DOMs
    pulses_per_dom = n_pulses // n_doms
    remaining = n_pulses % n_doms

    pulse_features = []
    for dom_id in range(n_doms):
        n_pulses_this_dom = pulses_per_dom + (1 if dom_id < remaining else 0)
        # Features: [time, charge, sensor_id, auxiliary]
        dom_pulses = torch.tensor([
            [float(i), 1.0, float(dom_id), 0.0]
            for i in range(n_pulses_this_dom)
        ])
        pulse_features.append(dom_pulses)

    pulse_features = torch.cat(pulse_features, dim=0)

    return {
        'pulse_features': pulse_features,
        'event_id': torch.tensor(event_id),
        'target': torch.tensor([0.5, 0.5])  # Mock azimuth, zenith
    }


class TestDOMPacking:
    """Test DOM packing collate function."""

    def test_single_event_sparse_doms(self):
        """Test packing a single event with sparse DOMs (few pulses per DOM)."""
        event = create_mock_event(n_pulses=100, n_doms=50)  # ~2 pulses per DOM
        batch = [event]

        result = collate_dom_packing(batch, max_seq_len=512)

        # Check output shapes
        assert result['packed_sequences'].dim() == 3
        assert result['packed_sequences'].shape[1] == 512  # max_seq_len
        assert result['packed_sequences'].shape[2] == 4   # features

        assert result['dom_boundaries'].shape == result['packed_sequences'].shape[:2]
        assert result['dom_mask'].shape == result['packed_sequences'].shape[:2]

        # Check metadata
        assert result['metadata']['total_doms'] == 50
        assert result['metadata']['dom_to_event_idx'].shape[0] == 50

    def test_multiple_events(self):
        """Test packing multiple events."""
        batch = [
            create_mock_event(n_pulses=50, n_doms=25, event_id=0),
            create_mock_event(n_pulses=80, n_doms=40, event_id=1),
        ]

        result = collate_dom_packing(batch, max_seq_len=512)

        # Total DOMs should be sum of DOMs from both events
        assert result['metadata']['total_doms'] == 25 + 40

        # Check that event IDs are preserved
        assert len(result['metadata']['event_ids']) == 2

    def test_large_dom_chunking(self):
        """Test that DOMs with >max_seq_len pulses are chunked."""
        # Create event with one DOM having many pulses
        event = {
            'pulse_features': torch.tensor([
                [float(i), 1.0, 0.0, 0.0] for i in range(1000)  # 1000 pulses, 1 DOM
            ]),
            'event_id': torch.tensor(0),
            'target': torch.tensor([0.5, 0.5])
        }
        batch = [event]

        result = collate_dom_packing(batch, max_seq_len=512)

        # Should chunk to max_seq_len
        valid_pulses = result['dom_mask'][0].sum()
        assert valid_pulses <= 512

    def test_dom_boundaries_correct(self):
        """Test that DOM boundaries are correctly set."""
        event = create_mock_event(n_pulses=20, n_doms=5)  # 4 pulses per DOM
        batch = [event]

        result = collate_dom_packing(batch, max_seq_len=512)

        # Check that boundaries increment correctly
        boundaries = result['dom_boundaries'][0]
        mask = result['dom_mask'][0]

        # Valid positions should have boundaries
        valid_boundaries = boundaries[mask.bool()]
        assert valid_boundaries.min() >= 0
        assert len(torch.unique(valid_boundaries)) == 5  # 5 unique DOMs


class TestDOMTransformer:
    """Test updated DOMTransformer with packing."""

    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct shapes."""
        # Create model
        model = DOMTransformer(
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=512
        )

        # Create batch
        batch = [create_mock_event(n_pulses=100, n_doms=50)]
        collated = collate_dom_packing(batch, max_seq_len=512)

        # Forward pass
        dom_embeddings, metadata = model(collated)

        # Check output shape
        assert dom_embeddings.shape == (50, 64)  # (total_doms, d_model)
        assert metadata['total_doms'] == 50

    def test_multiple_events_forward(self):
        """Test forward pass with multiple events."""
        model = DOMTransformer(d_model=64, n_heads=4, n_layers=2, max_seq_len=512)

        batch = [
            create_mock_event(n_pulses=50, n_doms=25, event_id=0),
            create_mock_event(n_pulses=80, n_doms=40, event_id=1),
        ]
        collated = collate_dom_packing(batch, max_seq_len=512)

        dom_embeddings, metadata = model(collated)

        # Total 65 DOMs across both events
        assert dom_embeddings.shape == (65, 64)
        assert metadata['total_doms'] == 65

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = DOMTransformer(d_model=64, n_heads=4, n_layers=2, max_seq_len=512)

        batch = [create_mock_event(n_pulses=50, n_doms=25)]
        collated = collate_dom_packing(batch, max_seq_len=512)

        # Forward pass
        dom_embeddings, _ = model(collated)

        # Dummy loss
        loss = dom_embeddings.mean()
        loss.backward()

        # Check that gradients exist
        assert model.input_projection.weight.grad is not None
        assert model.layers[0].qkv_proj.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

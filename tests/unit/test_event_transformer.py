"""
Unit tests for EventTransformer (T2) and EventAccumulator.
"""

import torch
import pytest
from iceaggr.models import EventTransformer, EventAccumulator


def create_mock_dom_embeddings(n_events: int, max_doms_per_event: int = 100, d_model: int = 128):
    """Create mock DOM embeddings for testing."""
    # Variable number of DOMs per event
    doms_per_event = torch.randint(10, max_doms_per_event, (n_events,))
    total_doms = doms_per_event.sum().item()

    # Create embeddings
    dom_embeddings = torch.randn(total_doms, d_model)

    # Create sensor IDs (0-5159 valid range)
    sensor_ids = torch.randint(0, 5160, (total_doms,))

    # Create event mapping
    dom_to_event_idx = torch.cat([
        torch.full((n,), i, dtype=torch.long)
        for i, n in enumerate(doms_per_event)
    ])

    return {
        'dom_embeddings': dom_embeddings,
        'sensor_ids': sensor_ids,
        'dom_to_event_idx': dom_to_event_idx,
        'batch_size': n_events,
        'doms_per_event': doms_per_event,
    }


class TestEventTransformer:
    """Test EventTransformer (T2) module."""

    def test_initialization(self):
        """Test that EventTransformer initializes correctly."""
        model = EventTransformer(d_model=64, n_heads=4, n_layers=2)

        assert model.d_model == 64
        assert model.n_heads == 4
        assert model.n_layers == 2
        assert model.sensor_geometry.shape[0] == 5160  # IceCube has 5160 sensors

    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct output shapes."""
        model = EventTransformer(d_model=128, n_heads=8, n_layers=2)

        # Create mock data
        batch = create_mock_dom_embeddings(n_events=4, max_doms_per_event=50)

        # Forward pass
        predictions = model(
            batch['dom_embeddings'],
            batch['sensor_ids'],
            batch['dom_to_event_idx'],
            batch['batch_size']
        )

        # Check output shape
        assert predictions.shape == (4, 2)  # (batch_size, 2) for azimuth, zenith
        assert not torch.isnan(predictions).any()

    def test_variable_dom_counts(self):
        """Test handling of events with different numbers of DOMs."""
        model = EventTransformer(d_model=64, n_heads=4, n_layers=2)

        # Create events with very different DOM counts
        dom_embeddings = torch.randn(155, 64)  # Total 155 DOMs
        sensor_ids = torch.randint(0, 5160, (155,))

        # Event 0: 5 DOMs, Event 1: 50 DOMs, Event 2: 100 DOMs
        dom_to_event_idx = torch.cat([
            torch.zeros(5, dtype=torch.long),
            torch.ones(50, dtype=torch.long),
            torch.full((100,), 2, dtype=torch.long),
        ])

        predictions = model(dom_embeddings, sensor_ids, dom_to_event_idx, batch_size=3)

        assert predictions.shape == (3, 2)
        assert not torch.isnan(predictions).any()

    def test_single_dom_event(self):
        """Test handling of events with just one DOM."""
        model = EventTransformer(d_model=64, n_heads=4, n_layers=2)

        # Single event with single DOM
        dom_embeddings = torch.randn(1, 64)
        sensor_ids = torch.tensor([100])
        dom_to_event_idx = torch.tensor([0])

        predictions = model(dom_embeddings, sensor_ids, dom_to_event_idx, batch_size=1)

        assert predictions.shape == (1, 2)
        assert not torch.isnan(predictions).any()

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = EventTransformer(d_model=64, n_heads=4, n_layers=2)

        batch = create_mock_dom_embeddings(n_events=2, max_doms_per_event=20, d_model=64)

        # Forward pass
        predictions = model(
            batch['dom_embeddings'],
            batch['sensor_ids'],
            batch['dom_to_event_idx'],
            batch['batch_size']
        )

        # Dummy loss
        loss = predictions.mean()
        loss.backward()

        # Check gradients
        assert model.geometry_encoder[0].weight.grad is not None
        assert model.prediction_head[0].weight.grad is not None

    def test_batch_consistency(self):
        """Test that batch processing gives same results as individual processing."""
        model = EventTransformer(d_model=64, n_heads=4, n_layers=2)
        model.eval()

        # Create batch of 2 events
        batch = create_mock_dom_embeddings(n_events=2, max_doms_per_event=30, d_model=64)

        with torch.no_grad():
            # Process as batch
            batch_pred = model(
                batch['dom_embeddings'],
                batch['sensor_ids'],
                batch['dom_to_event_idx'],
                batch['batch_size']
            )

            # Process individually
            event0_mask = (batch['dom_to_event_idx'] == 0)
            event1_mask = (batch['dom_to_event_idx'] == 1)

            pred0 = model(
                batch['dom_embeddings'][event0_mask],
                batch['sensor_ids'][event0_mask],
                torch.zeros(event0_mask.sum(), dtype=torch.long),
                batch_size=1
            )

            pred1 = model(
                batch['dom_embeddings'][event1_mask],
                batch['sensor_ids'][event1_mask],
                torch.zeros(event1_mask.sum(), dtype=torch.long),
                batch_size=1
            )

            # Should be very close (within numerical precision)
            assert torch.allclose(batch_pred[0], pred0[0], atol=1e-5)
            assert torch.allclose(batch_pred[1], pred1[0], atol=1e-5)


class TestEventAccumulator:
    """Test EventAccumulator for multi-batch event assembly."""

    def test_single_batch_accumulation(self):
        """Test accumulating DOMs from a single T1 batch."""
        accumulator = EventAccumulator()

        # Mock T1 output
        dom_embeddings = torch.randn(50, 128)
        metadata = {
            'dom_to_event_idx': torch.cat([
                torch.zeros(20, dtype=torch.long),  # Event 0: 20 DOMs
                torch.ones(30, dtype=torch.long),   # Event 1: 30 DOMs
            ]),
            'sensor_ids': torch.randint(0, 5160, (50,)),
            'event_ids': torch.tensor([100, 200]),
            'targets': torch.tensor([[0.5, 0.5], [1.0, 1.0]]),
        }

        accumulator.add_batch(dom_embeddings, metadata)

        assert len(accumulator.events) == 2
        assert 100 in accumulator.events
        assert 200 in accumulator.events

    def test_multi_batch_accumulation(self):
        """Test accumulating DOMs across multiple T1 batches."""
        accumulator = EventAccumulator()

        # Batch 1: Partial event 100 + full event 200
        dom_emb1 = torch.randn(40, 128)
        meta1 = {
            'dom_to_event_idx': torch.cat([
                torch.zeros(15, dtype=torch.long),  # Event 100: partial (15 DOMs)
                torch.ones(25, dtype=torch.long),   # Event 200: full (25 DOMs)
            ]),
            'sensor_ids': torch.randint(0, 5160, (40,)),
            'event_ids': torch.tensor([100, 200]),
            'targets': torch.tensor([[0.5, 0.5], [1.0, 1.0]]),
        }

        # Batch 2: Rest of event 100 + new event 300
        dom_emb2 = torch.randn(45, 128)
        meta2 = {
            'dom_to_event_idx': torch.cat([
                torch.zeros(20, dtype=torch.long),  # Event 100: rest (20 DOMs)
                torch.ones(25, dtype=torch.long),   # Event 300: full (25 DOMs)
            ]),
            'sensor_ids': torch.randint(0, 5160, (45,)),
            'event_ids': torch.tensor([100, 300]),
            'targets': torch.tensor([[0.5, 0.5], [2.0, 2.0]]),
        }

        accumulator.add_batch(dom_emb1, meta1)
        accumulator.add_batch(dom_emb2, meta2)

        # Check accumulation
        assert len(accumulator.events) == 3

        # Event 100 should have 2 chunks (15 + 20 DOMs)
        assert len(accumulator.events[100]['dom_embeddings']) == 2

        # Event 200 should have 1 chunk
        assert len(accumulator.events[200]['dom_embeddings']) == 1

        # Event 300 should have 1 chunk
        assert len(accumulator.events[300]['dom_embeddings']) == 1

    def test_get_complete_events(self):
        """Test retrieving complete events for T2."""
        accumulator = EventAccumulator()

        # Add some events
        for event_id in [100, 200, 300]:
            dom_emb = torch.randn(20, 128)
            meta = {
                'dom_to_event_idx': torch.zeros(20, dtype=torch.long),
                'sensor_ids': torch.randint(0, 5160, (20,)),
                'event_ids': torch.tensor([event_id]),
                'targets': torch.tensor([[0.5 * event_id/100, 0.5 * event_id/100]]),
            }
            accumulator.add_batch(dom_emb, meta)

        # Get batches
        batches = list(accumulator.get_complete_events(batch_size=2))

        assert len(batches) == 2  # 3 events, batch_size=2 → 2 batches

        # First batch: 2 events
        assert batches[0]['batch_size'] == 2
        assert batches[0]['dom_embeddings'].shape[0] == 40  # 2 events * 20 DOMs

        # Second batch: 1 event
        assert batches[1]['batch_size'] == 1
        assert batches[1]['dom_embeddings'].shape[0] == 20  # 1 event * 20 DOMs

    def test_clear(self):
        """Test clearing accumulated events."""
        accumulator = EventAccumulator()

        dom_emb = torch.randn(10, 128)
        meta = {
            'dom_to_event_idx': torch.zeros(10, dtype=torch.long),
            'sensor_ids': torch.randint(0, 5160, (10,)),
            'event_ids': torch.tensor([100]),
        }
        accumulator.add_batch(dom_emb, meta)

        assert len(accumulator.events) == 1

        accumulator.clear()
        assert len(accumulator.events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
End-to-end integration tests for T1 → T2 pipeline.
"""

import torch
import pytest
from iceaggr.models import DOMTransformer, EventTransformer, EventAccumulator
from iceaggr.data import IceCubeDataset, collate_dom_packing
from torch.utils.data import DataLoader


class TestE2EPipeline:
    """Test end-to-end T1 → T2 pipeline."""

    def test_single_batch_pipeline(self):
        """Test T1 → T2 on a single batch (all events fit in one T1 batch)."""
        # Initialize models
        t1 = DOMTransformer(d_model=64, n_heads=4, n_layers=2, max_seq_len=512)
        t2 = EventTransformer(d_model=64, n_heads=4, n_layers=2)

        # Load small batch of real data
        dataset = IceCubeDataset(
            config_path="src/iceaggr/data/data_config.yaml",
            split="train",
            max_events=8
        )

        loader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=512)
        )

        batch = next(iter(loader))

        # T1 forward pass
        dom_embeddings, metadata = t1(batch)

        assert dom_embeddings.shape[1] == 64  # d_model
        assert dom_embeddings.shape[0] == metadata['total_doms']

        # T2 forward pass
        predictions = t2(
            dom_embeddings,
            metadata['sensor_ids'],
            metadata['dom_to_event_idx'],
            batch_size=8
        )

        assert predictions.shape == (8, 2)  # (batch_size, 2) for azimuth, zenith
        assert not torch.isnan(predictions).any()

    def test_multi_batch_pipeline_with_accumulator(self):
        """Test T1 → EventAccumulator → T2 for events split across batches."""
        # Initialize models
        t1 = DOMTransformer(d_model=64, n_heads=4, n_layers=2, max_seq_len=512)
        t2 = EventTransformer(d_model=64, n_heads=4, n_layers=2)
        accumulator = EventAccumulator()

        # Load data (simulate multiple T1 batches)
        dataset = IceCubeDataset(
            config_path="src/iceaggr/data/data_config.yaml",
            split="train",
            max_events=16
        )

        loader = DataLoader(
            dataset,
            batch_size=8,  # Small batches to simulate splitting
            collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=512)
        )

        # Process T1 batches
        for batch in loader:
            dom_embeddings, metadata = t1(batch)
            accumulator.add_batch(dom_embeddings, metadata)

        # Get complete events for T2
        t2_batches = list(accumulator.get_complete_events(batch_size=8))

        # Should have accumulated all 16 events
        total_events = sum(b['batch_size'] for b in t2_batches)
        assert total_events == 16

        # Process T2 batches
        all_predictions = []
        for t2_batch in t2_batches:
            predictions = t2(
                t2_batch['dom_embeddings'],
                t2_batch['dom_ids'],
                t2_batch['dom_to_event_idx'],
                t2_batch['batch_size']
            )
            all_predictions.append(predictions)

        all_predictions = torch.cat(all_predictions, dim=0)

        assert all_predictions.shape == (16, 2)
        assert not torch.isnan(all_predictions).any()

    def test_gradient_flow_e2e(self):
        """Test that gradients flow through entire T1 → T2 pipeline."""
        # Initialize models
        t1 = DOMTransformer(d_model=64, n_heads=4, n_layers=2, max_seq_len=512)
        t2 = EventTransformer(d_model=64, n_heads=4, n_layers=2)

        # Load small batch
        dataset = IceCubeDataset(
            config_path="src/iceaggr/data/data_config.yaml",
            split="train",
            max_events=4
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=lambda batch: collate_dom_packing(batch, max_seq_len=512)
        )

        batch = next(iter(loader))

        # Forward pass
        dom_embeddings, metadata = t1(batch)
        predictions = t2(
            dom_embeddings,
            metadata['sensor_ids'],
            metadata['dom_to_event_idx'],
            batch_size=4
        )

        # Compute loss
        targets = metadata['targets']  # (4, 2)
        loss = torch.nn.functional.mse_loss(predictions, targets)

        # Backward pass
        loss.backward()

        # Check gradients in both models
        assert t1.input_projection.weight.grad is not None
        assert t1.layers[0].qkv_proj.weight.grad is not None

        assert t2.geometry_encoder[0].weight.grad is not None
        assert t2.prediction_head[0].weight.grad is not None

    def test_extreme_event_handling(self):
        """Test pipeline handles extreme events (many DOMs)."""
        # Initialize models
        t1 = DOMTransformer(d_model=64, n_heads=4, n_layers=2, max_seq_len=512)
        t2 = EventTransformer(d_model=64, n_heads=4, n_layers=2)

        # Find an extreme event
        dataset = IceCubeDataset(
            config_path="src/iceaggr/data/data_config.yaml",
            split="train",
            max_events=1000
        )

        # Find event with many DOMs
        max_doms = 0
        extreme_idx = 0
        for i in range(len(dataset)):
            sample = dataset[i]
            sensor_ids = sample['pulse_features'][:, 2].long()
            n_doms = len(torch.unique(sensor_ids))
            if n_doms > max_doms:
                max_doms = n_doms
                extreme_idx = i

        print(f"Found extreme event with {max_doms} DOMs at index {extreme_idx}")

        # Process extreme event
        batch = collate_dom_packing([dataset[extreme_idx]], max_seq_len=512)

        dom_embeddings, metadata = t1(batch)
        predictions = t2(
            dom_embeddings,
            metadata['sensor_ids'],
            metadata['dom_to_event_idx'],
            batch_size=1
        )

        assert predictions.shape == (1, 2)
        assert not torch.isnan(predictions).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

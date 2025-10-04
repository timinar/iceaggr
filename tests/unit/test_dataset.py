"""
Unit tests for IceCube dataset functionality.
"""

import pytest
import torch
import numpy as np
from iceaggr.data import IceCubeDataset, collate_variable_length, get_dataloader


class TestIceCubeDataset:
    """Test IceCubeDataset class."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for testing."""
        return IceCubeDataset(split="train", max_events=100)

    def test_dataset_length(self, small_dataset):
        """Test dataset returns correct length."""
        assert len(small_dataset) == 100

    def test_getitem_returns_dict(self, small_dataset):
        """Test __getitem__ returns a dictionary with expected keys."""
        event = small_dataset[0]

        assert isinstance(event, dict)
        assert "pulse_features" in event
        assert "event_id" in event
        assert "n_pulses" in event
        assert "target" in event  # train split should have targets

    def test_pulse_features_shape(self, small_dataset):
        """Test pulse features have correct shape."""
        event = small_dataset[0]

        pulse_features = event["pulse_features"]
        n_pulses = event["n_pulses"].item()

        # Should be (n_pulses, 4) for [time, charge, sensor_id, auxiliary]
        assert pulse_features.shape == (n_pulses, 4)
        assert pulse_features.dtype == torch.float32

    def test_pulse_features_content(self, small_dataset):
        """Test pulse features contain reasonable values."""
        event = small_dataset[0]
        pulse_features = event["pulse_features"]

        # time (column 0)
        time = pulse_features[:, 0]
        assert torch.isfinite(time).all(), "time should be finite"

        # charge (column 1)
        charge = pulse_features[:, 1]
        assert torch.isfinite(charge).all(), "charge should be finite"
        assert (charge > 0).all(), "charge should be positive"

        # sensor_id (column 2)
        sensor_id = pulse_features[:, 2]
        assert (sensor_id >= 0).all(), "sensor_id should be >= 0"
        assert (sensor_id <= 5159).all(), "sensor_id should be <= 5159"

        # auxiliary (column 3)
        auxiliary = pulse_features[:, 3]
        unique_aux = torch.unique(auxiliary)
        assert torch.all((unique_aux == 0) | (unique_aux == 1)), \
            "auxiliary should only be 0 or 1"

    def test_targets_shape(self, small_dataset):
        """Test targets have correct shape."""
        event = small_dataset[0]
        target = event["target"]

        # Should be (2,) for [azimuth, zenith]
        assert target.shape == (2,)
        assert target.dtype == torch.float32

    def test_targets_range(self, small_dataset):
        """Test targets are in valid ranges."""
        event = small_dataset[0]
        azimuth, zenith = event["target"]

        # Azimuth in [0, 2π]
        assert 0 <= azimuth <= 2 * np.pi + 0.01

        # Zenith in [0, π]
        assert 0 <= zenith <= np.pi + 0.01

    def test_event_id_type(self, small_dataset):
        """Test event_id has correct type."""
        event = small_dataset[0]
        event_id = event["event_id"]

        assert isinstance(event_id, torch.Tensor)
        assert event_id.dtype == torch.long

    def test_variable_event_lengths(self, small_dataset):
        """Test that dataset handles variable event lengths."""
        event_lengths = [small_dataset[i]["n_pulses"].item() for i in range(len(small_dataset))]

        # Should have variance (not all events same length)
        assert len(set(event_lengths)) > 1, "Events should have different lengths"
        assert min(event_lengths) >= 5, "All events should have at least 5 pulses"


class TestCollateFunction:
    """Test collate_variable_length function."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch of events for testing."""
        dataset = IceCubeDataset(split="train", max_events=10)
        return [dataset[i] for i in range(5)]

    def test_collate_returns_dict(self, sample_batch):
        """Test collate returns a dictionary."""
        batch = collate_variable_length(sample_batch)
        assert isinstance(batch, dict)

    def test_collate_keys(self, sample_batch):
        """Test collate returns expected keys."""
        batch = collate_variable_length(sample_batch)

        expected_keys = {
            "pulse_features",
            "pulse_to_event_idx",
            "event_lengths",
            "event_ids",
            "targets"
        }
        assert set(batch.keys()) == expected_keys

    def test_collate_flattens_pulses(self, sample_batch):
        """Test that collate properly flattens pulses."""
        batch = collate_variable_length(sample_batch)

        total_pulses = sum(event["n_pulses"].item() for event in sample_batch)

        assert batch["pulse_features"].shape[0] == total_pulses
        assert batch["pulse_to_event_idx"].shape[0] == total_pulses

    def test_collate_event_mapping(self, sample_batch):
        """Test pulse-to-event mapping is correct."""
        batch = collate_variable_length(sample_batch)

        pulse_to_event = batch["pulse_to_event_idx"]
        event_lengths = batch["event_lengths"]

        # Check mapping is correct
        for event_idx in range(len(sample_batch)):
            # Count pulses assigned to this event
            pulses_for_event = (pulse_to_event == event_idx).sum().item()
            assert pulses_for_event == event_lengths[event_idx].item()

    def test_collate_preserves_batch_size(self, sample_batch):
        """Test that batch size is preserved."""
        batch = collate_variable_length(sample_batch)

        batch_size = len(sample_batch)
        assert batch["event_lengths"].shape[0] == batch_size
        assert batch["event_ids"].shape[0] == batch_size
        assert batch["targets"].shape[0] == batch_size

    def test_collate_event_lengths_sum(self, sample_batch):
        """Test that event lengths sum to total pulses."""
        batch = collate_variable_length(sample_batch)

        total_pulses = batch["pulse_features"].shape[0]
        sum_lengths = batch["event_lengths"].sum().item()

        assert total_pulses == sum_lengths


class TestDataLoader:
    """Test get_dataloader function."""

    def test_dataloader_creation(self):
        """Test dataloader can be created."""
        dataloader = get_dataloader(split="train", batch_size=4, max_events=20)
        assert dataloader is not None

    def test_dataloader_iteration(self):
        """Test can iterate over dataloader."""
        dataloader = get_dataloader(split="train", batch_size=4, max_events=20, shuffle=False)

        batch = next(iter(dataloader))
        assert isinstance(batch, dict)
        assert "pulse_features" in batch

    def test_dataloader_batch_size(self):
        """Test dataloader respects batch size."""
        batch_size = 8
        max_events = 100
        dataloader = get_dataloader(
            split="train",
            batch_size=batch_size,
            max_events=max_events,
            shuffle=False
        )

        batch = next(iter(dataloader))

        # First batch should have batch_size events
        assert batch["event_ids"].shape[0] == batch_size
        assert batch["targets"].shape[0] == batch_size

    def test_dataloader_full_iteration(self):
        """Test dataloader can iterate through full dataset."""
        max_events = 50
        batch_size = 10
        dataloader = get_dataloader(
            split="train",
            batch_size=batch_size,
            max_events=max_events,
            shuffle=False
        )

        total_events = 0
        for batch in dataloader:
            total_events += batch["event_ids"].shape[0]

        assert total_events == max_events

    def test_dataloader_shuffle(self):
        """Test shuffle produces different order."""
        max_events = 100
        batch_size = 10

        # Create two dataloaders with different shuffle seeds
        dataloader1 = get_dataloader(
            split="train",
            batch_size=batch_size,
            max_events=max_events,
            shuffle=True
        )

        dataloader2 = get_dataloader(
            split="train",
            batch_size=batch_size,
            max_events=max_events,
            shuffle=False
        )

        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))

        # With high probability, shuffled and non-shuffled should differ
        # (may rarely fail due to random chance)
        assert not torch.equal(batch1["event_ids"], batch2["event_ids"])

"""
Unit tests for collator functions, particularly comparing
legacy vs vectorized DOM grouping implementations.
"""

import pytest
import torch
from iceaggr.data import (
    IceCubeDataset,
    collate_with_dom_grouping,
    collate_with_dom_grouping_legacy,
)


class TestCollatorEquivalence:
    """Test that vectorized collator produces equivalent output to legacy."""

    @pytest.fixture
    def sample_events(self):
        """Load sample events for testing."""
        dataset = IceCubeDataset(split="train", max_events=100)
        return [dataset[i] for i in range(32)]

    @pytest.fixture
    def small_batch(self):
        """Create a small batch for quick tests."""
        dataset = IceCubeDataset(split="train", max_events=20)
        return [dataset[i] for i in range(5)]

    @pytest.fixture
    def single_event_batch(self):
        """Single event batch for edge case testing."""
        dataset = IceCubeDataset(split="train", max_events=10)
        return [dataset[0]]

    def test_batch_size_match(self, small_batch):
        """Test batch_size field matches."""
        legacy_result = collate_with_dom_grouping_legacy(small_batch)
        vectorized_result = collate_with_dom_grouping(small_batch)

        assert legacy_result['batch_size'] == vectorized_result['batch_size']

    def test_total_doms_match(self, small_batch):
        """Test total_doms field matches."""
        legacy_result = collate_with_dom_grouping_legacy(small_batch)
        vectorized_result = collate_with_dom_grouping(small_batch)

        assert legacy_result['total_doms'] == vectorized_result['total_doms']

    def test_event_ids_match(self, small_batch):
        """Test event_ids are identical."""
        legacy_result = collate_with_dom_grouping_legacy(small_batch)
        vectorized_result = collate_with_dom_grouping(small_batch)

        assert torch.equal(legacy_result['event_ids'], vectorized_result['event_ids'])

    def test_targets_match(self, small_batch):
        """Test targets are identical."""
        legacy_result = collate_with_dom_grouping_legacy(small_batch)
        vectorized_result = collate_with_dom_grouping(small_batch)

        assert torch.equal(legacy_result['targets'], vectorized_result['targets'])

    def test_event_dom_counts_match(self, small_batch):
        """Test event_dom_counts match."""
        legacy_result = collate_with_dom_grouping_legacy(small_batch)
        vectorized_result = collate_with_dom_grouping(small_batch)

        assert torch.equal(
            legacy_result['event_dom_counts'],
            vectorized_result['event_dom_counts']
        )

    def test_dom_pulse_counts_match(self, small_batch):
        """Test dom_pulse_counts match (may be in different order)."""
        legacy_result = collate_with_dom_grouping_legacy(small_batch)
        vectorized_result = collate_with_dom_grouping(small_batch)

        # Same total DOMs
        assert legacy_result['dom_pulse_counts'].shape == vectorized_result['dom_pulse_counts'].shape

        # Same sum (total pulses)
        assert legacy_result['dom_pulse_counts'].sum() == vectorized_result['dom_pulse_counts'].sum()

        # When sorted, should match
        assert torch.equal(
            legacy_result['dom_pulse_counts'].sort()[0],
            vectorized_result['dom_pulse_counts'].sort()[0]
        )

    def test_total_pulses_match(self, small_batch):
        """Test total number of pulses matches."""
        legacy_result = collate_with_dom_grouping_legacy(small_batch)
        vectorized_result = collate_with_dom_grouping(small_batch)

        assert legacy_result['pulse_features'].shape[0] == vectorized_result['pulse_features'].shape[0]

    def test_pulse_features_content_match(self, small_batch):
        """Test pulse features contain the same data (possibly reordered)."""
        legacy_result = collate_with_dom_grouping_legacy(small_batch)
        vectorized_result = collate_with_dom_grouping(small_batch)

        # Sort by all columns to compare content regardless of order
        legacy_sorted, _ = legacy_result['pulse_features'].sort(dim=0)
        vectorized_sorted, _ = vectorized_result['pulse_features'].sort(dim=0)

        # Use allclose for floating point comparison
        assert torch.allclose(legacy_sorted, vectorized_sorted, atol=1e-6)

    def test_dom_to_event_mapping_consistent(self, small_batch):
        """Test dom_to_event_idx is consistent with pulse data."""
        result = collate_with_dom_grouping(small_batch)

        # Each DOM should map to exactly one event
        for dom_idx in range(result['total_doms']):
            dom_mask = result['pulse_to_dom_idx'] == dom_idx
            if dom_mask.any():
                # Check event_idx from pulse features matches dom_to_event_idx
                event_idx = result['dom_to_event_idx'][dom_idx]
                assert event_idx >= 0 and event_idx < result['batch_size']

    def test_pulse_idx_in_dom_valid(self, small_batch):
        """Test pulse_idx_in_dom starts at 0 for each DOM."""
        result = collate_with_dom_grouping(small_batch)

        for dom_idx in range(result['total_doms']):
            dom_mask = result['pulse_to_dom_idx'] == dom_idx
            if dom_mask.any():
                indices = result['pulse_idx_in_dom'][dom_mask]
                # Should be 0, 1, 2, ..., n-1
                expected = torch.arange(len(indices), dtype=torch.long)
                assert torch.equal(indices, expected), f"DOM {dom_idx} has invalid pulse indices"

    def test_n_pulses_in_dom_consistent(self, small_batch):
        """Test n_pulses_in_dom matches dom_pulse_counts."""
        result = collate_with_dom_grouping(small_batch)

        for dom_idx in range(result['total_doms']):
            dom_mask = result['pulse_to_dom_idx'] == dom_idx
            if dom_mask.any():
                # n_pulses_in_dom should be constant and equal to dom_pulse_counts[dom_idx]
                broadcast_counts = result['n_pulses_in_dom'][dom_mask]
                expected_count = result['dom_pulse_counts'][dom_idx]
                assert (broadcast_counts == expected_count).all()

    def test_single_event_batch(self, single_event_batch):
        """Test collator works with single event."""
        legacy_result = collate_with_dom_grouping_legacy(single_event_batch)
        vectorized_result = collate_with_dom_grouping(single_event_batch)

        assert legacy_result['batch_size'] == 1
        assert vectorized_result['batch_size'] == 1
        assert legacy_result['total_doms'] == vectorized_result['total_doms']

    def test_large_batch(self, sample_events):
        """Test collator works with larger batch (32 events)."""
        legacy_result = collate_with_dom_grouping_legacy(sample_events)
        vectorized_result = collate_with_dom_grouping(sample_events)

        assert legacy_result['total_doms'] == vectorized_result['total_doms']
        assert legacy_result['pulse_features'].shape == vectorized_result['pulse_features'].shape


class TestCollatorEdgeCases:
    """Test edge cases for the vectorized collator."""

    def test_dom_ids_valid_range(self):
        """Test all DOM IDs are in valid range [0, 5159]."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(10)]

        result = collate_with_dom_grouping(batch)

        assert (result['dom_ids'] >= 0).all()
        assert (result['dom_ids'] < 5160).all()

    def test_dom_to_event_idx_valid_range(self):
        """Test all event indices are valid."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(10)]

        result = collate_with_dom_grouping(batch)

        assert (result['dom_to_event_idx'] >= 0).all()
        assert (result['dom_to_event_idx'] < 10).all()

    def test_pulse_to_dom_idx_valid_range(self):
        """Test all DOM indices are valid."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(10)]

        result = collate_with_dom_grouping(batch)

        assert (result['pulse_to_dom_idx'] >= 0).all()
        assert (result['pulse_to_dom_idx'] < result['total_doms']).all()

    def test_event_dom_counts_sum(self):
        """Test event_dom_counts sums to total_doms."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(10)]

        result = collate_with_dom_grouping(batch)

        assert result['event_dom_counts'].sum().item() == result['total_doms']

    def test_dom_pulse_counts_sum(self):
        """Test dom_pulse_counts sums to total pulses."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(10)]

        result = collate_with_dom_grouping(batch)

        total_pulses = result['pulse_features'].shape[0]
        assert result['dom_pulse_counts'].sum().item() == total_pulses

    def test_reproducibility(self):
        """Test collator produces identical results on same input."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(10)]

        result1 = collate_with_dom_grouping(batch)
        result2 = collate_with_dom_grouping(batch)

        for key in result1:
            if isinstance(result1[key], torch.Tensor):
                assert torch.equal(result1[key], result2[key]), f"Mismatch in {key}"
            else:
                assert result1[key] == result2[key], f"Mismatch in {key}"


class TestDOMOrderPreservation:
    """Test that DOMs are ordered correctly (by event, then by sensor_id)."""

    def test_doms_sorted_by_event(self):
        """Test DOMs are grouped by event."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(10)]

        result = collate_with_dom_grouping(batch)

        # dom_to_event_idx should be non-decreasing (all DOMs from event 0, then event 1, etc.)
        event_indices = result['dom_to_event_idx']
        assert (event_indices[1:] >= event_indices[:-1]).all(), \
            "DOMs should be ordered by event index"

    def test_doms_sorted_by_sensor_within_event(self):
        """Test DOMs are sorted by sensor_id within each event."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(5)]

        result = collate_with_dom_grouping(batch)

        # Within each event, sensor IDs should be sorted
        for event_idx in range(5):
            event_mask = result['dom_to_event_idx'] == event_idx
            event_sensors = result['dom_ids'][event_mask]
            assert (event_sensors[1:] >= event_sensors[:-1]).all(), \
                f"Sensors in event {event_idx} should be sorted"

    def test_pulses_grouped_by_dom(self):
        """Test pulses are contiguous within each DOM."""
        dataset = IceCubeDataset(split="train", max_events=50)
        batch = [dataset[i] for i in range(5)]

        result = collate_with_dom_grouping(batch)

        # pulse_to_dom_idx should have contiguous runs
        dom_indices = result['pulse_to_dom_idx']

        # Find where DOM index changes
        changes = torch.where(dom_indices[1:] != dom_indices[:-1])[0]

        # DOM indices should only increase (never go back to a previous DOM)
        if len(changes) > 0:
            dom_at_changes = dom_indices[changes]
            assert (dom_at_changes[1:] > dom_at_changes[:-1]).all(), \
                "DOM indices should be monotonically increasing"

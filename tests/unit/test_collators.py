"""
Unit tests for collator functions, particularly comparing
legacy vs vectorized DOM grouping implementations.
"""

import pytest
import torch
from iceaggr.data import (
    GeometryLoader,
    IceCubeDataset,
    collate_with_dom_grouping,
    collate_with_dom_grouping_legacy,
    make_collate_flat,
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


class TestFlatCollatorTimeOrdering:
    """Test the opt-in order_doms_by_time flag of make_collate_flat.

    Uses a small synthetic batch with known per-DOM first-pulse times so the
    expected token ordering is unambiguous.
    """

    # sensor_id -> (earliest pulse time). Deliberately chosen so that ascending
    # time does NOT match ascending sensor_id, otherwise the two orderings would
    # coincide and the test would not distinguish them.
    EVENT0 = {  # sensor_id: earliest_time
        10: 300.0,
        11: 100.0,
        12: 200.0,
    }
    EVENT1 = {
        20: 50.0,
        21: 400.0,
        22: 150.0,
    }

    @pytest.fixture
    def geometry(self, tmp_path):
        """Tiny synthetic geometry with one distinct position per sensor_id.

        Each DOM gets a unique x so a token row can be mapped back to its
        sensor_id (and therefore its known earliest time) from the geometry
        columns of the flat DOM vector.
        """
        lines = ["sensor_id,x,y,z"]
        for sid in [10, 11, 12, 20, 21, 22]:
            # x = sensor_id (distinct, normalized small), y/z fixed
            lines.append(f"{sid},{sid * 0.001},0.0,0.0")
        csv_path = tmp_path / "geom.csv"
        csv_path.write_text("\n".join(lines) + "\n")
        return GeometryLoader(str(csv_path))

    @staticmethod
    def _make_event(event_id, dom_times):
        """Build one event dict. Each DOM gets two pulses; the earlier one sets
        dom_min_time. Pulses are appended in a shuffled order to exercise the
        within-DOM sort path."""
        rows = []
        for sid, t0 in dom_times.items():
            # two pulses for the DOM; second is later so min_time == t0
            rows.append([t0 + 5.0, 1.0, float(sid), 0.0])  # later pulse first
            rows.append([t0, 1.0, float(sid), 0.0])         # earliest pulse
        pulse_features = torch.tensor(rows, dtype=torch.float32)
        return {
            "pulse_features": pulse_features,
            "event_id": torch.tensor(event_id, dtype=torch.long),
            "target": torch.tensor([0.1, 0.2], dtype=torch.float32),
        }

    @pytest.fixture
    def batch(self):
        return [self._make_event(0, self.EVENT0), self._make_event(1, self.EVENT1)]

    def test_off_is_byte_identical(self, geometry, batch):
        """order_doms_by_time=False must reproduce the historical (default-arg)
        output exactly, byte-for-byte."""
        default_fn = make_collate_flat(geometry, max_pulses_per_dom=4, max_doms=8)
        explicit_off_fn = make_collate_flat(
            geometry, max_pulses_per_dom=4, max_doms=8, order_doms_by_time=False
        )
        out_default = default_fn(batch)
        out_off = explicit_off_fn(batch)

        for key in out_default:
            if isinstance(out_default[key], torch.Tensor):
                assert torch.equal(out_default[key], out_off[key]), \
                    f"order_doms_by_time=False changed '{key}'"
            else:
                assert out_default[key] == out_off[key], f"mismatch in '{key}'"

    def _sensor_x_lookup(self, geometry):
        """Map normalized x-coordinate back to sensor_id for row identification."""
        return {sid: geometry[torch.tensor([sid])][0, 0].item()
                for sid in [10, 11, 12, 20, 21, 22]}

    def test_on_reorders_by_min_time(self, geometry, batch):
        """order_doms_by_time=True keeps the same multiset of DOM rows per event
        but orders them so dom_min_time is non-decreasing along the sequence."""
        off_fn = make_collate_flat(geometry, max_pulses_per_dom=4, max_doms=8)
        on_fn = make_collate_flat(
            geometry, max_pulses_per_dom=4, max_doms=8, order_doms_by_time=True
        )
        out_off = off_fn(batch)
        out_on = on_fn(batch)

        # Same kept DOMs per event (mask unchanged).
        assert torch.equal(out_off["padding_mask"], out_on["padding_mask"])

        x_to_sid = self._sensor_x_lookup(geometry)
        times = {**self.EVENT0, **self.EVENT1}

        def sid_of_row(row):
            x = row[0].item()
            best = min(x_to_sid, key=lambda s: abs(x_to_sid[s] - x))
            return best

        for ev, dom_times in enumerate([self.EVENT0, self.EVENT1]):
            mask = out_on["padding_mask"][ev]
            on_rows = out_on["dom_vectors"][ev][mask]
            off_rows = out_off["dom_vectors"][ev][mask]

            # (a) same multiset of token rows (sort rows lexicographically)
            on_sorted = on_rows[on_rows[:, 0].argsort(stable=True)]
            off_sorted = off_rows[off_rows[:, 0].argsort(stable=True)]
            assert torch.allclose(on_sorted, off_sorted), \
                f"event {ev}: time-ordering changed the set of DOM rows"

            # (b) along the time-ordered sequence, dom_min_time is non-decreasing
            seq_times = [times[sid_of_row(r)] for r in on_rows]
            assert seq_times == sorted(seq_times), \
                f"event {ev}: tokens not in non-decreasing min-time order: {seq_times}"

            # sanity: this batch actually differs from sensor order
            off_times = [times[sid_of_row(r)] for r in off_rows]
            assert off_times != seq_times, \
                f"event {ev}: test batch fails to distinguish the two orderings"


class TestFlatCollatorPadAssembly:
    """Guard the pad-assembly fast path in make_collate_flat.

    After subsampling every event holds <= max_doms DOMs, so the internal
    ``valid = dom_idx_in_event < max_doms`` mask is always all-True and the
    collator scatters DOM vectors straight into the padded tensor (skipping a
    full copy). These tests lock in the resulting invariants for both the
    no-subsample and subsample paths, so a future change to that fast path can't
    silently corrupt the output.
    """

    @pytest.fixture
    def geometry(self, tmp_path):
        lines = ["sensor_id,x,y,z"]
        for sid in range(40):
            lines.append(f"{sid},{sid * 0.001},{sid * 0.002},{sid * 0.003}")
        csv_path = tmp_path / "geom.csv"
        csv_path.write_text("\n".join(lines) + "\n")
        return GeometryLoader(str(csv_path))

    @staticmethod
    def _event(event_id, n_doms, base_time=0.0):
        """One event with n_doms distinct DOMs (sensor_ids 0..n_doms-1), one pulse
        each; earlier sensor_ids get earlier times so the earliest-time subsample
        keeps the low sensor_ids."""
        rows = [[base_time + sid, 1.0, float(sid), 0.0] for sid in range(n_doms)]
        return {
            "pulse_features": torch.tensor(rows, dtype=torch.float32),
            "event_id": torch.tensor(event_id, dtype=torch.long),
            "target": torch.tensor([0.1, 0.2], dtype=torch.float32),
        }

    def _check(self, out, expected_counts, max_doms, input_dim):
        mask = out["padding_mask"]
        vecs = out["dom_vectors"]
        assert vecs.shape == (len(expected_counts), max_doms, input_dim)
        for ev, n in enumerate(expected_counts):
            kept = min(n, max_doms)
            # exactly `kept` valid tokens, contiguous from position 0
            assert int(mask[ev].sum()) == kept
            assert mask[ev, :kept].all()
            assert not mask[ev, kept:].any()
            # padded (masked-out) rows are exactly zero
            assert torch.equal(vecs[ev][~mask[ev]], torch.zeros(max_doms - kept, input_dim))

    def test_no_subsample(self, geometry):
        max_doms = 16
        K = 4
        batch = [self._event(0, 5), self._event(1, 12), self._event(2, 1)]
        collate = make_collate_flat(geometry, max_pulses_per_dom=K, max_doms=max_doms)
        out = collate(batch)
        self._check(out, [5, 12, 1], max_doms, 4 + 3 * K)

    def test_with_subsample(self, geometry):
        max_doms = 8
        K = 4
        # event 1 has 20 DOMs > max_doms -> must subsample to the 8 earliest
        batch = [self._event(0, 5), self._event(1, 20), self._event(2, 8)]
        collate = make_collate_flat(geometry, max_pulses_per_dom=K, max_doms=max_doms)
        out = collate(batch)
        self._check(out, [5, 20, 8], max_doms, 4 + 3 * K)

    def test_all_events_oversized(self, geometry):
        """Every event exceeds max_doms: the fast path must still hold."""
        max_doms = 4
        K = 2
        batch = [self._event(i, 30, base_time=i * 100.0) for i in range(3)]
        collate = make_collate_flat(geometry, max_pulses_per_dom=K, max_doms=max_doms)
        out = collate(batch)
        self._check(out, [30, 30, 30], max_doms, 4 + 3 * K)

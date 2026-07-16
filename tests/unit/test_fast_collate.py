"""Tests for the opt-in `fast_collate` path (bf16 output + vectorized
deterministic subsample) of the flat / npe15 / hybrid collators.

Two guarantees:
  1. `fast_collate=False` (the default) is byte-identical to the historical path.
  2. `fast_collate=True` keeps the same number of DOMs per event, emits bfloat16,
     and subsamples by earliest hit-time with a *deterministic* lowest-sensor_id
     tie-break (`earliest_dom_keep_mask`).
"""

import pytest
import torch

from iceaggr.data import GeometryLoader, make_collate_flat
from iceaggr.data.collators import earliest_dom_keep_mask


class TestEarliestKeepMask:
    """Unit-test the vectorized subsample selection directly."""

    def test_no_subsample_all_true(self):
        # two events, 3 and 2 DOMs, cap 5 -> nothing dropped
        dom_min_time = torch.tensor([10., 20., 30., 5., 15.])
        dom_event_idx = torch.tensor([0, 0, 0, 1, 1])
        counts = torch.tensor([3, 2])
        keep = earliest_dom_keep_mask(dom_min_time, dom_event_idx, counts, max_doms=5, batch_size=2)
        assert keep.all()

    def test_keeps_earliest(self):
        # one event, 4 DOMs at times [30,10,40,20], cap 2 -> keep the two earliest
        # (times 10 and 20 -> positions 1 and 3). Mask is over positions, ascending.
        dom_min_time = torch.tensor([30., 10., 40., 20.])
        dom_event_idx = torch.tensor([0, 0, 0, 0])
        counts = torch.tensor([4])
        keep = earliest_dom_keep_mask(dom_min_time, dom_event_idx, counts, max_doms=2, batch_size=1)
        assert keep.tolist() == [False, True, False, True]

    def test_tie_break_lowest_position(self):
        # DOMs 0,1,2 all at time 10 (a 3-way tie), DOM 3 at time 99. cap 2 -> keep
        # the two earliest; among the tied, the deterministic rule keeps the two
        # LOWEST positions (== lowest sensor_id in the sorted DOM array): 0 and 1.
        dom_min_time = torch.tensor([10., 10., 10., 99.])
        dom_event_idx = torch.tensor([0, 0, 0, 0])
        counts = torch.tensor([4])
        keep = earliest_dom_keep_mask(dom_min_time, dom_event_idx, counts, max_doms=2, batch_size=1)
        assert keep.tolist() == [True, True, False, False]


class TestFastCollateFlat:
    @pytest.fixture
    def geometry(self, tmp_path):
        lines = ["sensor_id,x,y,z"]
        for sid in range(40):
            lines.append(f"{sid},{sid * 0.001},{sid * 0.002},{sid * 0.003}")
        p = tmp_path / "geom.csv"
        p.write_text("\n".join(lines) + "\n")
        return GeometryLoader(str(p))

    @staticmethod
    def _event(event_id, n_doms):
        rows = [[float(sid), 1.0, float(sid), 0.0] for sid in range(n_doms)]
        return {
            "pulse_features": torch.tensor(rows, dtype=torch.float32),
            "event_id": torch.tensor(event_id, dtype=torch.long),
            "target": torch.tensor([0.1, 0.2], dtype=torch.float32),
        }

    def test_default_byte_identical(self, geometry):
        batch = [self._event(0, 5), self._event(1, 20), self._event(2, 8)]
        default = make_collate_flat(geometry, max_pulses_per_dom=4, max_doms=8)
        off = make_collate_flat(geometry, max_pulses_per_dom=4, max_doms=8, fast_collate=False)
        a, b = default(batch), off(batch)
        assert torch.equal(a["dom_vectors"], b["dom_vectors"])
        assert torch.equal(a["padding_mask"], b["padding_mask"])

    def test_fast_dtype_and_counts(self, geometry):
        batch = [self._event(0, 5), self._event(1, 20), self._event(2, 8)]
        off = make_collate_flat(geometry, max_pulses_per_dom=4, max_doms=8)
        on = make_collate_flat(geometry, max_pulses_per_dom=4, max_doms=8, fast_collate=True)
        a, b = off(batch), on(batch)
        assert b["dom_vectors"].dtype == torch.bfloat16
        # same number of kept DOMs per event
        assert torch.equal(a["padding_mask"].sum(1), b["padding_mask"].sum(1))
        # event 1 (20 DOMs) subsampled to 8; events 0,2 unchanged
        assert b["padding_mask"].sum(1).tolist() == [5, 8, 8]

    def test_fast_values_match_where_no_ties(self, geometry):
        # distinct sensor times -> no min_time ties -> fast keeps the same DOM set
        # as default; values differ only by bf16 rounding (small).
        batch = [self._event(0, 20)]  # 20 DOMs, distinct times, cap 8
        off = make_collate_flat(geometry, max_pulses_per_dom=4, max_doms=8)
        on = make_collate_flat(geometry, max_pulses_per_dom=4, max_doms=8, fast_collate=True)
        a, b = off(batch), on(batch)
        m = a["padding_mask"]
        assert torch.equal(m, b["padding_mask"])
        diff = (a["dom_vectors"][m] - b["dom_vectors"].float()[m]).abs().max()
        assert float(diff) < 0.05  # bf16 rounding only

"""
Unit tests for the hybrid raw+aggregate DOM tokenization (collators_hybrid).

Coverage:
  * layout correctness — the base / aggregate / raw slots of the 256-dim token
    are byte-for-byte the corresponding slices of make_collate_flat (K=80) and
    make_collate_npe15 on the same hand-built batch (proves the reuse is exact
    and the slots land where the docstring says);
  * context_only layout (base + 6 event-context + 82 raw triplets = 256);
  * include_event_context flag behaviour in both modes;
  * padding + earliest-first-pulse subsampling;
  * both input_mode 'none' and 'linear' build and run on the hybrid tokens.
"""

import pytest
import torch

from iceaggr.data import (
    GeometryLoader,
    HYBRID_WIDTH,
    K_RAW_FULL,
    K_RAW_CONTEXT,
    make_collate_flat,
    make_collate_npe15,
    make_collate_hybrid,
)
from iceaggr.models import FlatTransformerV2

# sensor_id -> (x, y, z) raw positions for the synthetic detector
POSITIONS = {
    10: (10.0, 0.0, 0.0),
    12: (0.0, 12.0, 0.0),
    20: (20.0, 0.0, 0.0),
    21: (0.0, 0.0, 21.0),
    22: (0.0, 22.0, 0.0),
    30: (5.0, 5.0, 5.0),
}

# event_id -> {sensor_id: [(time, charge), ...]}
EVENT0 = {
    10: [(100.0, 1.0), (250.0, 2.0), (700.0, 1.0)],
    12: [(300.0, 5.0)],
}
EVENT1 = {
    20: [(1000.0, 2.0), (1050.0, 2.0)],
    21: [(2000.0, 1.0)],
    22: [(1500.0, 1.0), (1600.0, 1.0), (3000.0, 1.0)],
}

# the 12 npe columns the 'full' hybrid keeps (drops absolute x,y,z at 9,10,11)
NPE_KEEP_FULL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]


def _make_event(event_id, dom_pulses, aux=0.0):
    rows = []
    for sid, pulses in dom_pulses.items():
        for t, q in pulses:
            rows.append([t, q, float(sid), aux])
    return {
        "pulse_features": torch.tensor(rows, dtype=torch.float32),
        "event_id": torch.tensor(event_id, dtype=torch.long),
        "target": torch.tensor([0.1 * event_id, 0.2], dtype=torch.float32),
    }


@pytest.fixture
def geometry(tmp_path):
    lines = ["sensor_id,x,y,z"]
    for sid in sorted(POSITIONS):
        x, y, z = POSITIONS[sid]
        lines.append(f"{sid},{x},{y},{z}")
    p = tmp_path / "geom.csv"
    p.write_text("\n".join(lines) + "\n")
    return GeometryLoader(str(p))


@pytest.fixture
def batch():
    # non-zero aux so the aux-0.5 raw normalization is exercised
    return [_make_event(0, EVENT0, aux=1.0), _make_event(1, EVENT1, aux=1.0)]


class TestFullLayout:
    """The 'full' token = base(4) | npe-minus-xyz(12) | raw K=80 triplets(240)."""

    def test_width_dtype_finite(self, geometry, batch):
        out = make_collate_hybrid(geometry, max_doms=8, mode="full")(batch)
        assert out["dom_vectors"].shape == (2, 8, HYBRID_WIDTH)
        assert out["dom_vectors"].dtype == torch.float32
        assert torch.isfinite(out["dom_vectors"]).all()
        assert out["padding_mask"][0].sum().item() == 2   # event0 -> 2 DOMs
        assert out["padding_mask"][1].sum().item() == 3   # event1 -> 3 DOMs

    def test_slots_match_flat_and_npe(self, geometry, batch):
        """Base + raw slots == make_collate_flat(K=80); aggregate slot == npe15."""
        hyb = make_collate_hybrid(geometry, max_doms=8, mode="full")(batch)["dom_vectors"]
        flat = make_collate_flat(geometry, max_pulses_per_dom=K_RAW_FULL, max_doms=8)(batch)["dom_vectors"]
        npe = make_collate_npe15(geometry, max_doms=8)(batch)["dom_vectors"]

        # flat token = [x, y, z, n_tilde, 80*(t,q,a)] -> width 4 + 240 = 244
        assert flat.shape[-1] == 4 + 3 * K_RAW_FULL

        # base [0:4] == flat base [0:4]
        assert torch.allclose(hyb[..., 0:4], flat[..., 0:4], atol=1e-5)
        # raw block [16:256] == flat raw [4:244]
        assert torch.allclose(hyb[..., 16:256], flat[..., 4:244], atol=1e-6)
        # aggregate block [4:16] == npe15 minus absolute xyz
        assert torch.allclose(hyb[..., 4:16], npe[..., NPE_KEEP_FULL], atol=1e-5)

    def test_masks_match_across_tokenizations(self, geometry, batch):
        hyb = make_collate_hybrid(geometry, max_doms=8, mode="full")(batch)
        flat = make_collate_flat(geometry, max_pulses_per_dom=K_RAW_FULL, max_doms=8)(batch)
        npe = make_collate_npe15(geometry, max_doms=8)(batch)
        assert torch.equal(hyb["padding_mask"], flat["padding_mask"])
        assert torch.equal(hyb["padding_mask"], npe["padding_mask"])
        assert torch.equal(hyb["event_ids"], torch.tensor([0, 1]))

    def test_padded_rows_zero(self, geometry, batch):
        out = make_collate_hybrid(geometry, max_doms=8, mode="full")(batch)["dom_vectors"]
        # event0 has 2 valid DOMs -> rows 2.. are padding
        assert torch.equal(out[0, 2:], torch.zeros(6, HYBRID_WIDTH))


class TestContextOnlyLayout:
    """The 'context_only' token = base(4) | context(6) | raw K=82 triplets(246)."""

    def test_width_and_raw_slot(self, geometry, batch):
        hyb = make_collate_hybrid(geometry, max_doms=8, mode="context_only")(batch)["dom_vectors"]
        flat = make_collate_flat(geometry, max_pulses_per_dom=K_RAW_CONTEXT, max_doms=8)(batch)["dom_vectors"]
        assert hyb.shape[-1] == HYBRID_WIDTH
        # base [0:4] and raw block [10:256] mirror flat (K=82)
        assert torch.allclose(hyb[..., 0:4], flat[..., 0:4], atol=1e-5)
        assert torch.allclose(hyb[..., 10:256], flat[..., 4:250], atol=1e-6)

    def test_context_block_equals_npe_context(self, geometry, batch):
        """context dims 0:5 are x_rel,y_rel,z_rel,dt_first,dt_last from npe15."""
        hyb = make_collate_hybrid(geometry, max_doms=8, mode="context_only")(batch)["dom_vectors"]
        npe = make_collate_npe15(geometry, max_doms=8)(batch)["dom_vectors"]
        mask = make_collate_hybrid(geometry, max_doms=8, mode="context_only")(batch)["padding_mask"]
        # x_rel,y_rel,z_rel == npe cols 12,13,14
        assert torch.allclose(hyb[..., 4:7][mask], npe[..., 12:15][mask], atol=1e-5)
        # dt_first,dt_last == npe cols 3,4 (event-median-centered t_first,t_last)
        assert torch.allclose(hyb[..., 7:9][mask], npe[..., 3:5][mask], atol=1e-5)


class TestEventContextFlag:
    def test_full_context_off_zeros_xrel_and_reanchors_time(self, geometry, batch):
        on = make_collate_hybrid(geometry, max_doms=8, mode="full",
                                 include_event_context=True)(batch)
        off = make_collate_hybrid(geometry, max_doms=8, mode="full",
                                  include_event_context=False)(batch)
        mask = on["padding_mask"]
        von, voff = on["dom_vectors"], off["dom_vectors"]
        # x_rel,y_rel,z_rel are the last 3 of the 12-dim aggregate block -> cols 13:16
        assert torch.equal(voff[..., 13:16][mask], torch.zeros_like(voff[..., 13:16][mask]))
        assert not torch.allclose(von[..., 13:16][mask], voff[..., 13:16][mask])
        # charge sums (cols 4:7) and raw block unchanged
        assert torch.allclose(von[..., 4:7], voff[..., 4:7], atol=1e-6)
        assert torch.allclose(von[..., 16:256], voff[..., 16:256], atol=1e-6)
        # time columns (t_first..t_mean at cols 7:12) are re-anchored -> shifted
        assert not torch.allclose(von[..., 7:12][mask], voff[..., 7:12][mask])

    def test_context_only_off_zeros_context_block(self, geometry, batch):
        off = make_collate_hybrid(geometry, max_doms=8, mode="context_only",
                                  include_event_context=False)(batch)
        mask = off["padding_mask"]
        ctx = off["dom_vectors"][..., 4:10]
        assert torch.equal(ctx[mask], torch.zeros_like(ctx[mask]))
        # raw block still present
        raw = off["dom_vectors"][..., 10:256]
        assert raw[mask].abs().sum() > 0


class TestSubsampling:
    def test_keeps_earliest_first_pulse(self, geometry):
        ev = _make_event(0, {
            10: [(100.0, 1.0)],
            12: [(300.0, 1.0)],
            30: [(50.0, 1.0)],
        })
        out = make_collate_hybrid(geometry, max_doms=2, mode="full")([ev])
        assert out["padding_mask"][0].sum().item() == 2
        rows = out["dom_vectors"][0][out["padding_mask"][0]]
        # base x is slot 0 (geometry raw): sensor 30 x=5, sensor 10 x=10; 12 (x=0) dropped
        xs = sorted(round(r[0].item(), 3) for r in rows)
        assert xs == [5.0, 10.0]

    def test_no_targets_when_absent(self, geometry, batch):
        stripped = [{k: v for k, v in e.items() if k != "target"} for e in batch]
        out = make_collate_hybrid(geometry, max_doms=8, mode="full")(stripped)
        assert "targets" not in out


class TestModelBuilds:
    """The 256-dim token must feed both input_mode none (identity) and linear."""

    def _tiny_config(self, input_mode):
        return {
            "version": "v2",
            "max_pulses_per_dom": 84,   # -> default input_dim = 4 + 3*84 = 256
            "d_model": 256,
            "max_doms": 8,
            "num_heads": 8,
            "num_layers": 2,
            "hidden_dim": 64,
            "head_hidden_dim": 32,
            "dropout": 0.0,
            "input_mode": input_mode,
        }

    @pytest.mark.parametrize("input_mode", ["none", "linear"])
    def test_forward_finite(self, geometry, batch, input_mode):
        out = make_collate_hybrid(geometry, max_doms=8, mode="full")(batch)
        model = FlatTransformerV2(self._tiny_config(input_mode)).eval()
        with torch.no_grad():
            pred = model(out["dom_vectors"], out["padding_mask"])
        direction = pred["direction"] if isinstance(pred, dict) else pred
        assert direction.shape == (2, 3)
        assert torch.isfinite(direction).all()

    def test_none_is_identity_no_projection(self, geometry):
        model = FlatTransformerV2(self._tiny_config("none"))
        # input_dim == d_model == 256 -> no padding, Identity projection
        assert model.input_dim == HYBRID_WIDTH
        assert model._pad_input == 0
        assert isinstance(model.input_proj, torch.nn.Identity)

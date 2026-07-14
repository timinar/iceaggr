"""
Unit tests for the NPE-15 summary-statistics tokenization (collators_npe).

Coverage:
  * feature correctness on a hand-built 2-event batch (known pulses -> known stats),
    cross-checked against an independent pure-numpy reference over all DOMs;
  * per-event normalization (charge log1p, median-t50 time shift, t_std scaling);
  * padding + earliest-first-pulse subsampling behaviour;
  * the reference percentile quirk (t_20 == t_50 == t_first) and the opt-in
    correct-percentile path.
"""

import numpy as np
import pytest
import torch

from iceaggr.data import (
    GeometryLoader,
    FEATURE_COLS_NPE,
    INPUT_DIM_NPE,
    compute_dom_features_npe,
    normalize_dom_features_npe,
    make_collate_npe15,
)

FI = {name: i for i, name in enumerate(FEATURE_COLS_NPE)}

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


def _make_event(event_id, dom_pulses):
    rows = []
    for sid, pulses in dom_pulses.items():
        for t, q in pulses:
            rows.append([t, q, float(sid), 0.0])
    return {
        "pulse_features": torch.tensor(rows, dtype=torch.float32),
        "event_id": torch.tensor(event_id, dtype=torch.long),
        "target": torch.tensor([0.1 * event_id, 0.2], dtype=torch.float32),
    }


def _concat(batch):
    feats = [e["pulse_features"] for e in batch]
    lengths = torch.tensor([f.shape[0] for f in feats])
    all_features = torch.cat(feats, dim=0)
    pulse_event_idx = torch.repeat_interleave(torch.arange(len(batch)), lengths)
    return all_features, pulse_event_idx


def _numpy_reference(batch, geom_pos):
    """Independent per-DOM feature computation (pure numpy loops)."""
    out = {}
    for ei, event in enumerate(batch):
        pf = event["pulse_features"].numpy().astype(np.float64)
        t, q, sid = pf[:, 0], pf[:, 1], pf[:, 2].astype(int)
        # event charge-weighted CoG over all pulses
        epos = geom_pos[sid]
        cog = (epos * q[:, None]).sum(0) / q.sum()
        for s in np.unique(sid):
            m = sid == s
            tt, qq = t[m], q[m]
            tmin = tt.min()
            ct = qq.sum()
            tmean = (qq * tt).sum() / ct
            tstd = np.sqrt(max((qq * tt * tt).sum() / ct - tmean**2, 0.0))
            p = geom_pos[s]
            out[(ei, int(s))] = {
                "c_total": ct,
                "c_500ns": qq[(tt - tmin) <= 500].sum(),
                "c_100ns": qq[(tt - tmin) <= 100].sum(),
                "t_first": tmin, "t_last": tt.max(),
                "t_20": tmin, "t_50": tmin,  # reference quirk
                "t_mean": tmean, "t_std": tstd,
                "x": p[0], "y": p[1], "z": p[2],
                "x_rel": p[0] - cog[0], "y_rel": p[1] - cog[1], "z_rel": p[2] - cog[2],
            }
    return out


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
def geom_pos():
    arr = np.zeros((max(POSITIONS) + 1, 3))
    for sid, xyz in POSITIONS.items():
        arr[sid] = xyz
    return arr


@pytest.fixture
def batch():
    return [_make_event(0, EVENT0), _make_event(1, EVENT1)]


class TestFeatureCorrectness:
    def test_hand_computed_single_dom(self, geometry, batch):
        """Explicit hand-computed stats for event0 / sensor 10."""
        all_features, pei = _concat(batch)
        feats, dev, dsid, _ = compute_dom_features_npe(all_features, pei, geometry, 2)
        # locate (event0, sensor10)
        row = ((dev == 0) & (dsid == 10)).nonzero().item()
        f = feats[row]
        # times [100,250,700] charges [1,2,1]
        assert f[FI["c_total"]].item() == pytest.approx(4.0)
        assert f[FI["c_500ns"]].item() == pytest.approx(3.0)   # t<=600: 100,250
        assert f[FI["c_100ns"]].item() == pytest.approx(1.0)   # t<=200: 100
        assert f[FI["t_first"]].item() == pytest.approx(100.0)
        assert f[FI["t_last"]].item() == pytest.approx(700.0)
        assert f[FI["t_mean"]].item() == pytest.approx(325.0)  # 1300/4
        assert f[FI["t_std"]].item() == pytest.approx(225.0)   # sqrt(50625)

    def test_matches_numpy_reference_all_doms(self, geometry, batch, geom_pos):
        all_features, pei = _concat(batch)
        feats, dev, dsid, _ = compute_dom_features_npe(all_features, pei, geometry, 2)
        ref = _numpy_reference(batch, geom_pos)
        assert feats.shape[0] == len(ref)
        for r in range(feats.shape[0]):
            key = (int(dev[r]), int(dsid[r]))
            exp = ref[key]
            for name, i in FI.items():
                assert feats[r, i].item() == pytest.approx(exp[name], abs=1e-6, rel=1e-9), \
                    f"{key} feature {name}"

    def test_percentile_quirk_default(self, geometry, batch):
        all_features, pei = _concat(batch)
        feats, _, _, _ = compute_dom_features_npe(all_features, pei, geometry, 2)
        assert torch.allclose(feats[:, FI["t_20"]], feats[:, FI["t_first"]])
        assert torch.allclose(feats[:, FI["t_50"]], feats[:, FI["t_first"]])

    def test_correct_percentiles_path(self, geometry, batch):
        all_features, pei = _concat(batch)
        feats, dev, dsid, _ = compute_dom_features_npe(
            all_features, pei, geometry, 2, correct_percentiles=True
        )
        row = ((dev == 0) & (dsid == 10)).nonzero().item()
        # event0/sensor10: cum charge [1,3,4]/4; median (0.5) interpolates
        # between t=100 (cum 1) and t=250 (cum 3): 100 + (2-1)/(3-1)*150 = 175
        assert feats[row, FI["t_50"]].item() == pytest.approx(175.0, abs=1e-4)
        # 20th percentile lands inside the first pulse's charge -> t_first
        assert feats[row, FI["t_20"]].item() == pytest.approx(100.0, abs=1e-4)
        # single-pulse DOMs still collapse to their only time
        row12 = ((dev == 0) & (dsid == 12)).nonzero().item()
        assert feats[row12, FI["t_50"]].item() == pytest.approx(300.0, abs=1e-4)


class TestNormalization:
    def test_charge_and_time_normalization(self, geometry, batch):
        all_features, pei = _concat(batch)
        feats, dev, dsid, _ = compute_dom_features_npe(all_features, pei, geometry, 2)
        raw = feats.clone()
        normalize_dom_features_npe(feats, dev, 2, normalize_positions=False)

        # charges: 0.2 * ln(1+Q)
        for i in range(3):
            assert torch.allclose(feats[:, i], 0.2 * torch.log1p(raw[:, i]))

        # event0 t_ref = median t_first over {100, 300} = 200
        row10 = ((dev == 0) & (dsid == 10)).nonzero().item()
        assert feats[row10, FI["t_first"]].item() == pytest.approx((100 - 200) / 1e4)
        assert feats[row10, FI["t_last"]].item() == pytest.approx((700 - 200) / 1e4)
        assert feats[row10, FI["t_mean"]].item() == pytest.approx((325 - 200) / 1e4)
        # t_std: scale only, no shift
        assert feats[row10, FI["t_std"]].item() == pytest.approx(225 / 1e4)

        # event1 t_ref = median of {1000, 1500, 2000} = 1500
        row20 = ((dev == 1) & (dsid == 20)).nonzero().item()
        assert feats[row20, FI["t_first"]].item() == pytest.approx((1000 - 1500) / 1e4)

    def test_normalize_positions_flag(self, geometry, batch):
        all_features, pei = _concat(batch)
        f_no = compute_dom_features_npe(all_features, pei, geometry, 2)[0]
        dev = compute_dom_features_npe(all_features, pei, geometry, 2)[1]
        f_yes = f_no.clone()
        normalize_dom_features_npe(f_no, dev, 2, normalize_positions=False)
        normalize_dom_features_npe(f_yes, dev, 2, normalize_positions=True)
        # positions differ by exactly /500; time/charge cols identical
        assert torch.allclose(f_yes[:, 9:15] * 500.0, f_no[:, 9:15], atol=1e-4)
        assert torch.allclose(f_yes[:, :9], f_no[:, :9])


class TestCollator:
    def test_shapes_and_mask(self, geometry, batch):
        collate = make_collate_npe15(geometry, max_doms=8)
        out = collate(batch)
        assert out["dom_vectors"].shape == (2, 8, INPUT_DIM_NPE)
        assert out["padding_mask"].shape == (2, 8)
        assert out["padding_mask"][0].sum().item() == 2   # event0 has 2 DOMs
        assert out["padding_mask"][1].sum().item() == 3   # event1 has 3 DOMs
        assert torch.equal(out["event_ids"], torch.tensor([0, 1]))
        assert out["targets"].shape == (2, 2)
        # padded rows are zero
        assert torch.equal(out["dom_vectors"][0, 2:], torch.zeros(6, INPUT_DIM_NPE))

    def test_no_targets_when_absent(self, geometry, batch):
        stripped = [{k: v for k, v in e.items() if k != "target"} for e in batch]
        out = make_collate_npe15(geometry, max_doms=8)(stripped)
        assert "targets" not in out

    def test_subsampling_keeps_earliest(self, geometry):
        # one event, 3 DOMs with first-pulse times 100/300/50; max_doms=2 keeps
        # the two earliest (sensors 30 @50 and 10 @100), drops sensor 12 @300.
        ev = _make_event(0, {
            10: [(100.0, 1.0)],
            12: [(300.0, 1.0)],
            30: [(50.0, 1.0)],
        })
        collate = make_collate_npe15(geometry, max_doms=2)
        out = collate([ev])
        assert out["padding_mask"][0].sum().item() == 2
        rows = out["dom_vectors"][0][out["padding_mask"][0]]
        # recover sensor by matching absolute x position (normalize_positions=False,
        # geometry raw) -> sensor 30 has x=5, sensor 10 has x=10, sensor 12 x=0.
        xs = sorted(round(r[FI["x"]].item(), 3) for r in rows)
        assert xs == [5.0, 10.0]   # sensors 30 and 10 kept; 12 (x=0) dropped

    def test_finite_and_dtype(self, geometry, batch):
        out = make_collate_npe15(geometry, max_doms=8)(batch)
        assert out["dom_vectors"].dtype == torch.float32
        assert torch.isfinite(out["dom_vectors"]).all()

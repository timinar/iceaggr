#!/usr/bin/env python3
"""
Validation gate for the NPE-15 tokenization (task #15, phase 1).

Computes our UN-normalized per-DOM features directly from the raw Kaggle pulse
parquets via the SAME shared code the collator uses
(``iceaggr.data.compute_dom_features_npe``) and compares them, joined on
(event_id, sensor_id), against Johann's precomputed store
``/groups/icecube/jniko/NaturalAggregationInarTroels/train_pmtfied_npe15``.

Reports per-feature max abs / max rel deviation and an overall PASS/FAIL, then
writes the table to validation.md. Uses RAW geometry (sensor_geometry.csv) and
float64 inputs so the comparison isolates feature-definition correctness (the
production collator normalizes and runs in float32; that is exercised by the unit
tests, not here).

Run:  uv run python delegation/humanstats_scaled/validate_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from iceaggr.data import FEATURE_COLS_NPE, compute_dom_features_npe  # noqa: E402


class Float64Geometry:
    """Float64 geometry lookup for the gate. The production GeometryLoader stores
    positions as float32 (~2e-5 m rounding at ~500 m), which would inflate the
    relative deviation of near-zero CoG-relative positions and mask the fact that
    the feature LOGIC is exact. Reading positions in float64 — matching the store's
    polars CSV read — isolates the feature computation from that storage choice.
    Both models actually consume float32 positions downstream, so the difference is
    physically irrelevant; this is purely to keep the gate's threshold meaningful."""

    def __init__(self, csv_path):
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
        n = int(data[:, 0].max()) + 1
        self._pos = torch.zeros(n, 3, dtype=torch.float64)
        self._pos[data[:, 0].astype(np.int64)] = torch.from_numpy(data[:, 1:4])

    def __getitem__(self, sensor_ids):
        return self._pos[sensor_ids.long()]

RAW_DIR = "/groups/pheno/inar/icecube_kaggle/train"
META_DIR = "/groups/pheno/inar/icecube_kaggle/train_meta"
STORE_DIR = "/groups/icecube/jniko/NaturalAggregationInarTroels/train_pmtfied_npe15"
GEO_RAW = "/groups/pheno/inar/icecube_kaggle/sensor_geometry.csv"

BATCHES = [651, 1]
EVENTS_PER_BATCH = 1500          # >= 2000 events total across the two batches
# Per-feature PASS: every element matches to <1e-3 relative OR <1e-6 absolute.
REL_TOL = 1e-3
ABS_FLOOR = 1e-6

SUM_FEATURES = {"c_total", "c_500ns", "c_100ns", "t_first", "t_last", "t_20", "t_50"}


def load_events(batch):
    meta = pq.read_table(f"{META_DIR}/batch_{batch}.parquet",
                         columns=["event_id", "first_pulse_index", "last_pulse_index"])
    ev = meta.column("event_id").to_numpy()
    fp = meta.column("first_pulse_index").to_numpy()
    lp = meta.column("last_pulse_index").to_numpy()
    n = min(EVENTS_PER_BATCH, len(ev))
    lo, hi = int(fp[0]), int(lp[n - 1]) + 1
    raw = pq.read_table(f"{RAW_DIR}/batch_{batch}.parquet",
                        columns=["time", "charge", "sensor_id"])
    t = raw.column("time").to_numpy()[lo:hi].astype(np.float64)
    q = raw.column("charge").to_numpy()[lo:hi].astype(np.float64)
    s = raw.column("sensor_id").to_numpy()[lo:hi].astype(np.float64)
    # build concatenated pulse features + per-pulse event index
    all_features = torch.from_numpy(np.stack([t, q, s, np.zeros_like(t)], axis=1))
    starts = (fp[:n] - lo).astype(np.int64)
    ends = (lp[:n] - lo + 1).astype(np.int64)
    lengths = ends - starts
    pulse_event_idx = torch.from_numpy(np.repeat(np.arange(n), lengths))
    return all_features, pulse_event_idx, ev[:n], n


def load_store_rows(batch, target_event_ids):
    tbl = pq.read_table(f"{STORE_DIR}/batch_{batch}_pmtfied.parquet",
                        columns=["event_id", "sensor_id", *FEATURE_COLS_NPE])
    e = tbl.column("event_id").to_numpy()
    mask = np.isin(e, target_event_ids)
    sub = tbl.filter(mask)
    cols = {c: sub.column(c).to_numpy() for c in sub.column_names}
    key_to_row = {(int(cols["event_id"][i]), int(cols["sensor_id"][i])): i
                  for i in range(len(cols["event_id"]))}
    return cols, key_to_row


def main():
    geom = Float64Geometry(GEO_RAW)
    per_feat_abs = {f: 0.0 for f in FEATURE_COLS_NPE}
    per_feat_rel = {f: 0.0 for f in FEATURE_COLS_NPE}
    n_dom_compared = 0
    n_events = 0
    n_unmatched = 0

    for batch in BATCHES:
        all_features, pei, ev_ids, n = load_events(batch)
        n_events += n
        feats, dom_event_idx, dom_sid, _ = compute_dom_features_npe(
            all_features, pei, geom, n, correct_percentiles=False
        )
        feats = feats.numpy()
        dev = dom_event_idx.numpy()
        dsid = dom_sid.numpy()
        cols, key_to_row = load_store_rows(batch, ev_ids)

        for r in range(feats.shape[0]):
            eid = int(ev_ids[dev[r]])
            key = (eid, int(dsid[r]))
            j = key_to_row.get(key)
            if j is None:
                n_unmatched += 1
                continue
            n_dom_compared += 1
            for fi, fname in enumerate(FEATURE_COLS_NPE):
                mine = float(feats[r, fi])
                theirs = float(cols[fname][j])
                a = abs(mine - theirs)
                rel = a / max(abs(theirs), ABS_FLOOR)
                if a > per_feat_abs[fname]:
                    per_feat_abs[fname] = a
                if rel > per_feat_rel[fname]:
                    per_feat_rel[fname] = rel

    def feature_pass(f):
        # element-level criterion collapsed to the per-feature maxima: pass if the
        # worst element is within rel tol, or its abs deviation is below the floor.
        return (per_feat_rel[f] < REL_TOL) or (per_feat_abs[f] < ABS_FLOOR)

    all_pass = all(feature_pass(f) for f in FEATURE_COLS_NPE)

    lines = []
    lines.append("# NPE-15 validation gate\n")
    lines.append(f"Batches {BATCHES}, {n_events} events, {n_dom_compared} DOM rows "
                 f"joined on (event_id, sensor_id). Unmatched DOM rows: {n_unmatched}.\n")
    lines.append("Our un-normalized features (float64, raw geometry) vs Johann's "
                 f"precomputed store. PASS = max rel < {REL_TOL} or max abs < {ABS_FLOOR}.\n")
    lines.append("| feature | max_abs | max_rel | PASS |")
    lines.append("|---|---|---|---|")
    for f in FEATURE_COLS_NPE:
        lines.append(f"| {f} | {per_feat_abs[f]:.3e} | {per_feat_rel[f]:.3e} | "
                     f"{'PASS' if feature_pass(f) else 'FAIL'} |")
    lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**\n")
    lines.append("Notes:")
    lines.append("- t_20 and t_50 are compared to the store's values, which equal "
                 "t_first for every DOM. The reference generator's polars "
                 "`map_elements` runs the charge-cumulative percentile per pulse "
                 "(not per DOM) and takes `.first()`, collapsing both quantiles to "
                 "the earliest pulse time. `compute_dom_features_npe` reproduces this "
                 "by default so the scaled run is a faithful scale-up of the same "
                 "encoding; pass `correct_percentiles=True` for genuine quantile times.")
    report = "\n".join(lines) + "\n"

    out = Path(__file__).resolve().parent / "validation.md"
    out.write_text(report)
    print(report)
    print(f"Wrote {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
NPE-15 summary-statistics DOM tokenization for the scaled encoding comparison.

This produces the same 15 hand-crafted per-DOM summary statistics as Johann's
"human summary statistics" baseline (reference generator
``scripts/pmtfy_batches_npe.py`` + adapter ``src/iceaggr/data/pmtfied_npe.py`` in
``/groups/icecube/jniko/NaturalAggregationInarTroels/iceaggr``), but computed
directly from the raw pulse batches in a single collate pass — no precomputed
per-DOM parquet store is required. This lets the N2-T2 pipeline train on the
human-features encoding using the exact same ``IceCubeDataset`` /
``BatchAwareSampler`` machinery as the learned encoding.

Feature order (``FEATURE_COLS_NPE``), statistics over ALL pulses of each DOM:
  charge   (3): c_total, c_500ns, c_100ns
  time     (6): t_first, t_last, t_20, t_50, t_mean, t_std
  position (3): x, y, z                    (DOM coordinates)
  position (3): x_rel, y_rel, z_rel         (relative to the charge-weighted
                                             event centre of gravity)

Definitions (verified byte-for-byte against Johann's precomputed store
``train_pmtfied_npe15/`` on 23,523 DOMs of batch 651 — all 13 non-percentile
features match to float64 precision):
  * c_500ns / c_100ns : charge summed over pulses with (t - t_first_dom) <= 500 / 100 ns.
  * t_mean / t_std    : CHARGE-WEIGHTED mean and std of pulse time.
  * x_rel/y_rel/z_rel : DOM position minus the event centre of gravity, where the
                        CoG is the charge-weighted mean pulse position over the event.

REFERENCE QUIRK — t_20 and t_50 (see ``correct_percentiles`` below):
  The reference *intends* t_20/t_50 to be charge-cumulative quantile times, but
  its polars ``map_elements`` runs the percentile per-pulse instead of per-DOM and
  then takes ``.first()``, so the stored value collapses to the DOM's earliest
  pulse time. Empirically ``t_20 == t_50 == t_first`` for 100% of the store's DOMs
  (all 122,880 rows of batch 651, including all 29,667 multi-pulse DOMs). We
  reproduce this faithfully by default so that (a) the validation gate matches the
  store exactly and (b) the scaled run is a true scale-up of the same encoding.
  Pass ``correct_percentiles=True`` to instead emit genuine charge-cumulative
  20%/50% quantile times (a non-degenerate, *different* encoding — do not mix with
  runs meant to reproduce the reference).

Positions & normalization:
  The reference stores RAW-meter positions and divides by 500 at load time. Our
  ``GeometryLoader`` is normally built on ``sensor_geometry_normalized.csv`` (already
  /500), so positions — and CoG-relative positions, since /500 is linear and
  commutes with the charge-weighted mean — arrive already normalized and must NOT
  be divided again. ``make_collate_npe15`` therefore leaves positions as the
  geometry provides them (``normalize_positions=False`` by default). The per-event
  time normalization (shift by the event's median t_50, /1e4) and charge
  normalization (0.2*ln(1+Q)) match ``normalize_dom_array_npe`` exactly.

Subsampling: events with more than ``max_doms`` DOMs are truncated to the DOMs with
the EARLIEST first-pulse time (deliberately unified with ``make_collate_flat`` per
Johann's Q2 agreement). This differs from the reference adapter, which truncated by
top c_total charge.
"""

from typing import Callable, Dict, List, Tuple, TYPE_CHECKING

import torch

from .collators import MAX_SENSOR_ID

if TYPE_CHECKING:
    from .geometry import GeometryLoader

FEATURE_COLS_NPE = [
    "c_total", "c_500ns", "c_100ns",
    "t_first", "t_last", "t_20", "t_50", "t_mean", "t_std",
    "x", "y", "z", "x_rel", "y_rel", "z_rel",
]
INPUT_DIM_NPE = len(FEATURE_COLS_NPE)   # 15
_T50_IDX = FEATURE_COLS_NPE.index("t_50")


def compute_dom_features_npe(
    all_features: torch.Tensor,
    pulse_event_idx: torch.Tensor,
    geometry: "GeometryLoader",
    batch_size: int,
    correct_percentiles: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute RAW (un-normalized) NPE-15 per-DOM features for a whole batch.

    Vectorized over all pulses of all events; statistics use ALL pulses per DOM.
    Reductions run in float64 for numerical parity with the polars/double reference
    (t_std via E[t^2]-E[t]^2 is cancellation-sensitive at t ~ 1e4).

    Args:
        all_features: (N, >=3) pulse features [time, charge, sensor_id, ...],
            concatenated across the batch (as built by the flat collators).
        pulse_event_idx: (N,) event index in [0, batch_size) for each pulse.
        geometry: GeometryLoader; positions are used in whatever units it holds
            (normalized /500 in production, raw meters for the validation gate).
        batch_size: number of events in the batch.
        correct_percentiles: if False (default) emit t_20 = t_50 = t_first,
            reproducing the reference store exactly; if True emit genuine
            charge-cumulative 20%/50% quantile times (interpolated).

    Returns:
        feats: (D, 15) float64 raw features in FEATURE_COLS_NPE order.
        dom_event_idx: (D,) event index per DOM (non-decreasing / contiguous).
        dom_sensor_ids: (D,) sensor_id per DOM.
        dom_min_time: (D,) t_first per DOM (float64), for earliest-time subsampling.
    """
    sensor_ids = all_features[:, 2].long()
    times = all_features[:, 0].double()
    charges = all_features[:, 1].double()

    # --- DOM grouping: one group per (event, sensor). sorted=True makes the
    # unique keys — and thus dom_event_idx — non-decreasing (events contiguous). ---
    combined_key = pulse_event_idx * MAX_SENSOR_ID + sensor_ids
    unique_keys, inverse_idx, dom_counts = torch.unique(
        combined_key, return_inverse=True, return_counts=True, sorted=True
    )
    D = unique_keys.shape[0]

    def zeros():
        return torch.zeros(D, dtype=torch.float64)

    # charge sums
    c_total = zeros().scatter_add_(0, inverse_idx, charges)

    # per-DOM first/last time
    t_first = torch.full((D,), float("inf"), dtype=torch.float64)
    t_first.scatter_reduce_(0, inverse_idx, times, reduce="amin", include_self=False)
    t_last = torch.full((D,), float("-inf"), dtype=torch.float64)
    t_last.scatter_reduce_(0, inverse_idx, times, reduce="amax", include_self=False)

    # charge within 500 / 100 ns of the DOM's own first hit
    dt = times - t_first[inverse_idx]
    c_500ns = zeros().scatter_add_(0, inverse_idx, charges * (dt <= 500).double())
    c_100ns = zeros().scatter_add_(0, inverse_idx, charges * (dt <= 100).double())

    # charge-weighted time moments
    sum_qt = zeros().scatter_add_(0, inverse_idx, charges * times)
    sum_qt2 = zeros().scatter_add_(0, inverse_idx, charges * times * times)
    t_mean = sum_qt / c_total
    t_std = (sum_qt2 / c_total - t_mean * t_mean).clamp_min(0.0).sqrt()

    if correct_percentiles:
        t_20 = _charge_cumulative_percentile(times, charges, inverse_idx, D, 0.20)
        t_50 = _charge_cumulative_percentile(times, charges, inverse_idx, D, 0.50)
    else:
        # Reference quirk: both quantiles collapse to the earliest pulse time.
        t_20 = t_first.clone()
        t_50 = t_first.clone()

    # positions (geometry units) + charge-weighted event centre of gravity
    dom_event_idx = torch.div(unique_keys, MAX_SENSOR_ID, rounding_mode="floor")
    dom_sensor_ids = unique_keys % MAX_SENSOR_ID
    dom_pos = geometry[dom_sensor_ids].double()   # (D, 3)

    event_charge = torch.zeros(batch_size, dtype=torch.float64).scatter_add_(
        0, dom_event_idx, c_total
    )
    event_wpos = torch.zeros(batch_size, 3, dtype=torch.float64).scatter_add_(
        0, dom_event_idx.unsqueeze(1).expand(-1, 3), dom_pos * c_total.unsqueeze(1)
    )
    event_cog = event_wpos / event_charge.unsqueeze(1)
    xyz_rel = dom_pos - event_cog[dom_event_idx]

    feats = torch.stack([
        c_total, c_500ns, c_100ns,
        t_first, t_last, t_20, t_50, t_mean, t_std,
        dom_pos[:, 0], dom_pos[:, 1], dom_pos[:, 2],
        xyz_rel[:, 0], xyz_rel[:, 1], xyz_rel[:, 2],
    ], dim=1)   # (D, 15)

    return feats, dom_event_idx, dom_sensor_ids, t_first


def _charge_cumulative_percentile(
    times: torch.Tensor,
    charges: torch.Tensor,
    inverse_idx: torch.Tensor,
    D: int,
    p: float,
) -> torch.Tensor:
    """Interpolated charge-cumulative quantile time per DOM (the *intended*, non-buggy
    definition; only used when ``correct_percentiles=True``). Returns t such that the
    cumulative charge up to t reaches fraction ``p`` of the DOM total, linearly
    interpolated on (cumulative-charge, time)."""
    # Sort pulses by (DOM, time): stable sort on time, then on DOM index.
    o1 = torch.argsort(times, stable=True)
    o2 = torch.argsort(inverse_idx[o1], stable=True)
    order = o1[o2]
    st, sq, sdom = times[order], charges[order], inverse_idx[order]

    # cumulative charge within each DOM
    csum = torch.cumsum(sq, dim=0)
    seg_start_cumsum = torch.zeros(D, dtype=torch.float64)
    n = st.shape[0]
    first_in_seg = torch.ones(n, dtype=torch.bool)
    first_in_seg[1:] = sdom[1:] != sdom[:-1]
    seg_start_cumsum[sdom[first_in_seg]] = (csum - sq)[first_in_seg]
    cum_in_dom = csum - seg_start_cumsum[sdom]         # cumulative charge within DOM
    total = torch.zeros(D, dtype=torch.float64).scatter_add_(0, inverse_idx, charges)

    target = p * total[sdom]
    out = torch.zeros(D, dtype=torch.float64)
    # For each DOM walk to the first pulse whose cumulative charge >= target and
    # interpolate between it and the previous pulse. Done vectorized per pulse:
    reached = cum_in_dom >= target
    # first reached pulse index per DOM
    big = torch.full((n,), n, dtype=torch.long)
    idx_arange = torch.arange(n)
    cand = torch.where(reached, idx_arange, big)
    first_reached = torch.full((D,), n, dtype=torch.long)
    first_reached.scatter_reduce_(0, sdom, cand, reduce="amin", include_self=True)
    fr = first_reached.clamp_max(n - 1)
    t_hi = st[fr]
    c_hi = cum_in_dom[fr]
    prev = (fr - 1).clamp_min(0)
    same_dom = sdom[prev] == torch.arange(D)
    t_lo = torch.where(same_dom, st[prev], st[fr])
    c_lo = torch.where(same_dom, cum_in_dom[prev], torch.zeros_like(c_hi))
    denom = (c_hi - c_lo).clamp_min(1e-12)
    frac = ((p * total) - c_lo) / denom
    out = t_lo + frac.clamp(0.0, 1.0) * (t_hi - t_lo)
    return out


def _segment_median(values: torch.Tensor, dom_event_idx: torch.Tensor,
                    batch_size: int) -> torch.Tensor:
    """Per-event median of ``values`` (one value per DOM), broadcast back to DOMs.

    ``dom_event_idx`` is non-decreasing (events contiguous). Matches numpy.median
    semantics (mean of the two central order statistics for even counts).
    """
    counts = torch.bincount(dom_event_idx, minlength=batch_size)
    starts = torch.zeros(batch_size, dtype=torch.long)
    starts[1:] = counts.cumsum(0)[:-1]

    # sort values within each event (events already contiguous)
    o = torch.argsort(values, stable=True)
    o = o[torch.argsort(dom_event_idx[o], stable=True)]
    sorted_vals = values[o]

    nonempty = counts > 0
    mid = starts + counts // 2                      # upper-middle order statistic
    med = torch.zeros(batch_size, dtype=values.dtype)
    mid_c = mid.clamp_max(sorted_vals.shape[0] - 1)
    hi = sorted_vals[mid_c]
    lo = sorted_vals[(mid_c - 1).clamp_min(0)]
    even = (counts % 2 == 0) & nonempty
    med = torch.where(even, 0.5 * (lo + hi), hi)
    med = torch.where(nonempty, med, torch.zeros_like(med))
    return med[dom_event_idx]


def normalize_dom_features_npe(
    feats: torch.Tensor,
    dom_event_idx: torch.Tensor,
    batch_size: int,
    normalize_positions: bool = False,
) -> torch.Tensor:
    """Apply the reference training-time normalization in place, per event.

    charge:   Q_tilde = 0.2 * ln(1 + Q)                    (c_total/c_500ns/c_100ns)
    time:     (t - t_ref) / 1e4                            (t_first/t_last/t_20/t_50/t_mean)
              t_std scaled by 1/1e4 only (spread; no shift)
    position: / 500 iff ``normalize_positions`` (geometry already /500 in production)
    where t_ref is the per-event median of t_50 (matches normalize_dom_array_npe).
    """
    if feats.shape[0] == 0:
        return feats
    feats[:, 0:3] = torch.log1p(feats[:, 0:3]) * 0.2
    t_ref = _segment_median(feats[:, _T50_IDX].clone(), dom_event_idx, batch_size)
    feats[:, 3:8] = (feats[:, 3:8] - t_ref.unsqueeze(1)) / 1.0e4
    feats[:, 8] = feats[:, 8] / 1.0e4
    if normalize_positions:
        feats[:, 9:15] = feats[:, 9:15] / 500.0
    return feats


def make_collate_npe15(
    geometry: "GeometryLoader",
    max_doms: int = 128,
    normalize_positions: bool = False,
    correct_percentiles: bool = False,
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """Factory for the NPE-15 summary-statistics collator.

    Consumes the same raw-event batches as ``make_collate_flat`` (each item has
    ``pulse_features`` (n_pulses, 4) = [time, charge, sensor_id, auxiliary],
    ``event_id`` and optionally ``target``) and emits 15-dim per-DOM tokens.

    Args:
        geometry: GeometryLoader with DOM positions (normalized /500 in production).
        max_doms: cap on DOMs per event; overflow keeps the EARLIEST-first-pulse
            DOMs (unified with make_collate_flat; the reference used top-charge).
        normalize_positions: divide positions by 500 (leave False when geometry is
            already the /500-normalized file — the production default).
        correct_percentiles: emit genuine charge-cumulative quantile times for
            t_20/t_50 instead of reproducing the reference's t_first collapse.

    Returns:
        collate_fn producing dict with dom_vectors (B, max_doms, 15),
        padding_mask (B, max_doms) True=valid, targets (B, 2) if present, event_ids.
    """
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)

        pulse_features_list = [event["pulse_features"] for event in batch]
        event_lengths = torch.tensor(
            [pf.shape[0] for pf in pulse_features_list], dtype=torch.long
        )
        all_features = torch.cat(pulse_features_list, dim=0)   # (N, 4)
        pulse_event_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), event_lengths
        )

        feats, dom_event_idx, _dom_sensor_ids, dom_min_time = compute_dom_features_npe(
            all_features, pulse_event_idx, geometry, batch_size,
            correct_percentiles=correct_percentiles,
        )
        normalize_dom_features_npe(feats, dom_event_idx, batch_size,
                                   normalize_positions=normalize_positions)
        dom_vectors = feats.float()                            # (D, 15)

        # --- subsample >max_doms events by earliest first-pulse time, then pad ---
        event_dom_counts = torch.bincount(dom_event_idx, minlength=batch_size)
        dom_event_starts = torch.zeros(batch_size + 1, dtype=torch.long)
        dom_event_starts[1:] = event_dom_counts.cumsum(0)
        dom_idx_in_event = (
            torch.arange(dom_vectors.shape[0], dtype=torch.long)
            - dom_event_starts[dom_event_idx]
        )

        needs_subsample = event_dom_counts > max_doms
        if needs_subsample.any():
            priority = -dom_min_time                            # earliest = highest
            keep = torch.ones(dom_vectors.shape[0], dtype=torch.bool)
            for ev in needs_subsample.nonzero(as_tuple=True)[0]:
                s = dom_event_starts[ev]
                e = dom_event_starts[ev + 1]
                _, top = priority[s:e].topk(max_doms, largest=True)
                keep[s:e] = False
                keep[s + top] = True
            kept_idx = keep.nonzero(as_tuple=True)[0]
            dom_vectors = dom_vectors[kept_idx]
            dom_event_idx = dom_event_idx[kept_idx]

            clamped = event_dom_counts.clamp(max=max_doms)
            kept_starts = torch.zeros(batch_size + 1, dtype=torch.long)
            kept_starts[1:] = clamped.cumsum(0)
            dom_idx_in_event = (
                torch.arange(dom_vectors.shape[0], dtype=torch.long)
                - kept_starts[dom_event_idx]
            )

        valid = dom_idx_in_event < max_doms
        ev_idx = dom_event_idx[valid]
        d_idx = dom_idx_in_event[valid]

        padded = torch.zeros(batch_size, max_doms, INPUT_DIM_NPE, dtype=dom_vectors.dtype)
        mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)
        padded[ev_idx, d_idx] = dom_vectors[valid]
        mask[ev_idx, d_idx] = True

        result = {
            "dom_vectors": padded,
            "padding_mask": mask,
            "event_ids": torch.stack([b["event_id"] for b in batch]),
            "batch_size": batch_size,
        }
        if "target" in batch[0]:
            result["targets"] = torch.stack([b["target"] for b in batch])
        return result

    return collate_fn

"""
Hybrid raw+aggregate DOM tokenization (T1 / T2 of the hi-E improvement plan).

Each DOM token is a fixed 256-dim vector that carries BOTH the fine leading-edge
timing of the raw first-K pulses (``make_collate_flat``) AND the whole-series
integrated summary statistics of the human-features encoding
(``make_collate_npe15``), so neither the truncation blur of raw nor the
per-pulse-structure loss of the aggregate encoding limits the bright-DOM regime.

The width is 256 = ``d_model`` exactly, so the transformer runs with
``input_mode: none`` (identity, no input projection) or ``input_mode: linear``.

Two token layouts, both exactly 256 wide
========================================

``mode='full'``  (T1 — the main encoding bet):
  slot        width  contents
  [0:3]         3    x, y, z              geometry position (/500), == make_collate_flat base
  [3]           1    n_tilde              log1p(n_pulses)/3 - 1, == make_collate_flat base
  [4:16]       12    npe aggregate block  the 15 npe15 features MINUS the absolute
                                          x,y,z (already in the base above), i.e.
                                          c_total, c_500ns, c_100ns, t_first, t_last,
                                          t_20, t_50, t_mean, t_std, x_rel, y_rel, z_rel
  [16:256]    240    raw pulse block      K'=80 triplets (t,q,a), make_collate_flat norm
  total        256   = 4 + 12 + 3*80
  K' math:  4 + 12 + 3*K' = 256  ->  K' = 80.

``mode='context_only'``  (T2(a) — the event-context isolation ablation):
  slot        width  contents
  [0:3]         3    x, y, z              geometry position (/500)
  [3]           1    n_tilde
  [4:10]        6    event-context block  x_rel, y_rel, z_rel (CoG-relative position),
                                          dt_first, dt_last (DOM first/last-hit time on
                                          the event-median-t reference), t_ref_global
                                          (the event's median-t anchor on the raw scale)
  [10:256]    246    raw pulse block      K''=82 triplets (t,q,a)
  total        256   = 4 + 6 + 3*82
  K'' math: 4 + 6 + 3*K'' = 256  ->  K'' = 82.

  The 6 context dims are exactly the "where does this DOM sit in the event's
  spatial (CoG) and temporal (median-t) frame" signals that npe15 carries over
  raw, WITHOUT any charge-weighted full-pulse aggregation (no c_total, no t_std,
  no t_mean). This isolates *event context* from *full-pulse aggregation*: if
  raw+context alone closes most of raw->humanstats hi-E gap, context is the
  driver and a lean token beats the full hybrid at lower width.

``include_event_context`` (default True)
========================================
When False, the event-relative reference frame is removed (T2 isolation knob):
  * ``mode='full'``: the CoG-relative trio (x_rel, y_rel, z_rel) is zeroed and the
    npe time columns (t_first..t_mean), which are event-median-centered, are
    re-anchored to the global t=1e4 reference (so no per-event time shift
    survives; t_std, a spread, is untouched). The absolute base x,y,z, n_tilde,
    the charge sums, and the raw pulse block are unchanged.
  * ``mode='context_only'``: the whole 6-dim context block is zeroed, leaving a
    matched-width raw-only baseline (base + zeros + raw triplets).

Reuse
=====
The 12-dim aggregate block is produced by ``compute_dom_features_npe`` +
``normalize_dom_features_npe`` (byte-identical to make_collate_npe15's stats,
including the t_20==t_50==t_first reference quirk). The raw pulse block mirrors
``make_collate_flat``'s first-K packing and normalization. Both share the same
DOM ordering because both group by ``combined_key = event*MAX_SENSOR_ID + sensor``
with ``torch.unique(..., sorted=True)``; alignment is asserted in the unit tests.

Subsampling: events with more than ``max_doms`` DOMs keep the EARLIEST-first-pulse
DOMs (house policy, unified with make_collate_flat / make_collate_npe15).
"""

from typing import Callable, Dict, List, TYPE_CHECKING

import torch

from .collators import MAX_SENSOR_ID, earliest_dom_keep_mask
from .collators_npe import (
    FEATURE_COLS_NPE,
    _T50_IDX,
    _segment_median,
    compute_dom_features_npe,
    normalize_dom_features_npe,
)

if TYPE_CHECKING:
    from .geometry import GeometryLoader

HYBRID_WIDTH = 256  # == d_model; enables input_mode 'none' (identity)

# K per mode chosen so the token is exactly 256 wide.
K_RAW_FULL = 80          # 4 base + 12 aggregate + 3*80 = 256
K_RAW_CONTEXT = 82       # 4 base + 6 context   + 3*82 = 256

# Indices into the 15-col npe feature vector that we keep for the 'full' aggregate
# block: everything EXCEPT the absolute positions x(9), y(10), z(11), which are
# already carried verbatim by the base [x, y, z] slot.
_NPE_KEEP_FULL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]  # 12 features
assert [FEATURE_COLS_NPE[i] for i in _NPE_KEEP_FULL] == [
    "c_total", "c_500ns", "c_100ns",
    "t_first", "t_last", "t_20", "t_50", "t_mean", "t_std",
    "x_rel", "y_rel", "z_rel",
]

# Column offsets inside the normalized 15-col npe feature vector.
_NPE_XREL = 12   # x_rel, y_rel, z_rel are cols 12..14
_NPE_TFIRST = 3  # t_first, t_last are cols 3, 4


def _pack_raw_block(
    all_features: torch.Tensor,
    pulse_event_idx: torch.Tensor,
    batch_size: int,
    K: int,
):
    """Pack the first-K pulses of each DOM into flat (t, q, a) triplets.

    Mirrors ``make_collate_flat``: DOM grouping via ``combined_key`` +
    ``torch.unique(sorted=True)`` (identical ordering to compute_dom_features_npe),
    first-K pulses in original (time-ascending) order, and the same per-feature
    normalization
        t -> (t - 1e4) / 3e4,  q -> log10(clamp(q, 1e-6)) / 3,  aux -> aux - 0.5.

    Returns:
        raw_flat: (D, 3*K) float tensor of packed triplets, DOM padding = 0.
        dom_sensor_ids: (D,) sensor_id per DOM (for alignment checks).
        dom_counts: (D,) total pulses per DOM (ALL pulses, for n_tilde).
    """
    total_pulses = all_features.shape[0]
    sensor_ids = all_features[:, 2].long()
    combined_key = pulse_event_idx * MAX_SENSOR_ID + sensor_ids
    unique_keys, inverse_idx, dom_counts = torch.unique(
        combined_key, return_inverse=True, return_counts=True, sorted=True
    )
    total_doms = unique_keys.shape[0]

    # within-DOM pulse index (stable argsort preserves time order within a DOM)
    sort_order = torch.argsort(inverse_idx, stable=True)
    sorted_dom_idx = inverse_idx[sort_order]
    dom_starts = torch.zeros(total_doms + 1, dtype=torch.long)
    dom_starts[1:] = dom_counts.cumsum(0)
    pulse_idx_in_dom_sorted = (
        torch.arange(total_pulses, dtype=torch.long) - dom_starts[sorted_dom_idx]
    )
    pulse_idx_in_dom = torch.empty(total_pulses, dtype=torch.long)
    pulse_idx_in_dom[sort_order] = pulse_idx_in_dom_sorted

    keep_mask = pulse_idx_in_dom < K
    kept_features = all_features[keep_mask]
    kept_dom_idx = inverse_idx[keep_mask]
    kept_pulse_idx = pulse_idx_in_dom[keep_mask]

    time_norm = (kept_features[:, 0] - 1e4) / 3e4
    charge_norm = torch.log10(kept_features[:, 1].clamp(min=1e-6)) / 3.0
    aux_norm = kept_features[:, 3] - 0.5

    pulse_tensor = torch.zeros(total_doms, K, 3, dtype=all_features.dtype)
    pulse_tensor[kept_dom_idx, kept_pulse_idx, 0] = time_norm
    pulse_tensor[kept_dom_idx, kept_pulse_idx, 1] = charge_norm
    pulse_tensor[kept_dom_idx, kept_pulse_idx, 2] = aux_norm

    raw_flat = pulse_tensor.reshape(total_doms, K * 3)
    dom_sensor_ids = unique_keys % MAX_SENSOR_ID
    return raw_flat, dom_sensor_ids, dom_counts


def make_collate_hybrid(
    geometry: "GeometryLoader",
    max_doms: int = 128,
    mode: str = "full",
    include_event_context: bool = True,
    normalize_positions: bool = False,
    correct_percentiles: bool = False,
    fast_collate: bool = False,
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """Factory for the hybrid raw+aggregate DOM collator (256-dim tokens).

    Args:
        geometry: GeometryLoader with DOM positions (normalized /500 in production).
        max_doms: cap on DOMs per event; overflow keeps the EARLIEST-first-pulse
            DOMs (unified with make_collate_flat / make_collate_npe15).
        mode: 'full' (base + 12 aggregate + 80 raw triplets) or 'context_only'
            (base + 6 event-context + 82 raw triplets). Both are exactly 256 wide.
        include_event_context: if False, remove the event-relative frame (see the
            module docstring) for the T2 isolation ablation.
        normalize_positions: divide positions by 500 (leave False when geometry is
            already the /500-normalized file — the production default).
        correct_percentiles: emit genuine charge-cumulative quantile times for
            t_20/t_50 instead of reproducing the reference's t_first collapse.

    Returns:
        collate_fn producing dict with dom_vectors (B, max_doms, 256),
        padding_mask (B, max_doms) True=valid, targets (B, 2) if present, event_ids.
    """
    if mode not in ("full", "context_only"):
        raise ValueError(f"Unknown hybrid mode: {mode!r} (use 'full' or 'context_only')")
    K = K_RAW_FULL if mode == "full" else K_RAW_CONTEXT

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)

        pulse_features_list = [event["pulse_features"] for event in batch]
        event_lengths = torch.tensor(
            [pf.shape[0] for pf in pulse_features_list], dtype=torch.long
        )
        all_features = torch.cat(pulse_features_list, dim=0)  # (N, 4)
        pulse_event_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), event_lengths
        )

        # --- aggregate (npe) block: reuse the npe15 feature computation ---
        feats, dom_event_idx, dom_sensor_ids, dom_min_time = compute_dom_features_npe(
            all_features, pulse_event_idx, geometry, batch_size,
            correct_percentiles=correct_percentiles,
        )
        D = feats.shape[0]
        # Per-event time reference (median of t_50) on RAW times, per-DOM broadcast.
        # Read before normalize mutates the columns (normalize computes the same
        # value the same way, so the two references are identical).
        t_ref = _segment_median(feats[:, _T50_IDX].clone(), dom_event_idx, batch_size)
        normalize_dom_features_npe(feats, dom_event_idx, batch_size,
                                   normalize_positions=normalize_positions)

        # --- raw pulse block: mirror make_collate_flat's first-K packing ---
        raw_flat, raw_sensor_ids, dom_counts = _pack_raw_block(
            all_features, pulse_event_idx, batch_size, K
        )
        # Both paths group by the same key with sorted unique -> identical DOM order.
        assert torch.equal(raw_sensor_ids, dom_sensor_ids), \
            "raw / aggregate DOM ordering diverged"

        # --- base [x, y, z, n_tilde] (== make_collate_flat base) ---
        # positions are the un-normalized geometry values in feats cols 9:12 when
        # normalize_positions=False (the production default) -> byte-identical to
        # make_collate_flat's geometry lookup.
        base_xyz = feats[:, 9:12]
        n_tilde = (torch.log1p(dom_counts.double()) / 3.0 - 1.0).unsqueeze(1)
        base = torch.cat([base_xyz, n_tilde], dim=1)  # (D, 4) float64

        if mode == "full":
            if not include_event_context:
                feats[:, _NPE_XREL:_NPE_XREL + 3] = 0.0                 # zero x_rel,y_rel,z_rel
                # re-anchor event-median-centered times to the global t=1e4 frame:
                # normalized value is (t - t_ref)/1e4; add (t_ref - 1e4)/1e4 -> (t - 1e4)/1e4.
                feats[:, 3:8] += ((t_ref - 1.0e4) / 1.0e4).unsqueeze(1)
            agg = feats[:, _NPE_KEEP_FULL]                              # (D, 12)
            dom_vectors = torch.cat([base, agg, raw_flat.double()], dim=1)
        else:  # context_only
            if include_event_context:
                t_ref_global = ((t_ref - 1.0e4) / 3.0e4).unsqueeze(1)
                ctx = torch.cat([
                    feats[:, _NPE_XREL:_NPE_XREL + 3],                 # x_rel, y_rel, z_rel
                    feats[:, _NPE_TFIRST:_NPE_TFIRST + 2],             # dt_first, dt_last
                    t_ref_global,                                       # event time anchor
                ], dim=1)                                              # (D, 6)
            else:
                ctx = torch.zeros(D, 6, dtype=feats.dtype)
            dom_vectors = torch.cat([base, ctx, raw_flat.double()], dim=1)

        assert dom_vectors.shape[1] == HYBRID_WIDTH, dom_vectors.shape
        dom_vectors = dom_vectors.float()                              # (D, 256)

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
            if fast_collate:
                keep = earliest_dom_keep_mask(
                    dom_min_time, dom_event_idx, event_dom_counts, max_doms, batch_size
                )
            else:
                priority = -dom_min_time                               # earliest = highest
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

        # fast_collate emits bf16 (halves pad zeros-alloc + pinned H2D copy)
        out_dtype = torch.bfloat16 if fast_collate else dom_vectors.dtype
        padded = torch.zeros(batch_size, max_doms, HYBRID_WIDTH, dtype=out_dtype)
        mask = torch.zeros(batch_size, max_doms, dtype=torch.bool)
        padded[ev_idx, d_idx] = dom_vectors[valid].to(out_dtype)
        mask[ev_idx, d_idx] = True

        result = {
            "dom_vectors": padded,
            "padding_mask": mask,
            "event_ids": torch.stack([b["event_id"] for b in batch]),
            "batch_size": batch_size,
        }
        if "target" in batch[0]:
            result["targets"] = torch.stack([b["target"] for b in batch])
        # True pulse count per event (pulse-count-sliced validation metrics); extra key only.
        if "n_pulses" in batch[0]:
            result["n_pulses"] = torch.stack([b["n_pulses"] for b in batch])
        return result

    return collate_fn

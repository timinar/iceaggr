#!/usr/bin/env python3
"""
Inference script for FlatTransformerV2 (K=84, ~5M params).

Saves parquet with columns:
  [event_id, azimuth_pred, zenith_pred, azimuth_true, zenith_true,
   angular_error_deg, n_pulses, n_doms_in_model]

n_doms_in_model is the count after padding-mask truncation (capped at
max_doms). Use scripts/add_event_complexity.py to join the true uncapped
n_doms and n_strings and write a final CSV.

Usage:
    uv run python scripts/inference.py
    uv run python scripts/inference.py --checkpoint checkpoints/v2-none-K84-5M-proper-split-10ep/best.pt
    uv run python scripts/inference.py --batch-range 656 660 --output our_predictions.parquet
"""

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from iceaggr.data import (
    IceCubeDataset,
    GeometryLoader,
    make_collate_flat,
    make_collate_hybrid,
    make_collate_npe15,
)
from iceaggr.models import FlatTransformerV2
from iceaggr.utils import get_logger

logger = get_logger(__name__)

CHECKPOINT = "checkpoints/v2-none-K84-5M-proper-split-10ep/best.pt"
GEOMETRY_PATH = "/groups/pheno/inar/icecube_kaggle/sensor_geometry_normalized.csv"
OUTPUT_PATH = "our_predictions_656_659.parquet"

MODEL_CONFIG = {
    "max_pulses_per_dom": 84,
    "d_model": 256,
    "max_doms": 128,
    "num_heads": 8,
    "num_layers": 6,
    "hidden_dim": 1024,
    "head_hidden_dim": 1024,
    "dropout": 0.0,  # No dropout at inference
    "input_mode": "none",
}


def unit_vector_to_angles(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert unit vectors to (azimuth, zenith) using our convention.

    Our convention (from angles_to_unit_vector in losses.py):
      x = cos(azimuth) * sin(zenith)
      y = sin(azimuth) * sin(zenith)
      z = cos(zenith)

    So:
      zenith = acos(z)
      azimuth = atan2(y, x), wrapped to [0, 2pi]
    """
    pred = F.normalize(pred.float(), dim=-1)
    zenith = torch.acos(pred[:, 2].clamp(-1.0, 1.0))
    azimuth = torch.atan2(pred[:, 1], pred[:, 0])
    azimuth = azimuth % (2 * math.pi)
    return azimuth, zenith


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load geometry
    logger.info(f"Loading geometry from {args.geometry}")
    geometry = GeometryLoader(args.geometry)

    # Load the checkpoint; the architecture comes from its saved config (every
    # run since the proper-split era carries one): the COMPLETE model section —
    # shape, input mode/dim, head type + vMF settings, ablation/RoPE/spacetime
    # flags — with dropout forced off. MODEL_CONFIG is only the fallback for
    # config-less legacy checkpoints. --max-doms still wins.
    logger.info(f"Loading model from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    _m = _cfg.get("model", {})
    config = dict(MODEL_CONFIG)
    config.update({k: v for k, v in _m.items() if k != "version"})
    config["dropout"] = 0.0
    if args.max_doms is not None:
        config["max_doms"] = args.max_doms
    logger.info(f"Model config from checkpoint: {config}")
    _order_time = bool(ckpt.get("config", {}).get("data", {}).get("order_doms_by_time",
                       _m.get("order_doms_by_time", False))) if isinstance(ckpt.get("config"), dict) else False
    model = FlatTransformerV2(config)

    # Checkpoint format: {"model": state_dict, "epoch": ..., "val_loss": ..., ...}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    # Strip torch.compile prefix "_orig_mod." if present
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    # Legacy vMF checkpoints predate the registered kappa_reg buffer
    if config.get("head_type") == "vmf" and "vmf_loss.kappa_reg" not in state_dict:
        state_dict["vmf_loss.kappa_reg"] = torch.tensor(float(config.get("vmf_kappa_reg", 1e-4)))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Load dataset for batches 656–660
    batch_min, batch_max = args.batch_range
    logger.info(f"Loading dataset for batches {batch_min}–{batch_max}")
    dataset = IceCubeDataset(
        split="train",
        batch_range=(batch_min, batch_max),
        cache_size=5,
        min_pulses=args.min_pulses,
    )
    logger.info(f"Events: {len(dataset):,}")

    # Tokenization follows the checkpoint's data config (flat / npe15 / hybrid);
    # a hybrid or npe15 model fed flat tokens would run but predict garbage.
    _d = _cfg.get("data", {})
    tokenization = _d.get("tokenization", "flat")
    # fast_collate changes the token dtype (bf16) and overflow-DOM tie-breaking,
    # so inference must use the same setting the model was trained with.
    fast_collate = bool(_d.get("fast_collate", False))
    if tokenization == "npe15":
        collate_fn = make_collate_npe15(
            geometry,
            max_doms=config["max_doms"],
            normalize_positions=_d.get("npe_normalize_positions", False),
            correct_percentiles=_d.get("npe_correct_percentiles", False),
            fast_collate=fast_collate,
        )
    elif tokenization == "hybrid":
        collate_fn = make_collate_hybrid(
            geometry,
            max_doms=config["max_doms"],
            mode=_d.get("hybrid_mode", "full"),
            include_event_context=_d.get("hybrid_include_event_context", True),
            normalize_positions=_d.get("npe_normalize_positions", False),
            correct_percentiles=_d.get("npe_correct_percentiles", False),
            fast_collate=fast_collate,
        )
    elif tokenization == "flat":
        collate_fn = make_collate_flat(
            geometry,
            max_pulses_per_dom=config["max_pulses_per_dom"],
            max_doms=config["max_doms"],
            order_doms_by_time=_order_time,
            fast_collate=fast_collate,
        )
    else:
        raise ValueError(f"Unknown data.tokenization in checkpoint: {tokenization!r}")
    # Autocast precision as trained (training.amp_dtype: fp16 default, bf16 for Muon runs)
    _t = _cfg.get("training", {})
    amp_dtype = torch.bfloat16 if str(_t.get("amp_dtype", "fp16")).lower() in ("bf16", "bfloat16") else torch.float16
    logger.info(f"Tokenization: {tokenization} (fast_collate={fast_collate}), autocast {amp_dtype}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # Run inference
    records = []
    total_batches = len(loader)
    logger.info(f"Running inference over {total_batches} mini-batches...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 50 == 0:
                logger.info(f"  batch {i}/{total_batches}")

            dom_vectors = batch["dom_vectors"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            event_ids = batch["event_ids"].numpy()
            targets = batch["targets"].numpy()  # (B, 2) [azimuth_true, zenith_true]

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                out = model(dom_vectors, padding_mask)
            # FlatTransformerV2 now returns a dict; the point estimate is the
            # (kappa-weighted, for vMF) unit direction under the 'direction' key.
            pred_vectors = out["direction"] if isinstance(out, dict) else out  # (B, 3)

            azimuth_pred, zenith_pred = unit_vector_to_angles(pred_vectors.float().cpu())

            # Count valid DOMs per event as a proxy for n_pulses
            n_doms = padding_mask.sum(dim=1).cpu().numpy()

            for j in range(len(event_ids)):
                records.append({
                    "event_id": int(event_ids[j]),
                    "azimuth_pred": float(azimuth_pred[j]),
                    "zenith_pred": float(zenith_pred[j]),
                    "azimuth_true": float(targets[j, 0]),
                    "zenith_true": float(targets[j, 1]),
                    "n_doms_in_model": int(n_doms[j]),
                })

    logger.info(f"Collected {len(records):,} predictions")

    # Add true n_pulses from dataset metadata
    n_pulses_map = {
        int(dataset.event_ids[k]): int(dataset.last_pulse_idx[k] - dataset.first_pulse_idx[k] + 1)
        for k in range(len(dataset))
    }
    for r in records:
        r["n_pulses"] = n_pulses_map.get(r["event_id"], -1)

    df = pl.DataFrame(records)

    # Compute angular error as an explicit column
    az = df["azimuth_pred"].to_numpy()
    zen = df["zenith_pred"].to_numpy()
    az_true = df["azimuth_true"].to_numpy()
    zen_true = df["zenith_true"].to_numpy()
    x_pred = np.cos(az) * np.sin(zen)
    y_pred = np.sin(az) * np.sin(zen)
    z_pred = np.cos(zen)
    x_true = np.cos(az_true) * np.sin(zen_true)
    y_true = np.sin(az_true) * np.sin(zen_true)
    z_true = np.cos(zen_true)
    dot = np.clip(x_pred * x_true + y_pred * y_true + z_pred * z_true, -1, 1)
    angular_err_deg = np.degrees(np.arccos(dot))
    df = df.with_columns(pl.Series("angular_error_deg", angular_err_deg))

    logger.info(f"Saving to {args.output}")
    df.write_parquet(args.output)

    # Quick sanity check
    logger.info(f"azimuth range: [{az.min():.3f}, {az.max():.3f}] (expect [0, 2π]={2*math.pi:.3f})")
    logger.info(f"zenith range: [{zen.min():.3f}, {zen.max():.3f}] (expect [0, π]={math.pi:.3f})")
    logger.info(f"Median angular error: {np.median(angular_err_deg):.2f}°")
    logger.info(f"Mean angular error: {np.mean(angular_err_deg):.2f}°")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with FlatTransformerV2")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--geometry", default=GEOMETRY_PATH, help="Path to sensor geometry CSV")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output parquet path")
    parser.add_argument("--batch-range", type=int, nargs=2, default=[656, 659],
                        metavar=("MIN", "MAX"), help="Batch range (inclusive)")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--max-doms", type=int, default=None,
                        help="Override max_doms from MODEL_CONFIG")
    parser.add_argument("--min-pulses", type=int, default=None,
                        help="Only include events with >= min_pulses")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

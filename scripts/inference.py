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

from iceaggr.data import IceCubeDataset, GeometryLoader, make_collate_flat
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

    # Override model config if needed
    config = dict(MODEL_CONFIG)
    if args.max_doms is not None:
        config["max_doms"] = args.max_doms

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = FlatTransformerV2(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=config["max_pulses_per_dom"],
        max_doms=config["max_doms"],
    )
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

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred_vectors = model(dom_vectors, padding_mask)  # (B, 3)

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

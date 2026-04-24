#!/usr/bin/env python3
"""
vMF inference that also saves κ and mixture weights for selection studies.

Extra columns vs infer_vmf.py:
    kappa_1, kappa_2, kappa_3            (B, K) per-component concentration
    weight_1, weight_2, weight_3         (B, K) softmax of log_weights
    kappa_eff                            scalar Σ_k w_k κ_k (overall confidence)
    kappa_max                            max κ across components
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/lustre/hpc/pheno/inar/iceaggr")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from iceaggr.data import IceCubeDataset, GeometryLoader, make_collate_flat
from iceaggr.models import FlatTransformerV2
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def unit_vector_to_angles(pred: torch.Tensor):
    pred = F.normalize(pred.float(), dim=-1)
    zenith = torch.acos(pred[:, 2].clamp(-1.0, 1.0))
    azimuth = torch.atan2(pred[:, 1], pred[:, 0]) % (2 * math.pi)
    return azimuth, zenith


def build_model_config_from_ckpt(ckpt_cfg: dict) -> dict:
    m = ckpt_cfg["model"]
    cfg = {
        "max_pulses_per_dom": m["max_pulses_per_dom"],
        "d_model": m["d_model"],
        "max_doms": m["max_doms"],
        "num_heads": m["num_heads"],
        "num_layers": m["num_layers"],
        "hidden_dim": m["hidden_dim"],
        "head_hidden_dim": m.get("head_hidden_dim", 128),
        "dropout": 0.0,
        "input_mode": m.get("input_mode", "mlp"),
        "head_type": m.get("head_type", "directional"),
    }
    for k in ("vmf_components", "vmf_kappa_min", "vmf_kappa_max", "vmf_kappa_reg"):
        if k in m:
            cfg[k] = m[k]
    return cfg


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = build_model_config_from_ckpt(ckpt["config"])
    if args.max_doms is not None:
        model_cfg["max_doms"] = args.max_doms
    assert model_cfg["head_type"] == "vmf", "This script is for vMF checkpoints"
    K = model_cfg.get("vmf_components", 1)
    kappa_min = model_cfg.get("vmf_kappa_min", 1.0)
    kappa_max = model_cfg.get("vmf_kappa_max", 500.0)
    logger.info(f"Model config: {model_cfg}")

    model = FlatTransformerV2(model_cfg)
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval().to(device)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    geometry = GeometryLoader(args.geometry)
    b_min, b_max = args.batch_range
    dataset = IceCubeDataset(
        split="train",
        batch_range=(b_min, b_max),
        cache_size=5,
        min_pulses=args.min_pulses,
    )
    logger.info(f"Events: {len(dataset):,}"
                + (f" (min_pulses={args.min_pulses})" if args.min_pulses else ""))

    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=model_cfg["max_pulses_per_dom"],
        max_doms=model_cfg["max_doms"],
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    records = []
    total = len(loader)
    logger.info(f"Running over {total} mini-batches...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 50 == 0:
                logger.info(f"  batch {i}/{total}")
            dom_vectors = batch["dom_vectors"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            event_ids = batch["event_ids"].numpy()
            targets = batch["targets"].numpy()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(dom_vectors, padding_mask)

            direction = out["direction"].float().cpu()
            raw_kappa = out["raw_kappa"].float().cpu()  # (B, K)
            log_weights = out["log_weights"].float().cpu()  # (B, K)

            kappa = torch.clamp(F.softplus(raw_kappa) + kappa_min, max=kappa_max)  # (B, K)
            weights = F.softmax(log_weights, dim=-1)  # (B, K)
            kappa_eff = (weights * kappa).sum(dim=-1)  # (B,)
            kappa_max_per = kappa.max(dim=-1).values  # (B,)

            az, zen = unit_vector_to_angles(direction)
            n_doms = padding_mask.sum(dim=1).cpu().numpy()

            kappa_np = kappa.numpy()
            weights_np = weights.numpy()

            for j in range(len(event_ids)):
                rec = {
                    "event_id": int(event_ids[j]),
                    "azimuth_pred": float(az[j]),
                    "zenith_pred": float(zen[j]),
                    "azimuth_true": float(targets[j, 0]),
                    "zenith_true": float(targets[j, 1]),
                    "n_doms_in_model": int(n_doms[j]),
                    "kappa_eff": float(kappa_eff[j]),
                    "kappa_max": float(kappa_max_per[j]),
                }
                for k_ in range(K):
                    rec[f"kappa_{k_+1}"] = float(kappa_np[j, k_])
                    rec[f"weight_{k_+1}"] = float(weights_np[j, k_])
                records.append(rec)

    logger.info(f"Collected {len(records):,} predictions")

    n_pulses_map = {
        int(dataset.event_ids[k]): int(dataset.last_pulse_idx[k] - dataset.first_pulse_idx[k] + 1)
        for k in range(len(dataset))
    }
    for r in records:
        r["n_pulses"] = n_pulses_map.get(r["event_id"], -1)

    df = pl.DataFrame(records)

    az = df["azimuth_pred"].to_numpy()
    zen = df["zenith_pred"].to_numpy()
    az_t = df["azimuth_true"].to_numpy()
    zen_t = df["zenith_true"].to_numpy()
    xp = np.cos(az) * np.sin(zen); yp = np.sin(az) * np.sin(zen); zp = np.cos(zen)
    xt = np.cos(az_t) * np.sin(zen_t); yt = np.sin(az_t) * np.sin(zen_t); zt = np.cos(zen_t)
    err = np.degrees(np.arccos(np.clip(xp * xt + yp * yt + zp * zt, -1, 1)))
    df = df.with_columns(pl.Series("angular_error_deg", err))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    logger.info(f"Wrote {args.output}")
    logger.info(f"Mean err: {np.mean(err):.2f}°  Median: {np.median(err):.2f}°")
    logger.info(f"κ_eff: P10={np.percentile(df['kappa_eff'], 10):.2f} "
                f"P50={np.percentile(df['kappa_eff'], 50):.2f} "
                f"P90={np.percentile(df['kappa_eff'], 90):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default=str(PROJECT_ROOT / "checkpoints/v2-vmf-K84-5M-mix3-10ep/best.pt"))
    p.add_argument("--geometry",
                   default="/groups/pheno/inar/icecube_kaggle/sensor_geometry_normalized.csv")
    p.add_argument("--output",
                   default=str(PROJECT_ROOT / "paper/predictions/our_predictions_vmf_kappa_656_659.parquet"))
    p.add_argument("--batch-range", type=int, nargs=2, default=[656, 659])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-doms", type=int, default=None)
    p.add_argument("--min-pulses", type=int, default=None,
                   help="Only include events with >= min_pulses (for high-activity eval)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

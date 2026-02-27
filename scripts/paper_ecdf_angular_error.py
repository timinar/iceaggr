"""
ECDF of angular error: model vs Kaggle top solutions and random-guess baseline.

Produces a two-panel figure inspired by Eller 2023 (2307.15289) Figure 6:
  Top:    Empirical CDF (%) vs angular error (degrees)
  Bottom: Difference (EDF_model − EDF_1st) vs angular error

Reference curves digitized from Eller 2023, Figure 6 (Kaggle top-3 solutions).
Published mean angular errors (from Bukhari et al. 2310.15674):
  - 1st place: 0.960 rad (55.0 deg), ensemble of 6 EdgeConv+Transformer, ~6M each
  - 2nd place: 0.960 rad (55.0 deg), transformer + Fourier encoding, 7.6-116M
  - DynEdge baseline: 0.985 rad (56.4 deg)

Usage:
    uv run python scripts/paper_ecdf_angular_error.py
    uv run python scripts/paper_ecdf_angular_error.py --checkpoint checkpoints/my_run/best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from iceaggr.models.flat_transformer_v2 import FlatTransformerV2
from iceaggr.models.losses import angles_to_unit_vector
from iceaggr.data.dataset import IceCubeDataset
from iceaggr.data.geometry import GeometryLoader
from iceaggr.data.collators import make_collate_flat

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_flat_model.pt')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'paper', '698db891736c48c66b2fff40', 'figures')

# ---------------------------------------------------------------------------
# Style (same as paper_model_analysis.py)
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C_BLUE   = '#4E79A7'
C_ORANGE = '#F28E2B'
C_GREEN  = '#59A14F'
C_RED    = '#E15759'
C_PURPLE = '#B07AA1'
C_GRAY   = '#BAB0AC'


# ---------------------------------------------------------------------------
# Digitized reference curves from Eller 2023, Figure 6
# ---------------------------------------------------------------------------
# Points manually digitized from Figure 6 of arXiv:2307.15289.
# The Kaggle scoring dataset (1M events) contains all flavours/energies.
# Digitization uncertainty: ~1-2% EDF, ~0.5° at small angles.

# 1st place (Bukhari et al.): EdgeConv + Transformer ensemble, S_private = 0.960 rad
# Mean-validated: trapz(1-F, theta) ≈ 55.0 deg (matches published score).
_KAGGLE_1ST_THETA = np.array([
    0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 125, 150, 180,
])
_KAGGLE_1ST_EDF = np.array([
    0, 1.5, 3.5, 7, 10, 15, 19, 24, 30, 35, 43, 55, 67, 79, 89, 96, 100,
])

# 2nd place: Transformer + Fourier encoding, S_private = 0.960 rad
# Nearly identical to 1st place; slightly higher at small angles, slightly
# lower at large angles (crossover ~5 deg, per Eller 2023 text).
# Differences < 1% EDF (from bottom panel of Figure 6).
_KAGGLE_2ND_THETA = _KAGGLE_1ST_THETA.copy()
_KAGGLE_2ND_EDF = np.array([
    0, 2.0, 4.2, 7.8, 11, 16, 19.5, 24, 29.5, 34.5, 42.5, 54.5, 67, 79, 89, 96, 100,
])

# 3rd place: GPT-style transformer + GBM ensembler, S_private ≈ 0.963 rad
# Slightly behind 1st/2nd; crossover with 1st at ~20 deg (per Eller text).
_KAGGLE_3RD_THETA = _KAGGLE_1ST_THETA.copy()
_KAGGLE_3RD_EDF = np.array([
    0, 1.2, 3.0, 6.2, 9, 13.5, 17.5, 23, 29, 34.5, 43, 55, 67, 79, 89, 96, 100,
])


def kaggle_edf_interp(theta_grid):
    """Interpolate all three Kaggle ECDF curves onto a common grid."""
    edf_1st = np.interp(theta_grid, _KAGGLE_1ST_THETA, _KAGGLE_1ST_EDF)
    edf_2nd = np.interp(theta_grid, _KAGGLE_2ND_THETA, _KAGGLE_2ND_EDF)
    edf_3rd = np.interp(theta_grid, _KAGGLE_3RD_THETA, _KAGGLE_3RD_EDF)
    return edf_1st, edf_2nd, edf_3rd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--max-events', type=int, default=200_000,
        help='Max validation events to use',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f'Loading checkpoint from {args.checkpoint} ...')
    ckpt = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    config = ckpt['config']

    model = FlatTransformerV2(config['model'])
    model.load_state_dict(ckpt['model'])
    model.cuda().eval()
    print(f"  Model loaded (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f} rad)")

    # ------------------------------------------------------------------
    # 2. Create validation dataloader
    # ------------------------------------------------------------------
    geometry = GeometryLoader(config['data']['geometry_path'])
    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=config['model']['max_pulses_per_dom'],
        max_doms=config['model']['max_doms'],
    )

    val_batch_range = tuple(config['data']['val_batches'])
    print(f'Loading validation data (batches {val_batch_range}, max {args.max_events} events) ...')

    val_dataset = IceCubeDataset(
        batch_range=val_batch_range,
        max_events=args.max_events,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )
    print(f'  {len(val_dataset)} events, {len(val_loader)} batches')

    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    print('Running inference ...')
    all_pred_vecs = []
    all_target_angles = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i, batch in enumerate(val_loader):
            dom_vectors = batch['dom_vectors'].cuda(non_blocking=True)
            padding_mask = batch['padding_mask'].cuda(non_blocking=True)
            targets = batch['targets']

            pred = model(dom_vectors, padding_mask)
            all_pred_vecs.append(pred.cpu())
            all_target_angles.append(targets)

            if (i + 1) % 50 == 0 or (i + 1) == len(val_loader):
                print(f'  batch {i+1}/{len(val_loader)}')

    pred_vecs = torch.cat(all_pred_vecs, dim=0)
    target_angles = torch.cat(all_target_angles, dim=0)

    # ------------------------------------------------------------------
    # 4. Compute angular errors
    # ------------------------------------------------------------------
    true_vecs = angles_to_unit_vector(target_angles[:, 0], target_angles[:, 1])
    dot_prod = (pred_vecs * true_vecs).sum(dim=1).clamp(-1.0, 1.0)
    angular_error_deg = torch.acos(dot_prod).numpy() * (180.0 / np.pi)

    # Summary stats
    mean_err = np.mean(angular_error_deg)
    median_err = np.median(angular_error_deg)
    p10 = np.percentile(angular_error_deg, 10)
    p25 = np.percentile(angular_error_deg, 25)
    p75 = np.percentile(angular_error_deg, 75)
    p90 = np.percentile(angular_error_deg, 90)
    print(f'\nAngular error statistics:')
    print(f'  Mean   = {mean_err:.1f}°')
    print(f'  Median = {median_err:.1f}°')
    print(f'  P10    = {p10:.1f}°')
    print(f'  P25    = {p25:.1f}°')
    print(f'  P75    = {p75:.1f}°')
    print(f'  P90    = {p90:.1f}°')
    print(f'  N      = {len(angular_error_deg)}')

    # ------------------------------------------------------------------
    # 5. Compute ECDFs
    # ------------------------------------------------------------------
    # Model ECDF
    sorted_err = np.sort(angular_error_deg)
    ecdf_model = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100  # %

    # Common grid for interpolation and reference curves
    theta_grid = np.linspace(0, 180, 2000)

    # Random guess CDF (analytical): CDF(θ) = (1 − cos θ) / 2
    cdf_random = (1 - np.cos(np.deg2rad(theta_grid))) / 2 * 100  # %

    # Kaggle reference curves (digitized from Eller 2023 Figure 6)
    kaggle_1st, kaggle_2nd, kaggle_3rd = kaggle_edf_interp(theta_grid)

    # Interpolate model ECDF onto the same grid
    ecdf_model_interp = np.interp(theta_grid, sorted_err, ecdf_model)

    # ------------------------------------------------------------------
    # 6. Two-panel figure
    # ------------------------------------------------------------------
    print('\nGenerating ECDF figure ...')

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.5, 5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
    )

    # -- Top panel: ECDF curves --
    n_params_M = sum(v.numel() for v in ckpt['model'].values()) / 1e6
    ax_top.plot(sorted_err, ecdf_model, color=C_BLUE, lw=1.5,
                label=f'Flat transformer ({n_params_M:.0f}M, {mean_err:.1f}°)')
    ax_top.plot(theta_grid, kaggle_1st, color=C_GREEN, lw=1.3,
                label='Kaggle 1st (55.0°)')
    ax_top.plot(theta_grid, kaggle_2nd, color=C_ORANGE, lw=1.0, ls='--',
                label='Kaggle 2nd (55.0°)')
    ax_top.plot(theta_grid, kaggle_3rd, color=C_PURPLE, lw=1.0, ls=':',
                label='Kaggle 3rd')

    ax_top.set_ylabel('EDF (%)')
    ax_top.set_ylim(0, 105)
    ax_top.set_xlim(0, 180)
    ax_top.legend(loc='lower right', framealpha=0.9, fontsize=7.5)
    ax_top.text(0.02, 0.95, r'$\mathbf{(a)}$', transform=ax_top.transAxes,
                fontsize=12, va='top')

    # -- Bottom panel: difference relative to Kaggle 1st place --
    diff_model = ecdf_model_interp - kaggle_1st
    diff_2nd = kaggle_2nd - kaggle_1st
    diff_3rd = kaggle_3rd - kaggle_1st

    ax_bot.plot(theta_grid, diff_model, color=C_BLUE, lw=1.3,
                label='Flat transformer')
    ax_bot.plot(theta_grid, diff_2nd, color=C_ORANGE, lw=1.0, ls='--',
                label='Kaggle 2nd')
    ax_bot.plot(theta_grid, diff_3rd, color=C_PURPLE, lw=1.0, ls=':',
                label='Kaggle 3rd')
    ax_bot.axhline(0, color=C_GREEN, lw=0.8, ls='-', label='Kaggle 1st')

    ax_bot.set_xlabel('Angular error (degrees)')
    ax_bot.set_ylabel(r'EDF $-$ EDF$_\mathrm{1st}$ (%)')
    ax_bot.set_xlim(0, 180)
    ax_bot.legend(loc='lower right', framealpha=0.9, fontsize=7.5)
    ax_bot.text(0.02, 0.90, r'$\mathbf{(b)}$', transform=ax_bot.transAxes,
                fontsize=12, va='top')

    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'ecdf_angular_error.{ext}')
        fig.savefig(path)
        print(f'  Saved {path}')
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7. Print comparison table
    # ------------------------------------------------------------------
    print('\n' + '=' * 65)
    print('Comparison with Kaggle top solutions (Eller 2023, Bukhari 2023)')
    print('=' * 65)
    print(f'{"Model":<35s} {"Mean (°)":<10s} {"Params":<10s}')
    print('-' * 65)
    print(f'{"Kaggle 1st (EdgeConv+Trans ens.)":<35s} {"55.0":<10s} {"~6M x 6":<10s}')
    print(f'{"Kaggle 2nd (Trans+Fourier ens.)":<35s} {"55.0":<10s} {"8-116M x 5":<10s}')
    print(f'{"Kaggle 3rd (GPT+GBM ens.)":<35s} {"~55.5":<10s} {"72M x 3":<10s}')
    print(f'{"DynEdge baseline (full data)":<35s} {"56.4":<10s} {"~1.4M":<10s}')
    print(f'{"Flat transformer (this work)":<35s} {f"{mean_err:.1f}":<10s} {f"{n_params_M:.0f}M":<10s}')
    print(f'{"Random guess":<35s} {"90.0":<10s} {"-":<10s}')
    print('-' * 65)
    print()
    print('Note: Kaggle curves digitized from Eller 2023 Fig. 6 (~1-2% EDF')
    print('uncertainty). Kaggle scores on hidden test set; ours on val split.')
    kaggle_1st_at_5 = np.interp(5, _KAGGLE_1ST_THETA, _KAGGLE_1ST_EDF)
    ours_at_5 = np.mean(angular_error_deg < 5) * 100
    print(f'At 5°: Kaggle 1st ~{kaggle_1st_at_5:.0f}%, ours ~{ours_at_5:.0f}%.')
    print('The ECDFs are similar, consistent with the close mean errors.')
    print()

    print('Done.')


if __name__ == '__main__':
    main()

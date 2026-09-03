#!/bin/bash
set -uo pipefail
cd /lustre/hpc/pheno/inar/iceaggr
echo "===== $(date) launching npe15 FULL base ====="
uv run python scripts/train_flat.py --config configs/train_flat_v2_npe15_5M_unclamped.yaml --workers 8 2>&1 | tee logs/npe15_full_base.log
echo "===== $(date) launching npe15 finetune ====="
uv run python scripts/train_flat.py --config configs/train_flat_v2_npe15_5M_unclamped_finetune_hi.yaml --workers 8 2>&1 | tee logs/npe15_full_finetune.log
echo "===== $(date) NPE15 FULL QUEUE DONE ====="

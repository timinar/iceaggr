#!/bin/bash
# Self-driving queue for the 100M/2ep encoding trio (all combined loss).
set -uo pipefail
cd /lustre/hpc/pheno/inar/iceaggr
while screen -ls | grep -q enc_K84; do sleep 120; done
for c in npe15_5M_unclamped_100M2ep npe15corr_combined_100M2ep none_K84_5M_unclamped_100M2ep; do
  echo "===== $(date) launching $c ====="
  uv run python scripts/train_flat.py --config configs/train_flat_v2_${c}.yaml --workers 8 2>&1 | tee logs/trio_${c}.log
done
echo "===== $(date) TRIO QUEUE DONE ====="

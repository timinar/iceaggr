#!/bin/bash
set -uo pipefail
cd /lustre/hpc/pheno/inar/iceaggr
for c in npe15_muonft_5em4 npe15_muonft_2p5em4; do
  echo "===== $(date) $c ====="
  uv run python scripts/train_flat.py --config configs/${c}.yaml --workers 8 2>&1 | tee logs/${c}.log
done
echo "===== MUON-FT PROBE DONE ====="

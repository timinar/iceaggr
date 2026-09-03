#!/bin/bash
set -uo pipefail
cd /lustre/hpc/pheno/inar/iceaggr
for c in muon_sweep_adamw_ref muon_sweep_lr0p005 muon_sweep_lr0p01 muon_sweep_lr0p04 muon_sweep_lr0p08; do
  echo "===== $(date) $c ====="
  uv run python scripts/train_flat.py --config configs/${c}.yaml --workers 8 2>&1 | tee logs/${c}.log
done
echo "===== MUON SWEEP DONE ====="

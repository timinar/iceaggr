#!/bin/bash
set -uo pipefail
cd /lustre/hpc/pheno/inar/iceaggr
echo "===== $(date) launching hybrid-muon base mlr0075 ====="
uv run python scripts/train_flat.py --config configs/hybrid_muon_base_mlr0075.yaml --workers 16 2>&1 | tee logs/hybrid_muon_base_mlr0075.log
echo "===== $(date) launching hybrid-muon finetune mlr0075 ====="
uv run python scripts/train_flat.py --config configs/hybrid_muon_finetune_mlr0075.yaml --workers 16 2>&1 | tee logs/hybrid_muon_ft_mlr0075.log
echo "===== HYBRID-MUON MLR0075 QUEUE DONE ====="

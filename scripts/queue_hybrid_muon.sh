#!/bin/bash
set -uo pipefail
cd /lustre/hpc/pheno/inar/iceaggr
echo "===== $(date) launching hybrid-muon base ====="
uv run python scripts/train_flat.py --config configs/hybrid_muon_base.yaml --workers 8 2>&1 | tee logs/hybrid_muon_base.log
echo "===== $(date) launching hybrid-muon finetune ====="
uv run python scripts/train_flat.py --config configs/hybrid_muon_finetune.yaml --workers 8 2>&1 | tee logs/hybrid_muon_ft.log
echo "===== HYBRID-MUON QUEUE DONE ====="

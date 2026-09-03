#!/bin/bash
set -uo pipefail
cd /lustre/hpc/pheno/inar/iceaggr
while screen -ls | grep -q muon_ft; do sleep 180; done
echo "===== $(date) launching npe15-muon base ====="
uv run python scripts/train_flat.py --config configs/npe15_muon_base.yaml --workers 8 2>&1 | tee logs/npe15_muon_base.log
echo "===== $(date) launching npe15-muon finetune ====="
uv run python scripts/train_flat.py --config configs/npe15_muon_finetune.yaml --workers 8 2>&1 | tee logs/npe15_muon_ft.log
echo "===== NPE15-MUON QUEUE DONE ====="

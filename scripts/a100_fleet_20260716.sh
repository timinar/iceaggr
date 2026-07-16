#!/bin/bash
# A100-box fleet launcher — 2026-07-16 hi-E improvement program.
# Usage (on the A100 box): git pull && git checkout feat/vmf-unclamped-K3 && uv sync
#                          ./scripts/a100_fleet_20260716.sh
# Each run gets one GPU + a detached screen; results stream to wandb project 'iceaggr'.
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- preflight ----
[ -f src/iceaggr/data/data_config.yaml ] || { echo "FATAL: src/iceaggr/data/data_config.yaml missing (gitignored, user-local)"; exit 1; }
[ -d /groups/pheno/inar/icecube_kaggle/train ] || echo "WARNING: default Kaggle data path not visible — ensure data_config.yaml points at this box's copy"
python3 -c "import torch" 2>/dev/null || { echo "FATAL: env not synced (run: uv sync)"; exit 1; }
NGPU=$(nvidia-smi -L | wc -l); echo "GPUs visible: $NGPU"
mkdir -p logs

# ---- fleet (slot 8 intentionally empty pending the rawctx preview verdict) ----
declare -a CFG=(fleet_combined_s2 fleet_combined_s3 fleet_combined_lam025 fleet_combined_lam100 \
                fleet_hybrid_linear fleet_hybrid_none fleet_npe15_s2)
for i in "${!CFG[@]}"; do
  c=${CFG[$i]}
  screen -dmS "fleet_$c" bash -c "CUDA_VISIBLE_DEVICES=$i uv run python scripts/train_flat.py \
    --config configs/${c}.yaml --workers 7 2>&1 | tee logs/${c}.log"
  echo "GPU $i <- $c"
done
echo "Launched ${#CFG[@]} runs. Watch: screen -ls | tail -f logs/fleet_*.log | wandb project iceaggr."

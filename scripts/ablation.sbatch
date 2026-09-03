#!/bin/bash
# Per-change architecture ablation — single-GPU slurm job for the gr10_gpu partition.
# Usage:  sbatch -J abl-<name> -o logs/ablation_<name>_%j.log scripts/ablation.sbatch <ablate> <name>
#   <ablate> in {none,vanilla,rmsnorm,qknorm,relu2,zeroinit,residscaling,bias}
#SBATCH --account=pheno
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=10:00:00
#SBATCH --job-name=abl
#SBATCH --output=logs/ablation_%j.log
set -euo pipefail
cd /lustre/hpc/pheno/inar/iceaggr
ABLATE="${1:?usage: sbatch scripts/ablation.sbatch <ablate> <name>}"
NAME="${2:?usage: sbatch scripts/ablation.sbatch <ablate> <name>}"
echo "[$(date)] host=$(hostname) ablate=$ABLATE name=$NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
uv run python scripts/train_flat.py \
  --config configs/ablation_base.yaml \
  --ablate "$ABLATE" --name "$NAME" --workers 3 --no-wandb

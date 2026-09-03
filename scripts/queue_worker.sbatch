#!/bin/bash
# Slurm wrapper: one queue worker on one A4500 (gr10_gpu), draining queue/small.
# Usage:  sbatch scripts/queue_worker.sbatch            (repeat for more workers)
# Jobs in queue/small must fit a 20GB GPU (<=10M params at bs<=1024).
#SBATCH --account=pheno
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --job-name=qworker
#SBATCH --output=logs/queue_worker_slurm_%j.log
set -uo pipefail
# slurm copies this script to a spool dir, so $0 cannot locate the repo: use the
# submit directory (sbatch from the repo root) or ICEAGGR_ROOT.
cd "${ICEAGGR_ROOT:-${SLURM_SUBMIT_DIR:-/lustre/hpc/pheno/inar/iceaggr}}" || exit 1
echo "[$(date)] host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES root=$PWD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Exit when the queue is empty for 10 min so the slot is not held idle.
export QUEUE_IDLE_EXIT=600
exec bash scripts/queue_worker.sh small 30

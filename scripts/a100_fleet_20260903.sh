#!/bin/bash
# A100-box launcher for the 130M-event DEPTH-scaling fleet (2026-09-03).
#
# Runs the box as N independent single-GPU queue workers (scripts/queue_worker.sh,
# class "a100") and enqueues the 130M runs. Why this shape and not one 8-GPU job:
# the July fleet choked on CPU, not I/O — each trainer's MAIN process kept torch's
# default thread pool (~half the cores) spin-waiting, times seven trainers. Here
# every main process is capped at MAIN_THREADS and the dataloader workers per job
# are sized from the core count. The deep models also need ~4x fewer events/s per
# GPU than the 5M did, so the loader is no longer the bottleneck.
#
# Usage (on the box):
#   git pull && git checkout feat/vmf-unclamped-K3 && uv sync
#   GEOM=/path/to/sensor_geometry_normalized.csv ./scripts/a100_fleet_20260903.sh [enqueue|workers|all]
# Optional env: WORKERS=<n> (loader workers per job), MAIN_THREADS=<n>, GPU_MEM_GB=<n>.
# Requires src/iceaggr/data/data_config.yaml (gitignored) pointing at the box's data copy.
# Watch:  tail -f logs/queue_worker_a100_*.log ; ls queue/a100/{pending,running,done,failed}
# Stop:   touch queue/a100/STOP   (workers exit after their current job)
# Resume after a crash: restart `workers`; each worker requeues its own stale claim, and
# scripts/train_flat.py resumes from checkpoints/ladder/<run>/epoch_XXX.pt via --checkpoint.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
MODE="${1:-all}"

# ---- preflight ----
[ -f src/iceaggr/data/data_config.yaml ] || { echo "FATAL: src/iceaggr/data/data_config.yaml missing (gitignored, user-local)"; exit 1; }
uv run python -c "import torch" 2>/dev/null || { echo "FATAL: env not synced (run: uv sync)"; exit 1; }
NGPU=$(nvidia-smi -L | wc -l); NCORE=$(nproc)
GPU_MEM_GB="${GPU_MEM_GB:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}')}"
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
echo "GPUs: $NGPU  cores: $NCORE  RAM: ${RAM_GB} GB  GPU mem: ${GPU_MEM_GB} GB"
GEOM="${GEOM:-/groups/pheno/inar/icecube_kaggle/sensor_geometry_normalized.csv}"
[ -f "$GEOM" ] || { echo "FATAL: geometry file not found: $GEOM (set GEOM=...)"; exit 1; }

# ---- per-job CPU budget ----
# Each trainer = 1 main process (MAIN_THREADS) + WORKERS single-threaded loader workers
# + 2 val-loader workers + 2 train-eval-loader workers. Budget everything within cores/NGPU.
MAIN_THREADS="${MAIN_THREADS:-4}"
PER_GPU=$(( NCORE / NGPU ))
W_DEFAULT=$(( PER_GPU - MAIN_THREADS - 4 )); [ "$W_DEFAULT" -lt 4 ] && W_DEFAULT=4; [ "$W_DEFAULT" -gt 12 ] && W_DEFAULT=12
WORKERS="${WORKERS:-$W_DEFAULT}"
# Each loader worker caches ~1 GB of pulse tables; warn if the box's RAM is tight.
EST_RAM=$(( NGPU * (WORKERS + 4) * 1 + NGPU * 6 ))
echo "per job: $WORKERS loader workers (+2 val, +2 train-eval), main_threads=$MAIN_THREADS; est. RAM for $NGPU jobs ~ ${EST_RAM} GB of ${RAM_GB} GB"
[ "$EST_RAM" -gt $(( RAM_GB * 8 / 10 )) ] && echo "WARNING: worker RAM estimate exceeds 80% of RAM - lower WORKERS=..."

# ---- per-depth micro-batch (measured on H100: ~1.4 GB per layer per 1024 events, bf16,
# d256, max_doms 128, torch.compile). Keep <= 75% of GPU memory; effective batch = 4096
# via gradient accumulation (identical optimizer trajectory up to bf16 noise). ----
micro_bs() {  # $1 = layers
  local L=$1 bs=4096 budget=$(( GPU_MEM_GB * 75 / 10 ))   # tenths of GB
  while [ "$bs" -gt 256 ] && [ $(( L * 14 * bs / 1024 )) -gt "$budget" ]; do bs=$(( bs / 2 )); done
  echo "$bs"
}
bs_args() { local bs; bs=$(micro_bs "$1"); echo "--batch-size $bs --grad-accum $(( 4096 / bs ))"; }
for L in 6 12 24 36; do echo "  L$L: $(bs_args $L)"; done

COMMON="--max-events 130000000 --val-events 200000 --val-per-epoch 5 --epochs 10 \
  --lr 3e-4 --warmup-steps 2000 --dropout 0.1 --weight-decay 0.01 --train-eval-events 200000 \
  --workers $WORKERS --main-threads $MAIN_THREADS --geometry-path $GEOM --tag stageC-130M --save-every 1 --enqueue a100"

enqueue() {
  # Order = priority (workers take the alphabetically first pending job).
  # Per-run overrides (bs/accum/workers) come AFTER $COMMON: argparse keeps the last value.
  uv run python scripts/scaling_ladder.py --name F01_130M_d256L24_s17    --d-model 256 --layers 24 --muon-lr 0.005 --seed 17 $COMMON $(bs_args 24) --note "130M: depth-24 winner of the 10M ladder"
  uv run python scripts/scaling_ladder.py --name F02_130M_d256L6_s17     --d-model 256 --layers 6  --muon-lr 0.005 --seed 17 $COMMON $(bs_args 6)  --note "130M: 5M control, identical recipe/seed/order (paired with F01)"
  uv run python scripts/scaling_ladder.py --name F03_130M_d256L24lin_s17 --d-model 256 --layers 24 --input-mode linear --muon-lr 0.005 --seed 17 $COMMON $(bs_args 24) --note "130M: depth-24 + learned linear input"
  uv run python scripts/scaling_ladder.py --name F04_130M_d256L12_s17    --d-model 256 --layers 12 --muon-lr 0.005 --seed 17 $COMMON $(bs_args 12) --note "130M: depth-12 (middle of the curve)"
  uv run python scripts/scaling_ladder.py --name F05_130M_d256L24_s41    --d-model 256 --layers 24 --muon-lr 0.005 --seed 41 $COMMON $(bs_args 24) --note "130M: depth-24 seed replicate"
  uv run python scripts/scaling_ladder.py --name F06_130M_d256L36_s17    --d-model 256 --layers 36 --muon-lr 0.005 --seed 17 $COMMON $(bs_args 36) --note "130M: depth-36 (does depth keep paying?)"
  # The record hi-E model's tokenization (raw K'=80 + 12 aggregates, Linear 256->d): deep vs control.
  uv run python scripts/scaling_ladder.py --name F07_130M_hyb_d256L24_s17 --tokenization hybrid --d-model 256 --layers 24 --muon-lr 0.005 --seed 17 $COMMON $(bs_args 24) --note "130M: HYBRID tokens, depth-24"
  uv run python scripts/scaling_ladder.py --name F08_130M_hyb_d256L6_s17  --tokenization hybrid --d-model 256 --layers 6  --muon-lr 0.005 --seed 17 $COMMON $(bs_args 6)  --note "130M: HYBRID tokens, 5M control (paper headline base, re-run under the ladder protocol)"
  ls queue/a100/pending
}

workers() {
  mkdir -p logs
  for i in $(seq 0 $((NGPU-1))); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$i \
      nohup bash scripts/queue_worker.sh a100 30 > logs/queue_worker_a100_$i.log 2>&1 &
    echo "worker on GPU $i (pid $!)"
    sleep 2
  done
}

case "$MODE" in
  enqueue) enqueue ;;
  workers) workers ;;
  all) enqueue; workers ;;
  *) echo "usage: $0 [enqueue|workers|all]"; exit 1 ;;
esac
echo "Done. Eight jobs for $NGPU GPUs; anything else you drop into queue/a100/pending/ runs when a worker frees up."

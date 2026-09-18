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
#   GEOM=/path/to/sensor_geometry_normalized.csv ./scripts/a100_fleet_20260903.sh [stageB|enqueue|extra|extra2|extra3|extra4|extra5|extra6|extra7|workers|all|stop]
#   (all = stageB + enqueue + workers; stageB = the 10M follow-ups, which sort first in the queue)
# Optional env: WORKERS=<n> (loader workers per job), MAIN_THREADS=<n>, GPU_MEM_GB=<n>,
#               SCREEN=1 (run each worker in a detached screen session a100_w<i>).
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
# Count ALL cores, not the shell's inherited affinity (tmux/numad pin login shells to a
# NUMA subset on the box; `nproc` would report 64 of 255 and starve the loaders).
NGPU=$(nvidia-smi -L | wc -l); NCORE=$(grep -c ^processor /proc/cpuinfo)
ALLCPU="0-$(( NCORE - 1 ))"
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
echo "per job: $WORKERS loader workers (+ up to 8 val / 8 train-eval workers, idle outside eval), main_threads=$MAIN_THREADS; est. RAM for $NGPU jobs ~ ${EST_RAM} GB of ${RAM_GB} GB"
[ "$EST_RAM" -gt $(( RAM_GB * 8 / 10 )) ] && echo "WARNING: worker RAM estimate exceeds 80% of RAM - lower WORKERS=..."

# ---- per-depth micro-batch (measured on H100: ~1.4 GB per layer per 1024 events, bf16,
# d256, max_doms 128, torch.compile). Keep <= 75% of GPU memory; effective batch = 4096
# via gradient accumulation (identical optimizer trajectory up to bf16 noise). ----
micro_bs() {  # $1 = layers
  local L=$1 bs=4096 budget=$(( GPU_MEM_GB * 75 / 10 ))   # tenths of GB
  while [ "$bs" -gt 256 ] && [ $(( L * 14 * bs / 1024 )) -gt "$budget" ]; do bs=$(( bs / 2 )); done
  echo "$bs"
}
bs_args() {  # $1 = layers, $2 = effective batch (default 4096)
  local bs eff="${2:-4096}"; bs=$(micro_bs "$1"); [ "$bs" -gt "$eff" ] && bs=$eff
  echo "--batch-size $bs --grad-accum $(( eff / bs ))"
}
for L in 6 12 24 36; do echo "  L$L: $(bs_args $L)"; done

# ---- Stage B follow-ups moved off the H100 (10M events, 5 ep, effective bs 1024, seed 17
# unless noted): the remaining 10M-ladder questions. Names sort before F* so the workers run
# these first (each 2–10 A100-hours), then the 130M fleet. Delete a job file from
# queue/a100/pending/ if that run already finished on the H100 (check wandb tag scaling-ladder).
stageB() {
  local C="--max-events 10000000 --val-events 200000 --val-per-epoch 2 --epochs 5 \
    --lr 3e-4 --warmup-steps 2000 --dropout 0.1 --weight-decay 0.01 --train-eval-events 200000 \
    --workers $WORKERS --main-threads $MAIN_THREADS --geometry-path $GEOM --tag stageB --enqueue a100"
  uv run python scripts/scaling_ladder.py --name B07_10M_d256L24lin_mlr005_s17 --d-model 256 --layers 24 --input-mode linear --muon-lr 0.005 --seed 17 $C $(bs_args 24 1024) --note "Stage B: L24 + learned linear input"
  uv run python scripts/scaling_ladder.py --name B08_10M_d256L12lin_mlr005_s17 --d-model 256 --layers 12 --input-mode linear --muon-lr 0.005 --seed 17 $C $(bs_args 12 1024) --note "Stage B: L12 + linear input"
  uv run python scripts/scaling_ladder.py --name B09_10M_d256L24_mlr005_s41    --d-model 256 --layers 24 --muon-lr 0.005 --seed 41 $C $(bs_args 24 1024) --note "Stage B: L24 seed replicate"
  uv run python scripts/scaling_ladder.py --name B09a_10M_hyb_d256L6_s17       --tokenization hybrid --d-model 256 --layers 6  --muon-lr 0.005 --seed 17 $C $(bs_args 6 1024)  --note "Stage B: HYBRID tokens, 5M control"
  uv run python scripts/scaling_ladder.py --name B09b_10M_hyb_d256L24_s17      --tokenization hybrid --d-model 256 --layers 24 --muon-lr 0.005 --seed 17 $C $(bs_args 24 1024) --note "Stage B: HYBRID tokens, depth 24"
  uv run python scripts/scaling_ladder.py --name B12_10M_d256L24_mlr0071_s17   --d-model 256 --layers 24 --muon-lr 0.0071 --seed 17 $C $(bs_args 24 1024) --note "Stage B: L24 Muon-LR bracket upper"
  uv run python scripts/scaling_ladder.py --name B13_10M_d256L24_mlr0035_s17   --d-model 256 --layers 24 --muon-lr 0.0035 --seed 17 $C $(bs_args 24 1024) --note "Stage B: L24 Muon-LR bracket lower"
  uv run python scripts/scaling_ladder.py --name B14_10M_d256L36_mlr005_s17    --d-model 256 --layers 36 --muon-lr 0.005 --seed 17 $C $(bs_args 36 1024) --note "Stage B: depth 36 (28.6M)"
  ls queue/a100/pending
}

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

# Added 2026-09-04 (Inar: "keep GPUs busy") once F02 finished: the 10M LR bracket (B12/B13) put the
# L24 Muon-LR optimum at or below 0.005, so F09 pairs with F01 at 0.0035; F10 is the hybrid-L24 seed
# replicate (pairs with F07). Same recipe/order as the F01-F08 fleet.
extra() {
  uv run python scripts/scaling_ladder.py --name F09_130M_d256L24_mlr0035_s17 --d-model 256 --layers 24 --muon-lr 0.0035 --seed 17 $COMMON $(bs_args 24) --note "130M: depth-24 at Muon lr 0.0035 (10M bracket winner), pairs with F01"
  uv run python scripts/scaling_ladder.py --name F10_130M_hyb_d256L24_s41  --tokenization hybrid --d-model 256 --layers 24 --muon-lr 0.005 --seed 41 $COMMON $(bs_args 24) --note "130M: HYBRID tokens, depth-24 seed replicate (pairs with F07)"
  ls queue/a100/pending
}

# Added 2026-09-06 when F04 finished (GPU 0 idle, queue empty): L6 + learned linear input at 130M
# completes the depth x linear-input 2x2 (F02 raw L6, F03 L24+lin, F01 L24) — B05 showed −0.24° at 5M/10M.
extra2() {
  uv run python scripts/scaling_ladder.py --name F11_130M_d256L6lin_s17 --d-model 256 --layers 6 --input-mode linear --muon-lr 0.005 --seed 17 $COMMON $(bs_args 6) --note "130M: 5M control + learned linear input (pairs with F02; with F03 completes the depth x linear 2x2)"
  ls queue/a100/pending
}

# Added 2026-09-07 when F11 finished: seed replicate of the raw-L6 control. Every paired locked-set
# comparison is measured against F02; the F01/F05 spread (0.014 NLL at epoch 6) says the control's own
# seed noise must be known before the depth effect is quoted.
extra3() {
  uv run python scripts/scaling_ladder.py --name F12_130M_d256L6_s41 --d-model 256 --layers 6 --muon-lr 0.005 --seed 41 $COMMON $(bs_args 6) --note "130M: 5M raw control, seed replicate (control seed noise for the paired comparisons)"
  ls queue/a100/pending
}

# Added 2026-09-07 (Inar: more hybrid runs + a dropout ablation). No matched dropout evidence exists
# for this architecture at >=10M events, and every 130M run has a negative train-dev gap, so dropout 0.1
# is suspected to cost accuracy. Twins differ from their reference in exactly one setting. Queue order
# (per-run overrides such as --dropout 0.0 must come AFTER $COMMON: argparse keeps the last value)
# = priority: the cheap L6 twins first, then the hybrid depth curve, the hybrid LR bracket, then the
# expensive L24 dropout twin (kill it Thursday if F13/F14 show no dropout-0 gain).
extra4() {
  uv run python scripts/scaling_ladder.py --name F13_130M_hyb_d256L6_drop0_s17  --tokenization hybrid --d-model 256 --layers 6  --muon-lr 0.005  --seed 17 $COMMON $(bs_args 6) --dropout 0.0  --note "130M: hybrid L6, NO dropout (twin of F08)"
  uv run python scripts/scaling_ladder.py --name F14_130M_d256L6_drop0_s17      --d-model 256 --layers 6  --muon-lr 0.005  --seed 17 $COMMON $(bs_args 6) --dropout 0.0  --note "130M: raw L6, NO dropout (twin of F02)"
  uv run python scripts/scaling_ladder.py --name F15_130M_hyb_d256L12_s17       --tokenization hybrid --d-model 256 --layers 12 --muon-lr 0.005  --seed 17 $COMMON $(bs_args 12) --note "130M: hybrid L12 (middle of the hybrid depth curve F08-F07)"
  uv run python scripts/scaling_ladder.py --name F16_130M_hyb_d256L6_s41        --tokenization hybrid --d-model 256 --layers 6  --muon-lr 0.005  --seed 41 $COMMON $(bs_args 6)  --note "130M: hybrid L6 seed replicate (hybrid control seed noise, twin of F08)"
  uv run python scripts/scaling_ladder.py --name F17_130M_hyb_d256L24_mlr0035_s17 --tokenization hybrid --d-model 256 --layers 24 --muon-lr 0.0035 --seed 17 $COMMON $(bs_args 24) --note "130M: hybrid L24 at Muon lr 0.0035 (twin of F07; raw bracket favoured the lower lr for depth)"
  uv run python scripts/scaling_ladder.py --name F18_130M_hyb_d256L24_drop0_s17 --tokenization hybrid --d-model 256 --layers 24 --muon-lr 0.005 --seed 17 $COMMON $(bs_args 24) --dropout 0.0 --note "130M: hybrid L24, NO dropout (twin of F07)"
  ls queue/a100/pending
}

# Added 2026-09-09 once the 2x2 (F01/F02/F03/F11), the LR bracket (F09 = F01 within seed noise) and the first
# dropout twin (F14: dropout 0 is worth -0.07..-0.11 deg bulk) were in: the combined recipe candidate.
extra5() {
  uv run python scripts/scaling_ladder.py --name F19_130M_d256L24lin_drop0_s17 --d-model 256 --layers 24 --input-mode linear --muon-lr 0.005 --seed 17 $COMMON $(bs_args 24) --dropout 0.0 --note "130M: raw L24 + linear input + NO dropout (twin of F03; combined-recipe candidate)"
  ls queue/a100/pending
}

# Added 2026-09-10: seed replicate of the best bulk model (F03 = raw L24 + linear input) for the paper table.
extra6() {
  uv run python scripts/scaling_ladder.py --name F20_130M_d256L24lin_s41 --d-model 256 --layers 24 --input-mode linear --muon-lr 0.005 --seed 41 $COMMON $(bs_args 24) --note "130M: raw L24 + linear input, seed replicate of F03"
  ls queue/a100/pending
}

# Added 2026-09-18 (Inar): does the depth gain depend on the optimizer? AdamW twin of the headline model F20
# (raw L24 + linear input, seed 41); AdamW lr 3e-4 = the side-group rate, nothing better found in the July 6-layer sweep.
extra7() {
  uv run python scripts/scaling_ladder.py --name F21_130M_d256L24lin_adamw_s41 --d-model 256 --layers 24 --input-mode linear --adamw --seed 41 $COMMON $(bs_args 24) --note "130M: raw L24 + linear input trained with AdamW (lr 3e-4) instead of Muon; twin of F20"
  ls queue/a100/pending
}

workers() {
  mkdir -p logs
  # taskset resets the inherited CPU affinity (see NCORE above). SCREEN=1 puts each worker
  # in a detached, attachable screen (`screen -r a100_w<i>`); the log file is the same.
  for i in $(seq 0 $((NGPU-1))); do
    if [ "${SCREEN:-0}" = 1 ]; then
      # exec: the worker itself is the screen window's process, so `stop` below (SIGTERM to
      # it) runs its cleanup trap. No tee pipeline: a wrapper shell dying would HUP the
      # worker before its trap could release the job (that orphaned 8 trainers once).
      screen -dmS a100_w$i -L -Logfile logs/queue_worker_a100_$i.log bash -c \
        "exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$i taskset -c $ALLCPU bash scripts/queue_worker.sh a100 30"
      echo "worker on GPU $i (screen a100_w$i)"
    else
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$i \
        taskset -c $ALLCPU nohup bash scripts/queue_worker.sh a100 30 > logs/queue_worker_a100_$i.log 2>&1 &
      echo "worker on GPU $i (pid $!)"
    fi
    sleep 2
  done
}

# Stop the workers gracefully: SIGTERM ONLY the queue_worker.sh processes (identified by
# their CUDA_VISIBLE_DEVICES), never a screen/tee wrapper. Each worker kills its job tree,
# requeues the job, and exits; the jobs resume from their epoch checkpoints on `workers`.
stop() {
  for q in $(pgrep -f "bash scripts/queue_worker.sh a100"); do
    g=$(tr '\0' '\n' < /proc/$q/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ -n "$g" ] || continue
    kill -TERM "$q" && echo "TERM -> worker on GPU $g (pid $q)"
  done
  for i in $(seq 1 40); do
    n=$(pgrep -f "bash scripts/queue_worker.sh a100" | wc -l)
    t=$(pgrep -f "python.* scripts/train_flat.py --config configs/ladder" | wc -l)
    [ "$n" -eq 0 ] && [ "$t" -eq 0 ] && break; sleep 3
  done
  echo "workers left: $n  trainer processes left: $t  running: $(ls queue/a100/running 2>/dev/null | wc -l)  pending: $(ls queue/a100/pending | wc -l)"
  [ "$t" -gt 0 ] && echo "WARNING: trainer processes survived — inspect with: pgrep -af train_flat.py"
  screen -wipe >/dev/null 2>&1 || true
}

case "$MODE" in
  stop) stop ;;
  extra) extra ;;
  extra2) extra2 ;;
  extra3) extra3 ;;
  extra4) extra4 ;;
  extra5) extra5 ;;
  extra6) extra6 ;;
  extra7) extra7 ;;
  enqueue) enqueue ;;
  stageB) stageB ;;
  workers) workers ;;
  all) stageB; enqueue; workers ;;
  *) echo "usage: $0 [enqueue|stageB|workers|all|stop]   (all = stageB + enqueue + workers)"; exit 1 ;;
esac
echo "Done. Eight jobs for $NGPU GPUs; anything else you drop into queue/a100/pending/ runs when a worker frees up."

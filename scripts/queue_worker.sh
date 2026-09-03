#!/bin/bash
# Minimal shared-filesystem job queue for the scaling ladder.
#
# Layout (relative to the repo root, so every machine/checkout sees its own queue):
#   queue/<class>/pending/*.sh   jobs waiting (plain bash scripts; they cd to the repo root themselves)
#   queue/<class>/running/       claimed by a worker (mv is atomic on one FS)
#   queue/<class>/done/ failed/  finished jobs
#   queue/<class>/STOP           touch to make workers exit after the current job
#   logs/queue/<job>.<host>.log  stdout+stderr of each job
#
# Usage:  CUDA_VISIBLE_DEVICES=0 scripts/queue_worker.sh <class> [poll_seconds]
# Classes: h100 (96GB; any job) | small (<=20GB GPUs) | a100 (the 8-GPU box)
# Env:   QUEUE_IDLE_EXIT=<s>  exit after that long with nothing to do (slurm workers)
#        ICEAGGR_ROOT=<dir>   override the repo root (default: this script's parent dir)
#
# Recovery: on SIGTERM/SIGINT (slurm time limit, Ctrl-C) the current job is killed and
# moved back to pending. If a worker dies hard (SIGKILL, host crash) its claim stays in
# running/; the next worker started with the SAME tag (host + GPU) requeues it at startup,
# since two live workers never share a tag.
set -uo pipefail
ROOT="${ICEAGGR_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}"
cd "$ROOT" || { echo "FATAL: cannot cd to repo root $ROOT"; exit 1; }
CLASS="${1:?usage: queue_worker.sh <class> [poll_seconds]}"
POLL="${2:-30}"
Q="queue/$CLASS"
mkdir -p "$Q/pending" "$Q/running" "$Q/done" "$Q/failed" logs/queue
HOST="$(hostname -s)"
TAG="${HOST}.gpu${CUDA_VISIBLE_DEVICES:-all}"
IDLE_EXIT="${QUEUE_IDLE_EXIT:-0}"
echo "[$(date)] worker $TAG draining $Q at $ROOT (poll ${POLL}s)"

# Requeue stale claims left by a previous worker with my tag (it must be dead).
for stale in "$Q"/running/*."$TAG".sh; do
  [ -e "$stale" ] || continue
  base="$(basename "$stale" ".$TAG.sh")"
  mv "$stale" "$Q/pending/$base.sh" && echo "[$(date)] REQUEUED stale claim $base"
done

CURRENT=""; CHILD=""
on_signal() {
  echo "[$(date)] signal received; releasing $CURRENT"
  [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
  if [ -n "$CURRENT" ] && [ -e "$Q/running/$CURRENT.$TAG.sh" ]; then
    mv "$Q/running/$CURRENT.$TAG.sh" "$Q/pending/$CURRENT.sh"
  fi
  exit 143
}
trap on_signal TERM INT

idle=0
while true; do
  if [ -e "$Q/STOP" ]; then echo "[$(date)] STOP present, exiting"; exit 0; fi
  job="$(ls -1 "$Q/pending"/*.sh 2>/dev/null | sort | head -n 1 || true)"
  if [ -z "$job" ]; then
    idle=$((idle + POLL))
    if [ "$IDLE_EXIT" -gt 0 ] && [ "$idle" -ge "$IDLE_EXIT" ]; then
      echo "[$(date)] idle ${idle}s >= ${IDLE_EXIT}s, exiting"; exit 0
    fi
    sleep "$POLL"; continue
  fi
  idle=0
  name="$(basename "$job" .sh)"
  # Claim atomically; another worker may have grabbed it first.
  if ! mv "$job" "$Q/running/$name.$TAG.sh" 2>/dev/null; then continue; fi
  CURRENT="$name"
  log="logs/queue/$name.$TAG.log"
  echo "[$(date)] START $name on $TAG -> $log"
  bash "$Q/running/$name.$TAG.sh" > "$log" 2>&1 &
  CHILD=$!
  wait "$CHILD"; rc=$?
  CHILD=""
  if [ $rc -eq 0 ]; then
    mv "$Q/running/$name.$TAG.sh" "$Q/done/$name.$TAG.sh"
    echo "[$(date)] DONE  $name (rc=0)"
  elif [ -e "$Q/running/$name.$TAG.sh" ]; then
    mv "$Q/running/$name.$TAG.sh" "$Q/failed/$name.$TAG.sh"
    echo "[$(date)] FAIL  $name (rc=$rc) see $log"
  fi
  CURRENT=""
done

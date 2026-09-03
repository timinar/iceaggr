# Task: re-verify the fixes to your audit (delegation/scaling-sol/audit.md) before push

Read-only audit again; write only `delegation/scaling-sol/audit2.md` (≤80 lines). The tree is
now committed as two commits on `feat/vmf-unclamped-K3`: `git show --stat HEAD~1` (June/July
production changes) and `git show --stat HEAD` (scaling ladder + fleet launcher). Review with
`git show HEAD` / `git diff HEAD~2..HEAD` plus the files themselves.

Claimed fixes (verify each; say FIXED / NOT FIXED / PARTIAL with file:line):
1. inference.py builds the model from the complete saved `config.model` (dropout 0, --max-doms
   override) and injects the legacy `vmf_loss.kappa_reg` buffer. Verified live: the L24 vMF
   ladder checkpoint and `checkpoints/v2-hybrid-muon-combined-10ep/best.pt` load and predict.
2. inference.py dispatches flat/npe15/hybrid collators from the checkpoint's `data.tokenization`.
3. `scripts/a100_fleet_20260903.sh` is mode 0755 in git.
4. Generated queue jobs and `queue_worker.sh` locate the repo root from their own path
   (`readlink -f "$0"`); `queue_worker.sbatch` uses ICEAGGR_ROOT / SLURM_SUBMIT_DIR.
5. Mid-epoch validation cadence is rounded to a multiple of grad_accum (checkpoints land right
   after an optimizer step); the epoch-end flush is keyed on `actual_batch`. Smoke-tested with
   accum=2, val_per_epoch=3, 2 epochs, then resumed from epoch_001.pt: epoch-2 result matched
   the uninterrupted run (3.2894 vs 3.2891 loss).
6. Checkpoints carry python/numpy/torch/cuda RNG + GradScaler state; resume restores them;
   `torch.load(..., weights_only=False)` on resume.
7. Launcher: depth-aware micro-batch from a measured 1.4 GB/layer/1024 events (L24@bs1024 =
   34 GB, L12 = 18 GB on H100), effective batch 4096 via accumulation, 75% GPU-memory budget.
8. CPU budget = cores/GPU − main_threads − 4 (val + train-eval loaders), clamp [4,12], RAM
   estimate printed; no hybrid worker bump.
9. `--max-events 130000000`.
10. `val/angular_error_rad` keeps the historical batch-mean-under-autocast computation;
    the event-weighted value is `val/angular_error_event_deg`.
11. `training.seed: null` → sampler seed 42.
12. Worker: SIGTERM/SIGINT trap requeues the running job; a restarted worker with the same
    host+GPU tag requeues its own stale claims.
13. dataset: a missing per-batch metadata file raises instead of silently skipping.
Not addressed on purpose (say if you disagree): 14 tail-window weighting, 15 median
convention, 16 `--print-params` tokenization.

Also flag anything NEW that the fixes introduced, and anything in the two commits that
should not be pushed. Unit suite already re-run: 124 passed.

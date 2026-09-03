# Task: audit the uncommitted working tree before it is committed and shipped to an 8×A100 box

You are auditing, not editing. **Do not modify, create, or delete any file except your report**
`delegation/scaling-sol/audit.md` (≤200 lines). You may run read-only commands (git diff, git
status, grep, python one-liners) and `uv run pytest tests/unit -q` (takes ~3 min).

## Context
Repo root = working dir, branch `feat/vmf-unclamped-K3`. The working tree holds TWO groups of
uncommitted work that will become two commits and then be `git pull`ed on a separate 8×A100
machine to run 130M-event trainings for ~a month:

**Group 1 — June/July production changes never committed** (they ran every July production
model and the paper's ablations, so they are believed correct, but were never reviewed):
- `src/iceaggr/models/flat_transformer_v2.py`: per-change ablation flags (`use_rmsnorm`,
  `use_qknorm`, `use_relu2`, `use_zero_init`, `use_resid_scaling`, `use_bias`), opt-in RoPE
  (`use_rope`), opt-in relative spacetime attention bias (`use_spacetime_bias`). All must be
  byte-identical to the old model at defaults (existing checkpoints must load and predict
  identically).
- `src/iceaggr/data/dataset.py`: Arrow allocator switch (memory-pool env) to bound worker RSS,
  per-batch metadata loading.
- `scripts/train_flat.py` (July hunks): `--ablate`, fast_collate wiring, rope/spacetime config
  passthrough, `apply_ablation`.
- `scripts/inference.py` (July hunks): rope / order_doms_by_time awareness.
- `tests/unit/test_collators.py` (+ new `tests/unit/test_flat_transformer_v2.py`), the base config
  `configs/train_flat_v2_none_K84_muon_combined_10ep.yaml` (muon_lr 0.02→0.005, fast_collate),
  ~34 untracked experiment configs `configs/*.yaml`, July scripts `scripts/queue_*.sh`,
  `scripts/muon_sweep.sh`, `scripts/ablation.sbatch`, `scripts/benchmark_*.py`.

**Group 2 — scaling-ladder infrastructure written 2026-09-02/03 (by Claude):**
- `scripts/train_flat.py`: `validate()` now returns per-event angular error stats, median,
  pulse-count slices (≥200/≥1000 via a new `n_pulses` batch key), pure per-event vMF NLL
  (`val/nll`); `training.seed` (python/numpy/torch + `BatchAwareSampler(seed=)`);
  `data.train_eval_events` = eval-mode scoring of fixed train events each epoch (gap);
  grad-norm/clip-rate logging inside the accumulation step; `training.main_threads`
  (`torch.set_num_threads` in the main process).
- `src/iceaggr/models/vmf_loss.py`: `_log_likelihood_fp32` factored out of `_forward_fp32`
  (training loss must be numerically identical), new `per_event_nll()` (no κ penalty).
- `src/iceaggr/data/collators.py`, `collators_hybrid.py`, `collators_npe.py`: extra `n_pulses`
  key in the batch dict when the dataset provides it.
- `scripts/inference.py`: model architecture read from the checkpoint's saved config
  (d_model, layers, K, input_mode, …), `--max-doms` still overrides.
- New: `scripts/scaling_ladder.py` (config generator + job enqueue; `count_params` must match
  the real model: d256-L6 = 4,997,403; d256-L24 = 19,153,215; d512-L6 = 19,415,579; hybrid
  d256-L6 with Linear(256,256) = 5,063,195), `scripts/ladder_summary.py` (wandb table),
  `scripts/queue_worker.sh` + `scripts/queue_worker.sbatch` (shared-FS file queue: claim by
  atomic `mv`, done/failed dirs, STOP file, idle-exit), `scripts/a100_fleet_20260903.sh`
  (the box launcher: sizes loader workers from `nproc`, caps main threads, picks bs/accum from
  GPU memory, enqueues 8 runs F01–F08 incl. two hybrid-tokenization runs), `configs/ladder/*.yaml`
  (generated), `.gitignore` (+`queue/`).

## What to check (ranked; be concrete, cite file:line)
1. **Silent behaviour changes on the default path** of Group 1 and 2 (model forward, loss value,
   collator outputs, sampler order, checkpoint resume, mid-epoch resume with `batch_idx`). In
   particular: does `training.seed` unset reproduce the historical behaviour exactly? Does the
   `validate()` rewrite change `val/loss` or `val/angular_error_rad` values or the best-checkpoint
   selection? Is the `vmf_loss` refactor bit-identical under torch.compile?
2. **Multi-GPU-box hazards** in the launcher and queue: CPU/thread oversubscription (each trainer
   = main process + `num_workers` single-threaded workers + a 2-worker val loader + a 2-worker
   train-eval loader), memory (bs/accum choice for 40 vs 80 GB), argument precedence in
   `$COMMON` vs per-run overrides, `mv`-claim races with 8 workers, what happens on job crash,
   `set -euo pipefail` interactions, the geometry-path override, anything hard-coded to
   `/groups/pheno/...` or `/lustre/...` that will not exist on the box.
3. **Correctness of the new metrics**: per-event angular error vs `angular_distance_loss`
   epsilon/abs conventions; pure NLL vs the training NLL (κ penalty, angular term); slices;
   train-eval loader construction (`batch_range=(first, first)`, `max_events`, `min_pulses`).
4. **`scaling_ladder.py`**: shape conventions (heads = d//32, ff = 4d), what `--tokenization
   hybrid` sets vs `configs/hybrid_muon_base.yaml`, `--kappa-reg0`, save_every, wandb tags.
5. **Commit hygiene**: anything that should not be committed (secrets, absolute user paths,
   large/generated files), and whether the proposed two-commit split (Group 1 = model/dataset/
   tests/configs/July scripts; Group 2 = everything else incl. the four shared files
   train_flat.py, inference.py, collators.py, vmf_loss.py) is sensible.
6. Run `uv run pytest tests/unit -q` and report failures verbatim.

## Deliverable
`delegation/scaling-sol/audit.md`: (a) verdict — safe to commit+ship or not; (b) numbered findings
with severity (BLOCKER / SHOULD-FIX / NIT), file:line, one-line fix each; (c) the exact list of
files per commit you recommend; (d) anything you could not verify.

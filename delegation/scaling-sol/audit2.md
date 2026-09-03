# Two-commit re-audit — 2026-09-03

## Verdict

**NOT SAFE TO PUSH AS-IS.** The architecture/loading blocker is fixed and most claimed
changes are present, but inference still does not reproduce the saved preprocessing/precision,
resume is not RNG-exact, and worker requeue can duplicate or restart long jobs from scratch.
The scaling commit also contains large unrelated delegation trees that should be removed.

## Claimed fixes

1. **FIXED** — the saved model mapping is merged (except schema-only `version`), dropout is
   zeroed, `--max-doms` wins, compile prefixes are stripped, and the legacy vMF buffer is
   injected (`scripts/inference.py:91-117`). I accept the reported live load/predict checks.
2. **PARTIAL** — flat/npe15/hybrid dispatch and their semantic options follow
   `data.tokenization` (`scripts/inference.py:135-161`), but `data.fast_collate` is omitted.
   Every ladder config enables it (`configs/ladder/B03_10M_d256L24_mlr005_s17.yaml:33-34`),
   and it changes dtype and overflow-DOM tie-breaking, so inference does not match training.
3. **FIXED** — Git records mode `100755` (`scripts/a100_fleet_20260903.sh:1`).
4. **FIXED** — generated jobs self-locate (`scripts/scaling_ladder.py:209-214`), the worker
   self-locates (`scripts/queue_worker.sh:21-22`), and Slurm prefers `ICEAGGR_ROOT` then
   `SLURM_SUBMIT_DIR` (`scripts/queue_worker.sbatch:14-16`). The final hard-coded fallback remains.
5. **FIXED** — optimizer/flush decisions use the resumed `actual_batch` at epoch end
   (`scripts/train_flat.py:503-520`), and cadence is floored to an accumulation multiple
   (`scripts/train_flat.py:903-912`). I accept the reported smoke result; it is close, not exact.
6. **PARTIAL** — Python/NumPy/Torch/CUDA and scaler states are captured
   (`scripts/train_flat.py:321-347`), saved in all checkpoint paths (`:583-595`, `:1002-1033`),
   and restored with `weights_only=False` (`:859-890`). But the DataLoader has no dedicated
   generator (`:301-311`): constructing its iterator after restore consumes global Torch RNG,
   unlike uninterrupted mid-epoch execution, so dropout continuation is not RNG-exact.
7. **FIXED (policy only)** — the 1.4-GB/layer estimate, 75% budget, power-of-two micro-batch,
   and accumulation to 4096 are implemented (`scripts/a100_fleet_20260903.sh:47-56`). A100 fit
   and the transferability of the H100 measurement remain unverified here.
8. **FIXED** — default workers are `cores/GPU-main_threads-4`, clamped to [4,12], and RAM is
   estimated/printed (`scripts/a100_fleet_20260903.sh:35-45`); hybrid jobs add no workers (`:71-73`).
9. **FIXED** — fleet common args use 130,000,000 (`scripts/a100_fleet_20260903.sh:58-60`).
10. **FIXED** — the historical autocast batch mean remains under the old key, while the fp32
    event mean is separate (`scripts/train_flat.py:637-675`) and logged as
    `val/angular_error_event_deg` (`:693-699`).
11. **FIXED** — null/unset maps to sampler seed 42 (`scripts/train_flat.py:255-258`).
12. **PARTIAL** — same-tag stale claims are requeued (`scripts/queue_worker.sh:32-37`) and the
    signal trap requeues the current claim (`:39-48`). It kills only the immediate bash PID,
    neither its process group nor waits for death, so `uv`/`tee` descendants may survive and race
    the requeued copy. Same-tag liveness is assumed rather than locked.
13. **FIXED** — a missing requested per-batch metadata file raises
    `FileNotFoundError` before concatenation (`src/iceaggr/data/dataset.py:113-140`).

## Deferred 14–16

14. **NOT FIXED, acceptable only as an explicit approximation** (`scripts/train_flat.py:500-506`).
    I would still fix it: every 130M run has a 1,152-event final effective-batch tail, whose update
    is under-scaled; it is only one optimizer update per epoch, so it is not the main blocker.
15. **NOT FIXED, acceptable nit** — lower-middle `Tensor.median` remains (`scripts/train_flat.py:675-686`).
16. **NOT FIXED, acceptable nit** — print-only counting bypasses tokenization (`scripts/scaling_ladder.py:183-188`).

## New / push-blocking findings

- **Inference pipeline mismatch:** besides omitted `fast_collate`, inference hard-codes CUDA
  autocast's default fp16 (`scripts/inference.py:187`) although ladder checkpoints specify bf16
  (`configs/ladder/B03_10M_d256L24_mlr005_s17.yaml:30-34`). Loading and predicting is not enough
  to establish metric-equivalent inference. Propagate both saved settings and reject unknown tokens.
- **Requeue is not resume:** generated jobs run only `train_flat.py --config ...`
  (`scripts/scaling_ladder.py:202-214`); generated configs retain `checkpoint.resume: null`.
  Thus the launcher claim that crash recovery resumes (`scripts/a100_fleet_20260903.sh:19-20`)
  is false: stale/signal-requeued 130M jobs restart unless a checkpoint is explicitly selected.
- **Commit hygiene:** remove the unrelated `delegation/enc_hiE/`, `delegation/final-critique/`,
  `delegation/hiE-sol/`, `delegation/paper-*`, and `delegation/specs/` trees from `HEAD`.
  `delegation/enc_hiE/jsrc/` duplicates product source and its `data_config.yaml:4-6` embeds local
  paths. Split/drop `docs/vmf_finetune_runbook.md` and `references.bib` unless intentionally scoped.

`git diff --check HEAD~2..HEAD` and all changed shell scripts pass syntax checks. Per instruction,
the reported `124 passed` suite was not rerun. No GPU/A100 fit, real signal, or live checkpoint test
was independently run in this read-only pass.

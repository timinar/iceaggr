# Uncommitted-tree audit — 2026-09-03

## Verdict

**NOT SAFE TO COMMIT + SHIP.** Group 1's default model path looks compatible, but Group 2 has
launcher, inference, and recovery blockers. Fix 1–6 and A100-smoke-test every batch size first.

## Findings

1. **BLOCKER — vMF checkpoints cannot load in the rewritten inference path**
   (`scripts/inference.py:91`). Only a small architecture subset is copied; `head_type`,
   `vmf_components`, `input_dim`, kappa settings, spacetime/ablation flags, etc. are omitted.
   A simulated ladder checkpoint rebuilt as `directional` and failed on incompatible state keys.
   **Fix:** construct from the complete saved `config.model`, overriding only dropout/max-doms.

2. **BLOCKER — hybrid inference silently uses flat tokens** (`scripts/inference.py:129`). Even
   after finding 1 is fixed, F07/F08 are collated with `make_collate_flat`, not the checkpoint's
   `data.tokenization: hybrid`, so 256 values have the wrong semantics.
   **Fix:** select flat/npe15/hybrid collator and its options from the saved data config.

3. **BLOCKER — the documented box launcher is not executable**
   (`scripts/a100_fleet_20260903.sh:14`, filesystem mode `0600`), so the shown `./scripts/...`
   invocation returns permission denied after checkout.
   **Fix:** commit it with mode `0755` (`chmod +x scripts/a100_fleet_20260903.sh`).

4. **BLOCKER — queue jobs are tied to this checkout** (`scripts/scaling_ladder.py:205`,
   `scripts/queue_worker.sh:14`). A generated job has `set -e` then `cd
   /lustre/hpc/pheno/inar/iceaggr`; on a differently located box every job fails immediately.
   The worker's failed `cd` is non-fatal and can instead operate in an unintended directory.
   **Fix:** derive repo root from each script path and emit repo-relative/self-locating jobs.

5. **BLOCKER — accumulated mid-epoch checkpoints are not resumable correctly**
   (`scripts/train_flat.py:474`, `:553`, `:840`; `src/iceaggr/data/samplers.py:87`). Validation
   may save between optimizer steps without serializing pending gradients; resume then skips that
   consumed micro-batch. After sampler skipping, `len(loader)` still reports the full epoch, so
   the “epoch end” flush can also fail, leaving the last accumulated gradients unapplied. The
   40-GB F runs (`accum=2`) and F06 exercise this; their `val_per_epoch=5` boundaries are not all
   accumulation boundaries.
   **Fix:** checkpoint only just after optimizer steps (or save gradients/phase), and compute the
   final flush from the number of remaining yielded batches rather than stale `len(loader)`.

6. **BLOCKER — seeded resume does not restore RNG/scaler state** (`scripts/train_flat.py:711`,
   `:946`). Resume reseeds CUDA at the start instead of continuing dropout RNG; checkpoints omit
   Python/NumPy/CPU/CUDA RNG and GradScaler state. End-epoch and mid-epoch resumes therefore do
   not reproduce uninterrupted training (GradScaler matters on fp16 paths).
   **Fix:** save and restore all RNG states plus `scaler.state_dict()` at checkpoint boundaries.

7. **SHOULD-FIX — GPU-memory policy ignores depth** (`scripts/a100_fleet_20260903.sh:34`). L6
   and L24 both receive bs4096 on 80 GB or bs2048 on 40 GB despite activation memory scaling with
   layers; only L36 is reduced. No target-A100 measurement supports the stated fit.
   **Fix:** choose micro-batch by depth/SKU from measured smoke tests and recover effective 4096
   with accumulation.

8. **SHOULD-FIX — CPU budget undercounts four loader workers and main threads**
   (`scripts/a100_fleet_20260903.sh:32`, `:54`). Each ordinary trainer budgets `W + 2 val + 2
   train-eval + 4 main = W+8`, but the formula reserves only two cores/GPU; hybrids add four more.
   On 128 cores/8 GPUs this requests 168 threads/workers and up to ~116 GB of worker pulse caches.
   **Fix:** derive `W <= floor(NCORE/NGPU)-MAIN_THREADS-4` and remove the hybrid `W+4` bump unless
   spare cores were explicitly reserved.

9. **SHOULD-FIX — 130M runs are recorded as 200M** (`scripts/a100_fleet_20260903.sh:39`,
   `scripts/scaling_ladder.py:124`, `scripts/ladder_summary.py:29`). Batches 1–650 contain exactly
   130,000,000 events, but `max_events=200000000` makes W&B tags/config/summary say 200M.
   **Fix:** pass `--max-events 130000000` (or record actual `len(dataset)` as the sample count).

10. **SHOULD-FIX — the historical angular metric key changed value**
    (`scripts/train_flat.py:614`, `:633`). Old `val/angular_error_rad` averaged batch means and did
    not renormalize predictions; new code renormalizes and averages events. With 200,000 events,
    the partial batch makes the weighting difference real. `val/loss` retains the old batch-mean
    calculation and best-checkpoint selection at `:943` is unchanged.
    **Fix:** keep the old calculation under the old key and publish the event-weighted metric under
    a new key, or explicitly version/rename the changed metric.

11. **SHOULD-FIX — `training.seed: null` is treated inconsistently**
    (`scripts/train_flat.py:257`, `:711`). The seeding block treats null as unset, but sampler
    construction executes `int(None)`; omission (not null) correctly preserves historical seed 42.
    **Fix:** use `sampler_seed = 42 if seed is None else int(seed)`.

12. **SHOULD-FIX — dead workers strand jobs forever** (`scripts/queue_worker.sh:39`, `:42`). An
   ordinary failure moves to `failed`, and competing claims are atomic, but host death/SIGKILL
   leaves a job in `running` with no lease/requeue mechanism.
    **Fix:** add a trap plus timestamp/owner lease and a startup watchdog that requeues stale claims.

13. **SHOULD-FIX — partial per-batch metadata silently drops data**
    (`src/iceaggr/data/dataset.py:117`, `:128`). Missing files are skipped, yet one present table
    makes the partial result authoritative; a damaged box copy can train/validate on the wrong set.
    **Fix:** require every requested file up to the deliberate max-events stopping point, otherwise
    fail loudly or fall back to the monolithic metadata.

14. **SHOULD-FIX — a short final accumulation window is underweighted**
    (`scripts/train_flat.py:470`, `:474`). Loss is always divided by `grad_accum`; if an epoch ends
    with fewer micro-batches, its final update is too small (the fleet's relevant lengths do not all
    divide evenly).
    **Fix:** divide the tail window by its actual micro-batch count, or make/drop full windows.

15. **NIT — medians use the lower middle observation** (`scripts/train_flat.py:634`, `:645`);
    `Tensor.median()` does not average the two middle values for even-sized sets.
    **Fix:** use `torch.quantile(x, 0.5)` if the conventional/NumPy median is intended.

16. **NIT — `--print-params` ignores tokenization** (`scripts/scaling_ladder.py:183`). With
    `--tokenization hybrid`, it still prints the flat/default input-mode count.
    **Fix:** route print-only arguments through the same tokenization normalization as `build()`.

## Verified behavior

- Default Group-1 model compatibility: seeded old-vs-new directional and vMF models had identical
  state-dict keys/tensors and exact eager outputs; exported graphs were also exactly identical.
- vMF refactor: eager loss/gradients and exported graphs were exact; compile-eager was also exact.
  `per_event_nll` correctly excludes the kappa penalty and optional angular term.
- `training.seed` omitted preserves the sampler's historical seed 42 and leaves global RNGs
  unseeded. Numeric seed controls Python, NumPy, Torch/CUDA, and sampler order from a fresh start.
- New slices use the dataset's true uncapped `n_pulses` and inclusive `>=200`/`>=1000` masks.
  Train-eval defaults to batch 1; locally that is exactly 200,000 events. `min_pulses` propagates.
- Parameter counts match real instantiated models exactly: 4,997,403; 19,153,215; 19,415,579;
  hybrid Linear(256,256) 5,063,195. All 31 ladder YAML counts/head/FFN conventions are consistent.
- Hybrid generation matches the base: full, dim 256, learned Linear(256,256); reg0 is fully zero; save cadence/tags are emitted.
- `$COMMON` precedes F06/F07/F08 overrides, so argparse's last-value rule gives intended bs/accum
  and worker overrides; GEOM reaches YAML. Atomic claims and pipefail child status work as intended.
- No credential/API secret was found in changed product files; changed YAML and shell/Python syntax
  parsed (an unrelated tracked `configs/fleet_hybrid_linear.yaml` is already malformed).
- Unit suite (`uv run --no-cache pytest tests/unit -q -p no:cacheprovider`): `124 passed, 4 warnings in 119.34s (0:01:59)`; no failures.

## Recommended two-commit split

The split is sensible only if `train_flat.py` and `inference.py` are hunk-staged: their July
plumbing belongs in commit 1 and scaling/checkpoint-config work in commit 2. Putting both whole
files in commit 2 leaves commit 1 unable to exercise its new model flags/RoPE correctly.

### Commit 1 — June/July production changes

- `src/iceaggr/models/flat_transformer_v2.py`; `src/iceaggr/data/dataset.py`
- `scripts/train_flat.py` (ablation, model-flag/RoPE/spacetime, fast-collate/tokenization hunks only)
- `scripts/inference.py` (RoPE/time-order and dict-output hunks only)
- `tests/unit/test_collators.py`; `tests/unit/test_flat_transformer_v2.py`
- `configs/train_flat_v2_none_K84_muon_combined_10ep.yaml`
- `configs/ablation_base.yaml`; `configs/ablation_base_rope.yaml`;
  `configs/ablation_base_rope_timeorder.yaml`; `configs/ablation_base_spacetime.yaml`
- `configs/enc_control_raw_K32_sigmoid.yaml`; `configs/enc_control_raw_K32_softplus700.yaml`;
  `configs/enc_control_raw_K32_unclamped.yaml`; `configs/enc_control_raw_K84_unclamped.yaml`;
  `configs/enc_ref_npe15_combined.yaml`; `configs/enc_ref_rawK84_combined.yaml`
- `configs/hybrid_muon_base.yaml`; `configs/hybrid_muon_base_mlr0075.yaml`;
  `configs/hybrid_muon_finetune.yaml`; `configs/hybrid_muon_finetune_mlr0075.yaml`
- `configs/muon_finetune_hi.yaml`; `configs/muon_sweep_adamw_ref.yaml`;
  `configs/muon_sweep_lr0p0025.yaml`; `configs/muon_sweep_lr0p005.yaml`;
  `configs/muon_sweep_lr0p01.yaml`; `configs/muon_sweep_lr0p04.yaml`;
  `configs/muon_sweep_lr0p08.yaml`
- `configs/npe15_muon_base.yaml`; `configs/npe15_muon_finetune.yaml`;
  `configs/npe15_muonft_2p5em4.yaml`; `configs/npe15_muonft_5em4.yaml`
- `configs/train_flat_v2_none_K169_d512_18M.yaml`;
  `configs/train_flat_v2_none_K169_d512_18M_finetune_hi.yaml`;
  `configs/train_flat_v2_none_K84_5M_rope_timeorder.yaml`;
  `configs/train_flat_v2_npe15_finetune_hi_md1024.yaml`;
  `configs/train_flat_v2_npe15corr_combined_100M2ep.yaml`;
  `configs/train_flat_v2_vmf_K3_5M_combined_finetune_hi.yaml`;
  `configs/train_flat_v2_vmf_K3_5M_sigmoid_finetune_hi.yaml`;
  `configs/train_flat_v2_vmf_K3_5M_unclamped_finetune_hi_md128.yaml`;
  `configs/train_flat_v2_vmf_K84_5M_spacetime.yaml`
- `scripts/ablation.sbatch`; `scripts/benchmark_dataloader.py`;
  `scripts/benchmark_inference.py`; `scripts/muon_sweep.sh`; `scripts/queue_100M2ep.sh`;
  `scripts/queue_hybrid_muon.sh`; `scripts/queue_hybrid_muon_mlr0075.sh`;
  `scripts/queue_muonft_probe.sh`; `scripts/queue_npe15_full.sh`; `scripts/queue_npe15_muon.sh`

### Commit 2 — scaling ladder (after fixes)

- `.gitignore`; remaining scaling hunks of `scripts/train_flat.py` and `scripts/inference.py`
- `src/iceaggr/models/vmf_loss.py`; `src/iceaggr/data/collators.py`;
  `src/iceaggr/data/collators_hybrid.py`; `src/iceaggr/data/collators_npe.py`
- `scripts/scaling_ladder.py`; `scripts/ladder_summary.py`; `scripts/queue_worker.sh`;
  `scripts/queue_worker.sbatch`; `scripts/a100_fleet_20260903.sh`
- `configs/ladder/A01_1M_d256L3_mlr005_s17.yaml`; `A02_1M_lin128L6_mlr005_s17.yaml`;
  `A03_1M_d256L12_mlr005_s17.yaml`; `A04_1M_d256L6_mlr005_s17.yaml`;
  `A05_1M_d384L6_mlr005_s17.yaml`; `A06_1M_d256L18_mlr005_s17.yaml`;
  `A07_1M_d256L6lin_mlr005_s17.yaml`; `A08_1M_d256L6_mlr005_s41.yaml`;
  `A09_1M_lin128L6_mlr005_s41.yaml` (all under `configs/ladder/`)
- `configs/ladder/B01_10M_d256L6_mlr005_s17.yaml`; `B02_10M_d256L12_mlr005_s17.yaml`;
  `B03_10M_d256L24_mlr005_s17.yaml`; `B04_10M_lin128L6_mlr005_s17.yaml`;
  `B05_10M_d256L6lin_mlr005_s17.yaml`; `B06_10M_d512L6_mlr005_s17.yaml`;
  `B07_10M_d256L24lin_mlr005_s17.yaml`; `B08_10M_d256L12lin_mlr005_s17.yaml`;
  `B09_10M_d256L24_mlr005_s41.yaml`; `B09a_10M_hyb_d256L6_s17.yaml`;
  `B09b_10M_hyb_d256L24_s17.yaml`; `B10_10M_d256L6_mlr005_s41.yaml`;
  `B11_10M_d256L6lin_mlr005_s41.yaml`; `B12_10M_d256L24_mlr0071_s17.yaml`;
  `B13_10M_d256L24_mlr0035_s17.yaml`; `B14_10M_d256L36_mlr005_s17.yaml`
- `configs/ladder/L1M_d256L6_e30_s1.yaml`; `configs/ladder/L1M_d512L6_e30_s1.yaml`;
  `configs/ladder/P01_1M_d256L6_mlr005_s17.yaml`;
  `configs/ladder/P02_1M_d256L24_mlr005_s17.yaml`;
  `configs/ladder/P03_1M_d256L24_mlr0025_s17.yaml`;
  `configs/ladder/P04_1M_d512L6_mlr005_s17.yaml`

Do **not** include `.claude/`, other `delegation/` trees, `docs/`, `references.bib`, runtime
`queue/`/`logs/`/`wandb/`, `__pycache__`, or five `delegation/enc_hiE/out/*.npz` files (~49 MB).
They are unrelated/local/generated; `.claude` contains a local allow-list and absolute paths. The
Group-1 shell/config artifacts are site-specific provenance, not portable box entrypoints.

## Could not verify

- CUDA/A100 Inductor bit identity, peak GPU memory, throughput, or 40-vs-80-GB batch fit (no GPU).
- End-to-end launch on the separate box, its repo/data/geometry paths, CPU count, RAM, or mounts.
- Recovery from real worker/host death and external W&B table behavior were not mutated/tested.
- Full old-vs-new dataset byte comparison was not run because it requires multi-GB duplicate loads;
  the unit suite covered the new path and code review found dtype/slice semantics preserved.

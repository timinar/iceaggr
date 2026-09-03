# Handoff for the Claude session on the 8×A100 box — N2-T2 model-scaling fleet (2026-09-03)

You are continuing a program run so far on the KU cluster (H100 + A4500s) by Inar and a
Claude session there. This box has no access to the cluster's `paper/`, `checkpoints/`, `logs/`
or Claude memory, so everything you need is in this file and in the repo on branch
`feat/vmf-unclamped-K3` (commits b9afcbe, c3faeae, fc30ce3 and later). Data and the repo are
on the box; pull first.

## 1. What the project is (30 seconds)
IceCube neutrino direction reconstruction (Kaggle "Neutrinos in Deep Ice"). Model N2-T2: one flat
transformer over DOM tokens (`src/iceaggr/models/flat_transformer_v2.py`), vMF mixture head
(`vmf_loss.py`), trained by `scripts/train_flat.py`. Metric: mean angular error (deg) over all
events ("bulk", floor-limited ~55°) and on bright events (≥1000 pulses, "hi-E", the paper's
headline). Splits: train = batches 1–650 (130M events), dev = batch 651 (200k), locked
validation = 652–655, TEST = 656–659 (**never train/select on 656–659**).

## 2. Why this fleet exists
The paper said "model scaling saturates" (5M vs 19M). That was an artefact: a 50k-event val
subset (noise > effect) plus a wide/AdamW/angular-loss scale-up. A 10M-event ladder under the
current recipe (Muon lr 0.005 + AdamW side 3e-4, vMF K=3 unclamped + 0.5·angular, bf16,
dropout 0.1, wd 0.01, K=84 raw tokens, max_doms 128, head_dim 32, ff = 4·d) shows **depth
scales**, 5 epochs, bs 1024, seed 17, dev = batch 651, final epoch:

| model | params | dev NLL | bulk° | median° | ≥1000 mean/med° | train–dev gap |
|---|---|---|---|---|---|---|
| d256-L6 (5M control) | 5.0M | 0.649 | 56.67 | 49.68 | 9.28/2.59 | +0.018 |
| d256-L12 | 9.7M | 0.596 | 56.32 | 49.18 | 8.57/2.31 | +0.024 |
| **d256-L24** | 19.2M | **0.579** | **56.20** | **48.91** | **8.35/2.19** | +0.029 |
| d512-L6 (paper's 19M shape) | 19.4M | 0.613 | 56.40 | 49.17 | 8.64/2.44 | +0.037 |
| d256-L6 + linear input | 5.06M | 0.624 | 56.43 | 49.15 | 8.82/2.55 | +0.020 |
| d128-L6 linear | 1.36M | 0.677 | 56.79 | 49.72 | 9.73/2.83 | +0.003 |

Depth beats width at matched 19M; no overfitting; a learned Linear(256→d) input ("linear
input") is worth −0.24° at 5M. The 130M fleet tests whether this holds at full data, for the raw
tokenization and for the **hybrid** tokenization (raw K'=80 pulses + 12 aggregate stats per DOM,
`data.tokenization: hybrid`, Linear 256→256 input) that produced the record hi-E model.

## 3. Setup on the box
```
git pull && git checkout feat/vmf-unclamped-K3 && uv sync
ls src/iceaggr/data/data_config.yaml      # gitignored; must point data.root at the box's copy
nvidia-smi -L; nproc; free -g              # 8 GPUs; the launcher sizes workers from cores
wandb login                                 # results go to wandb project 'iceaggr'
GEOM=<box path>/sensor_geometry_normalized.csv ./scripts/a100_fleet_20260903.sh all
```
`all` = `stageB` (eight 10M follow-ups, names B07…B14) + `enqueue` (eight 130M runs F01–F08)
+ `workers` (one queue worker per GPU). Jobs are plain scripts in `queue/a100/pending/`; workers
take them alphabetically, so the short 10M runs (2–10 h each) go first, then the 130M runs
(F02/F08 5M controls ≈ 1.5 days; L12 ≈ 3 d; L24 ≈ 5 d; L36 ≈ 8 d on one A100).
Before `workers`: if wandb shows B07/B08 already finished on the cluster's H100, delete those two
job files from `queue/a100/pending/`.

The launcher prints the derived CPU budget (loader workers per job, main-process thread cap)
and the depth-aware micro-batch (≈1.4 GB GPU memory per layer per 1024 events; effective batch
4096 for 130M, 1024 for the 10M jobs, via gradient accumulation). Check the first 10 minutes of
each job's log for `b/s` throughput and `nvidia-smi` memory; if a job OOMs, halve `--batch-size`
and double `--grad-accum` in its config and requeue.

## 4. Operating the queue
- Watch: `tail -f logs/queue_worker_a100_*.log`; per-run logs `logs/queue/<run>.<host>.gpuN.log`
  (lines "Epoch N done | … | nll … | >=200: … | >=1000: …" and "train-eval | … gap(val-train)").
- `ls queue/a100/{pending,running,done,failed}`; `touch queue/a100/STOP` pauses after current jobs.
- Crash recovery: restart `./scripts/a100_fleet_20260903.sh workers`; a worker requeues its own
  stale claim and every job resumes from its latest `checkpoints/ladder/<run>/epoch_XXX.pt`
  (optimizer, scheduler, RNG and scaler state are restored; verified to reproduce).
- Table of all ladder runs: `uv run python scripts/ladder_summary.py` (wandb tag scaling-ladder).
- Add a run: `uv run python scripts/scaling_ladder.py --help` (config generator + `--enqueue a100`).

## 5. Known pitfalls
- Dataloader workers must stay single-threaded (torch OpenMP pool is not fork-safe); the trainer
  already pins them. Never raise threads; scale by workers, within `cores / 8` per job.
- The hybrid collator is ~2× the CPU work of the raw one; it is the most likely to be
  loader-bound. If its `b/s` is far below the raw run's, that is why.
- `val/angular_error_rad` is the historical batch-mean metric; `val/angular_error_event_deg`,
  `val/nll` (pure per-event mixture NLL) and the slices are the ladder metrics. Select checkpoints
  by dev NLL; report the selected and the final epoch.
- Do not evaluate on 656–659. When a 130M run finishes, evaluate on the locked set
  652–655: `uv run python scripts/inference.py --checkpoint checkpoints/ladder/<run>/best.pt
  --output <run>_652_655.parquet --batch-range 652 655` (architecture, tokenization, fast_collate
  and bf16 are read from the checkpoint) and compare paired against F02 (raw) / F08 (hybrid).

## 6. What to report back (to Inar)
1. Per run: final and best-epoch dev NLL, bulk mean/median, ≥200 and ≥1000 slices, train–dev
   gap, throughput, peak GPU memory — as one table (ladder_summary.py) plus the locked-set
   numbers for finished 130M runs.
2. The two questions the fleet answers: (a) does depth still pay at 130M (F01/F04/F06 vs F02,
   seeds F05), and does the linear input add on top (F03)? (b) does it pay for the hybrid
   tokenization (F07 vs F08)?
3. Follow-on, once a 130M winner exists: the hi-E finetune (continue from the base on ≥1000-pulse
   events with max_doms 512, Muon lr 2.5e-4, 10 epochs, bs 512; templates
   `configs/hybrid_muon_finetune.yaml` / `configs/npe15_muonft_2p5em4.yaml`), then a single
   test evaluation on 656–659 with the selection rule frozen beforehand.

# vMF finetune (`v2-vmf-K84-5M-mix3-finetune-hi1000`) — analysis runbook

Drop-in commands to analyze the finetune after it completes.

## 0. Verify the finetune finished

```bash
ls -la checkpoints/v2-vmf-K84-5M-mix3-finetune-hi1000/
tail -15 logs/training_vmf_finetune_hi.log
```

Expect `best.pt` + `epoch_010.pt`, plus a `Training complete. Best val loss (NLL): …`
line. The wandb run is `polargeese/iceaggr/nwnmyvam`.

## 1. Run inference on the held-out test set (batches 656–659, ≥1000 pulses)

The finetune was trained with `max_doms=512` and `min_pulses=1000`, so inference
must match to compare fairly with the MAE finetune.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/inference_vmf.py \
    --checkpoint checkpoints/v2-vmf-K84-5M-mix3-finetune-hi1000/best.pt \
    --max-doms 512 \
    --min-pulses 1000 \
    --batch-size 64 \
    --output paper/predictions/our_predictions_vmf_hi1000_656_659.parquet
```

Expected: 9,074 events, runtime ~5–10 min. The parquet includes
per-component κ/weights, `kappa_eff`, and `angular_error_deg`.

Reference prior results on the same 9,074-event slice:

| Model | Mean | Median |
|---|---|---|
| DrHB B d64 (115.6M) | 11.04° | 3.32° |
| Ours MAE base (5M, max_doms=128) | 8.81° | 3.22° |
| Ours vMF base (5M, K=3, max_doms=128) | 7.72° | 2.81° |
| Ours MAE finetuned (5M, max_doms=512) | 7.30° | 1.96° |
| **Ours vMF finetuned (5M, K=3, max_doms=512)** | **?** | **?** |

If the vMF finetune clears 7.30° mean / 1.96° median, the vMF path is the new SOTA
for us. If it does not, the MAE finetune wins and we take the K=1+anneal experiment
more seriously.

## 2. Update the `error_vs_pulses_vmf` plot

Edit [paper/scripts/paper_error_vs_pulses_vmf.py](../paper/scripts/paper_error_vs_pulses_vmf.py)
to add a 5th curve for the vMF finetune. Minimal patch:

```python
# in PREDICTIONS dict, add:
'ours_vmf_hi': os.path.join(PROJECT_ROOT, 'paper', 'predictions',
                            'our_predictions_vmf_hi1000_656_659.parquet'),

# in main(), alongside the other loads:
df_vmf_hi = pl.read_parquet(PREDICTIONS['ours_vmf_hi'])
err_vmf_hi = angular_error_deg(df_vmf_hi, 'math')
np_vmf_hi  = df_vmf_hi['n_pulses'].to_numpy()

# in models list, add (after the vMF base entry):
{'errors': err_vmf_hi, 'n_pulses': np_vmf_hi, 'color': '#B07AA1', 'lw': 1.6,
 'label': 'K=84 vMF finetuned (5M, max_doms=512, ≥1000 pulses)', 'ls': '--'},
```

Run:

```bash
uv run python paper/scripts/paper_error_vs_pulses_vmf.py
```

Outputs `paper/698db891736c48c66b2fff40/figures/error_vs_pulses_vmf.{pdf,png}`.

## 3. κ-based selection analysis on the finetuned model

Two existing plot scripts can be re-pointed at the new parquet by editing their
`PREDICTIONS` path (one line each):

```bash
# In these two files, change `our_predictions_vmf_kappa_656_659.parquet`
# to `our_predictions_vmf_hi1000_656_659.parquet`:
#   paper/scripts/paper_error_vs_kappa.py
#   paper/scripts/paper_kappa_vs_pulses.py

uv run python paper/scripts/paper_error_vs_kappa.py
uv run python paper/scripts/paper_kappa_vs_pulses.py
```

Outputs (rename to avoid clobbering the base-model versions):

```bash
cd paper/698db891736c48c66b2fff40/figures/
mv error_vs_kappa.pdf   error_vs_kappa_finetune.pdf
mv error_vs_kappa.png   error_vs_kappa_finetune.png
mv kappa_vs_pulses.pdf  kappa_vs_pulses_finetune.pdf
mv kappa_vs_pulses.png  kappa_vs_pulses_finetune.png
```

Or, cleaner: duplicate each script to a `_finetune.py` variant that reads the
high-activity parquet and writes files with the `_finetune` suffix. Your call.

## What to look for

Compare finetune κ distribution to base vMF:

- Base vMF on ≥1000 pulses: κ_eff P50 = **70**, P90 = **111**
- If the finetune kept K=3 but one component continues to dominate (as in the
  base), you'll see similar qualitative behavior with κ's shifted upward — that's
  fine, just sharper calibration on the subset it was trained on.
- If κ_eff grew substantially on well-reconstructed events (say P50 into the
  200s or higher), the finetune opened up calibration headroom by effectively
  lowering the reg pressure on the mode.

For the κ-selection curve: see how the top-10% subset's median changes. The base
model's overall top-10% was 1.83°; the *finetuned* top-10% of *high-activity*
events should be considerably sharper — probably below 1°.

## 4. Update memory

Record the finetune result in `MEMORY.md` under the "vMF base model (completed
2026-04-23)" section, or replace the "In-flight work" section with the completed
finetune's headline numbers.

## 5. If the experiment worked, next step

Launch the K=1 + annealed-reg pretrain from branch `feat/vmf-k1-anneal`:

```bash
git checkout feat/vmf-k1-anneal
screen -dmS vmf_k1_pretrain bash -c '
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_flat.py \
      --config configs/train_flat_v2_vmf_K1_5M_anneal.yaml 2>&1 \
      | tee logs/training_vmf_K1_anneal.log'
```

Expected ~12–18 h for 10 epochs at bs=4096.

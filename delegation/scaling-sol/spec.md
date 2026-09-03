# Task: diagnose why model scaling "failed" for N2-T2 and design a systematic scaling program

You are the sparring partner for a scaling study. Two deliverables, written to
`delegation/scaling-sol/plan.md` (≤160 lines, dense, no fluff):

1. **Diagnosis critique.** Below is my (Claude's) reading of why 4× more parameters gave
   nothing. Attack it: which hypotheses are wrong, what discriminating cheap test would you run
   first, and what is the single most likely reason. Be concrete and quantitative.
2. **Scaling program.** A ladder that starts on 1M training events, then 10M, then decides the
   130M run — so that we *find the recipe* under which bigger models beat 5M *on the same data*.
   Exact sizes, per-size LR handling, batch size, epochs, regularization, metrics, seeds, decision
   rules, GPU-hours. End with the TOP-3 runs to launch in the next hour with exact config deltas.

Work from the repo (root = working dir). Do NOT run training. Read what you need:
`src/iceaggr/models/flat_transformer_v2.py`, `scripts/train_flat.py`, `src/iceaggr/utils/muon.py`,
`configs/train_flat_v2_none_K84_muon_combined_10ep.yaml` (the current best recipe),
`configs/train_flat_v2_none_K169_d512_18M.yaml` (the failed 19M scale-up),
`paper/698db891736c48c66b2fff40/main.tex` lines 1100–1125 (the paper's saturation claim).

## The setup
- Task: IceCube neutrino direction reconstruction (Kaggle "Neutrinos in Deep Ice"). Metric: mean
  angular error (rad/deg) over all events; we also care about the ≥1000-pulse (bright) slice.
- Model N2-T2: ONE flat transformer over DOM tokens. Each active DOM → one token =
  [x,y,z,log n_pulses, (t,q,aux)×K], K=84 first pulses, so input_dim=4+3K=256=d_model with
  `input_mode: none` (identity, no learned embedding; parameter-free RMSNorm right after). Only the
  first max_doms=128 DOMs (earliest hit) are kept. CLS token → head. nanochat-style blocks
  (RMSNorm no params, QK-norm, ReLU², zero-init projections, per-layer resid/x0 scalars), L=6,
  H=8, dff=1024, dropout 0.1 (also attention dropout), weight decay 0.01, grad-clip 1.0,
  OneCycle (warmup 2000 steps, cos, div 25, final_div 1000), torch.compile, bf16.
- Params: d256/L6 = 5.0M. d512/L6/dff2048 = 19.4M. Per block at d256 ≈ 0.79M.
- Data: 130M train events (batches 1–650), val = batches 651–655 (1M events; ≥200 pulses 5.8%,
  ≥1000 pulses 1.13% = 11.2k events), test = 656–659. `max_events: N` takes the FIRST N events
  (batches 1..k), so a 1M-event subset = batches 1–5.
- Best current recipe (bulk 55.60°/47.91° test): vMF K=3 mixture head, unclamped softplus κ,
  combined loss NLL + 0.5·angular, **Muon** on block 2D matrices (lr 0.005, momentum 0.95,
  NS5, update scaled by sqrt(max(1,rows/cols))) + AdamW side group (CLS/head/scalars, lr 3e-4),
  bs 4096, 10 epochs, bf16. Muon LR sweep at 5M-events/2ep: 0.0025:58.02, 0.005:58.00,
  0.01:58.50, 0.04:61.72, 0.08:62.05, AdamW ref 59.11 (default 0.02 LOSES).
- Hi-E finetune recipe: continue from base on ≥1000-pulse events with max_doms 512, 10 ep,
  lr 3e-5 AdamW (or Muon 2.5e-4), bs 512. Best single hi-E: 5.71°/0.95° (hybrid tokens).
- Compute: 1× H100 96GB (free now); 5M model at bs4096 ≈ 19k events/s incl. dataloader
  (130M×10ep = 19h); the 19M/K169 model ran ≈ 9k ev/s (33h). 10M×10ep of the 5M ≈ 1.5–2.4h.
  Possibly 2× A4500 20GB (slurm, contended) and 2× RTX3090 (not yet reachable).

## What was tried (all AdamW + plain angular-distance "MAE" loss, pre-vMF/pre-Muon recipe,
## 130M events × 10 epochs, val = first 50k events of 651–655)

| run | params | recipe | ep10 train loss (dropout on) | ep10 val loss | val ° |
|---|---|---|---|---|---|
| d256 L6 K84 none (5M, the paper model) | 5.0M | bs4096 lr3e-4 | 0.9741 | 0.9681 | 55.47 |
| same + RoPE (time-ordered) | 5.0M | same | 0.9726 | 0.9683 | 55.48 |
| d256 L6 K41 none, 10 ep | 5.0M | same | 0.9749 | 0.9690 | 55.52 |
| d256 L6 K41 none, **20 ep** | 5.0M | same | 0.9699 | 0.9693 | 55.54 |
| d384 L6 K41 **linear** proj (Feb) | ~10M | bs4096 lr2e-4 | 0.9737 | 0.9721 | 55.70 |
| d512 L6 K169 none dff2048 (June) | 19.4M | bs2048 lr2e-4 (2-ep sweep 1.5/2/3e-4) | 0.9728 | 0.9704 | 55.60 |

- Hi-E finetune (val, ≥1000): 5M-MAE ft 7.54°, 19M-MAE ft 7.43° (19M is 0.11° BETTER on val).
  The paper compares 7.43 (val) against 7.30 (the 5M's TEST number) and calls it saturation —
  an apples/oranges error I am fixing (test eval of the 19M-ft running now). The 19M BASE was
  never evaluated on the hi-E slice at all.
- Kaggle context: winning ensembles ≈ 0.959–0.960 rad (≈55.0°) on private LB; DrHB 116M
  per-pulse model 55.25° on our test where our 5M is 55.72° test. So the whole bulk-metric
  headroom below our 5M is ≲0.5–0.75°, i.e. ≲1.3% of the metric.
- Data scaling at 5M params: 10M events→57.5°, 130M→55.5° (matched 10 epochs).
- Width of the raw token: K=84 no better than K=32 at 10M events (raw needs data to use width).
- Small-scale ablation noise: full bundle 58.55±0.02° (n=4 seeds) at 10M events; zero-init
  projections are load-bearing (removing → 62±2.3, unstable).
- Finetune decomposition on hi-E: specialization −0.26°, raising the DOM cap 128→512 −0.48°.
- Even at 10M events × 10 ep the 5M model does not overfit (train 1.228 vs val 1.220 combined
  loss, dropout on).

## My (Claude's) diagnosis — attack this
H1 **Metric floor, not capacity floor.** Train loss is *flat* across 5M/10M/19M (0.974/0.974/0.973)
   and train≈val everywhere: bigger models did not fit the training set better. The bulk MAE is
   dominated by dim, information-free events; the achievable floor is ~0.96 rad, so a 4× model
   has <1% of metric to win and it cannot be resolved. Scaling must be measured on a sensitive
   metric: vMF NLL (proper score) + bright slices, not the bulk mean.
H2 **The 19M was a bad scale-up.** It changed K 84→169 (known useless width: K84≈K32 at 10M),
   kept depth at 6, coupled width to the sparse raw token via `none`, used a 3-point 2-epoch LR
   sweep, AdamW, MAE loss, bs 2048, single seed. Depth was never scaled at all.
H3 **Input-representation bottleneck.** max_doms=128 + first-K pulses caps the information any
   model gets; the finetune's DOM-cap gain (−0.48°) is bigger than any capacity effect seen.
H4 **Optimization not tuned per size.** AdamW lr 2e-4 for the 19M may be far from optimal for
   10 epochs; no muP/width-transfer; no seeds so ±0.1° differences are uninterpretable.
H5 (weaker) 10 epochs of repeated data + dropout 0.1 + wd 0.01 may be the wrong regularization
   for both sizes; the 20-epoch run gaining nothing suggests we are step/epoch-saturated too.

Questions I want answered explicitly:
- Which single cheap experiment best discriminates H1 (floor) from H4/H2 (bad optimization)?
  (I am already computing per-pulse-count-slice errors of the 5M vs 19M bases on val AND on
  400k train events in eval mode.)
- Ladder design: width via `input_mode: none` with zero-padding of the 256-dim raw token to
  d_model (supported: `_pad_input`), vs `linear` projection, vs keep d=256 and scale depth
  only (L 6→12→18), vs both. What sizes (≈1.3M, 2.5M, 5M, 10M, 20M, 40M?) and shapes?
- LR handling per size under Muon (does Muon's spectral scaling give width transfer here? what
  about depth?) and for the AdamW side group; how many LR points; batch size at 1M events
  (bs 4096 → only 2.4k steps in 10 epochs); epochs at 1M vs 10M; warmup.
- Regularization at 1M events (dropout/wd) so that "bigger overfits" is not misread as "bigger
  doesn't help"; whether to report best-epoch or final; how many seeds for the noise floor.
- Metrics: NLL vs angular vs slices (val is 1M events; ≥1000 slice = 11k events; first 200k
  events give 2.3k ≥1000 events); which to make the primary scaling metric and why.
- The decision rule from 1M→10M→130M, and what result would justify the 40h 130M run.
- Anything better than a ladder (e.g. iso-FLOP frame, train-loss-on-bright-slice probe,
  learned per-DOM embedding, raising max_doms in the base, deeper-not-wider).

Deliverable: `delegation/scaling-sol/plan.md`. Ranked, with GPU-hours and expected effect sizes
with reasoning. End with TOP-3 to launch now (exact YAML deltas vs
`configs/train_flat_v2_none_K84_muon_combined_10ep.yaml`).

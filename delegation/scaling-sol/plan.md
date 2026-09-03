# N2-T2 scaling diagnosis and program

## Verdict and critique

**Single most likely explanation:** the 19.4M result is an obsolete, confounded optimization/architecture point, not evidence of a capacity floor. It widened the trunk and raw token together, kept depth fixed, halved batch size, changed LR, and used AdamW+angular loss; the successful Muon+vMF-combined recipe has never been capacity-scaled. Bigger models may still fail, but the existing evidence cannot decide that.

1. **H1 (“metric floor”) is overstated and its train-loss argument is invalid.** `train loss` is an online epoch average under dropout while weights change; `val loss` is an end-state eval-mode average on other events. Their equality is not a generalization gap. The 0.001-rad differences called “flat” are 0.057°, already the scale of interest. Evaluate both checkpoints, in eval mode, on identical train events (the running 400k analysis) before discussing fit.
2. A floor near 0.96 rad is not established: 55.60° is 0.9704 rad and 55.0° is 0.9599 rad, leaving about 0.60°/0.0105 rad. Kaggle/private-LB and local test values are contextual, not a Bayes-floor estimate, but they disprove “nothing resolvable.” On 1M common events even the worst-case bounded SD gives SE <=0.09°; empirical paired SE should be smaller. Training-seed noise, not event-sampling noise, is the limiting uncertainty.
3. **H2 is mostly right, but K=169 is not proven useless.** K84≈K32 at 10M does not prove K169 is useless at 130M; it proves that changing K makes the size comparison uninterpretable. `none` pads 511→512 by one coordinate. Keeping H=8 also changed head width 32→64. First isolate capacity with K84 fixed and head width 32 fixed.
4. **H3 is plausible for bright events, not an explanation of the bulk null.** The −0.48° cap effect was measured after specialized high-E finetuning, not in a matched base comparison. If it affected only the 1.13% bright slice, its contribution to the bulk mean is only 0.005° (even across all >=200 events, 5.8%×0.48°=0.028°). Test max_doms separately from parameter scaling.
5. **H4 is plausible but “2e-4 was far from optimal” is unproven.** The three-point sweep had an interior winner, but two epochs can rank warm-up/transient behavior rather than 10-epoch endpoints. More importantly, Muon improved the 5M two-epoch result by 1.11° over AdamW—larger than all observed size effects—so optimizer/loss transfer must precede a saturation claim.
6. **H5 is mild overfit, not step saturation.** The K41 20-epoch model lowered dropout-on train loss by 0.0050 rad (0.29°) while val worsened 0.0003 rad (0.02°). More epochs did fit train better; they did not generalize. A OneCycle rerun with a different horizon is also not “continue at the LR floor.”
7. Implementation caveats: `validate()` reports combined loss, not pure vMF NLL; it averages batch means rather than events; `best.pt` is selected on that combined score; no config seed is consumed; Muon weight decay is applied to block matrices too; linear input projections are placed in the AdamW side group. Fix measurement/seeding before making scaling claims.

## First discriminating cheap experiment

Run a **three-arm controlled 1M scaling trial** on the same first 1M events: 5.00M d256/L6 at Muon LR .005 versus 19.15M d256/L24 at .005 and .0025; all K84/none/H8/dff1024, current dropout=.10/WD=.01/κ schedule/vMF-combined recipe, effective BS=1024, 20 epochs (19,540 updates), side LR 3e-4, warm-up=1,000, seed 17. Cost: about 2–3 H100-hours including fixed-train/dev inference. This changes capacity cleanly and asks whether the successful optimizer/objective makes it useful; another retrospective slice analysis cannot answer that.

Evaluate every saved epoch on the identical 400k train events and a fixed held-out set, in eval mode, with pure NLL and per-event angular error. Interpret before any full ladder: (a) lower 19M train and val loss => H2/H4 won; (b) lower train but not val => capacity exists and regularization/data are binding, so run the regularization branch; (c) no train gain at either LR => this depth parameterization is not using capacity (optimization/architecture). A null does not prove H1; only several shapes/LRs reaching the same end-state empirical risk would make a task/representation floor credible. The already-running old-checkpoint slices remain useful localization evidence.

## Measurement contract (implement once, then freeze)

- Add `training.seed`; before model construction seed Python/NumPy/Torch/CUDA and pass the seed to `BatchAwareSampler` (keep data order common across paired runs). Use seeds **17, 41, 73**. Current YAML seed keys would silently do nothing.
- Emit per-event **pure mixture NLL** (no angular term, κ regularizer, or batch averaging), angular error, event id, pulse count, and DOMs retained. Save every epoch at 1M; never select on the locked set.
- Development set: batch 651 (~200k) for LR/epoch selection. Locked promotion set: batches 652–655 (~800k); test 656–659 remains untouched until the 130M winner. Always report the full 651–655 result as a descriptive cross-check.
- Primary scaling outcome is locked-set **paired mean angular-error delta**, because it is the actual target and 800k events resolve <<0.1°. Pure NLL is the smoother optimization/model-selection diagnostic, not a substitute: it can improve confidence without improving direction.
- Report pure NLL, mean/median angle, and disjoint pulse slices `<200`, `200–999`, `>=1000`, plus cumulative `>=200`; include counts. Use all ~11.2k bright validation events for final slice claims, not the noisy ~2.3k in the first 200k.
- Select one checkpoint per run by minimum dev pure NLL at fixed once-per-epoch cadence. Report that checkpoint **and final epoch**; promotion uses the selected checkpoint on locked data. This permits fair early stopping without cherry-picking locked angular error.
- Save aligned predictions; give paired event-bootstrap 95% CIs (10k resamples) per seed and mean±SD of seed deltas. Bootstrap does not replace seed replication. Record throughput, peak memory, gradient norm/clipping rate, LR, train-eval NLL, and generalization gap.

## Architecture ladder (all raw K84, max_doms=128, head hidden=1024, vMF K=3)

| Params | d/L/dff/H | input | role |
|---:|---|---|---|
| 1.360M | 128/6/512/4 | linear 256→128 | lower anchor; representation-confounded |
| 2.638M | 256/3/1024/8 | none | clean shallow anchor |
| 4.997M | 256/6/1024/8 | none | mandatory matched control |
| 9.716M | 256/12/1024/8 | none | primary clean depth scale |
| 14.435M | 256/18/1024/8 | none | depth interpolation |
| 19.153M | 256/24/1024/8 | none | 4× clean depth endpoint |
| 11.027M | 384/6/1536/12 | none, pad 128 zeros | width control; head dim stays 32 |
| 19.416M | 512/6/2048/16 | none, pad 256 zeros | matched-size width control |
| 38.290M | 512/12/2048/16 | none, pad 256 zeros | width+depth only after a trend |

Depth at d=256 is the primary ladder because it preserves the raw token and optimizer groups. Width is a factorial shape test, not a replacement: compare 19.153M depth with 19.416M width. Add exactly one learned-embedding control per surviving width (`input_mode: linear`: 11.126M at d384/L6 or 19.547M at d512/L6); it is a representation/AdamW-side-group change, so do not merge it into the capacity curve. Do not scale K again until capacity is understood.

## Optimization, regularization, stages, and cost

Muon's orthogonal update and aspect-ratio factor make width transfer at fixed matrix aspect ratio plausible, not guaranteed: the code has no absolute-width or depth scaling, and the AdamW CLS/head/input side group has neither. This is not muP; claiming width transfer requires the sweep. Start every shape at Muon 0.005 to test transfer, then use coordinate sweeps, never a Cartesian grid.

- Muon LR, maximum three points/serious size: L<=12 or width-L6 `{0.0035,0.0050,0.0071}`; L18/L24 or d512/L12 `{0.0025,0.0035,0.0050}`. Screen seed 17; repeat only the winner. For the best Muon LR, side-LR sweep: d<=256 `{2.0,3.0,4.5}e-4`; d384 `{1.6,2.4,3.6}e-4`; d512 `{1.4,2.1,3.2}e-4` (centers use sqrt(256/d)). Thus at most five coordinate points including the anchor, not nine.
- Keep momentum=.95, NS=5, grad clip=1, bf16, OneCycle div=25/final_div=1000. Use warm-up `min(2000, 5% of optimizer steps)`: 1,000 at 1M, 1,250 at 10M, 2,000 at 130M. Log clipping; if >10% of updates clip, tune clip/LR before calling the model saturated.
- Regularized recipe starts at dropout=.10/WD=.01 for every size. If larger lowers train-eval NLL by >=0.01 nat but loses locked validation, test **light** (.05/.003) and **strong** (.20/.03) on both it and the 5M control. Choose by dev NLL; do not give only the large model extra search. If end-state train loss is still indistinguishable, a separate 100k-event dropout=WD=κ-reg=0 memorization audit may test optimization, but that result is never a promoted recipe.
- **Stage A—1M:** effective BS=1024, 20 epochs/20M examples; physical BS=1024, except d512/L12 BS=512+accum2. Run all anchor shapes once at seed17; LR-bracket only candidates with lower train NLL or a dev signal; repeat 5M and the best depth/width candidate at seeds41/73. Anchor screen ~4.5–7 H100-h; tuning+replication cap ~8–14 h.
- Stage-A gate: promote a larger shape if its three-seed mean locked delta is <=−0.10° with at least 2/3 seed wins, **or** pure NLL improves >=0.01 nat with angle non-inferior (upper paired CI <+0.05°). Bright improvement >=0.20° is a tie-breaker, not a substitute for the stated bulk goal.
- **Stage B—10M:** matched 10 epochs/100M examples, effective BS=4096; physical BS4096 for d256, BS2048+accum2 for d384/d512-L6, BS1024+accum4 for d512/L12. Run 5M plus at most two promoted shapes at seed17, re-bracket Muon by one point each side, then seeds41/73. Expected per run: 5M 1.5–2.4 h, 10–11M 2–3.5 h, 19M 2.5–4 h, 38M 4.5–7 h; stage cap ~20–35 H100-h.
- Stage-B→130M gate: larger must beat matched 5M across seeds by >=0.10° mean (>=2/3 wins; seed-mean 95% t-CI upper <0), improve pure NLL >=0.01 nat, and have bright-slice paired-CI upper <+0.20°. A 0.10° gain is ~17% of the estimated 0.60° headroom and ~5× the observed 0.02° seed SD: material enough to buy the run.
- **Stage C—130M:** one frozen winner, 10 epochs, effective BS4096, selected regularization/LRs, seed17; also run a d256/L6 seed17 control under the identical schedule unless the existing checkpoint's seed/order is verifiably identical. Expected: 5M ~19 h; d512/L6 19M ~33–40 h; d512/L12 38M ~55–70 h. Replicate the large run only if its locked-val gain survives and test gain is >=0.10°; otherwise stop.
- Primary curves are iso-data/iso-exposure. Also plot validation versus measured H100-hours and read each run at the 5M wall-clock budget (iso-compute); parameter count is a poor FLOP proxy across depth/width. Do not shorten the primary large runs to equal FLOPs—the question is whether capacity helps on the same data.

## Ranked adjacent experiments

1. **Clean depth/width factorial above**—highest value; it directly repairs the failed comparison. Expected detectable bulk gain is only 0.05–0.20° and bright gain 0.1–0.4°, so paired evaluation and seeds are mandatory.
2. **Learned per-DOM projection at fixed trunk**—compare d256/L6 none vs linear (5.063M) and, only if useful, the surviving width. Expected 0–0.15° bulk; it lets width use all channels but moves the projection to AdamW and can erase raw-feature geometry.
3. **Base max_doms 128→256 factorial at 5M and the winner** on 10M, with effective batch held fixed. Expect little bulk change (roughly 0–0.05° by slice weighting) but possibly 0.2–0.5° on >=1000; treat this as information scaling, not model scaling. Test 512 only in bright specialization because attention/memory cost is 16× versus 128.
4. Stratify a future 1M proxy across batches 1–650: first-five-batch results can select a batch-specific recipe. Preserve the current first-1M ladder for immediate controlled comparisons, then require the 10M stage before extrapolation.

## TOP-3 runs to launch now (deltas from `train_flat_v2_none_K84_muon_combined_10ep.yaml`)

These are the controlled trial, ranked 1→3; first add functional seeding and pure-NLL/train-eval logging. Common deltas: `training: {epochs: 20, batch_size: 1024, grad_accum_steps: 1, lr: 3e-4, warmup_steps: 1000, seed: 17}`, `data: {max_events: 1000000, val_events: 200000, val_per_epoch: 1}`, `checkpoint.save_every: 1`; retain dropout=.10, WD=.01, κ regularization/annealing, K84/none/max_doms128/vMF-combined/Muon momentum/bf16.

1. **probe-5M-L6-mlr005:** retain `d_model:256, num_layers:6, hidden_dim:1024, num_heads:8, muon_lr:0.005`; set `wandb.name` accordingly. Control; ~0.3–0.5 H100-h.
2. **probe-19M-L24-mlr005:** change `model.num_layers:24`, retain d256/dff1024/H8, `training.muon_lr:0.005`; ~0.8–1.2 h. Direct depth-transfer test.
3. **probe-19M-L24-mlr0025:** same as #2 but `training.muon_lr:0.0025`; ~0.8–1.2 h. Depth-LR discriminator; without it, a flat #2 is ambiguous.

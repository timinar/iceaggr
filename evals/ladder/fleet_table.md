# N2-T2 scaling fleet on the A100 box — run definitions and results (as of 2026-09-18 08:05 UTC (program complete, test evaluated))

Shared recipe (every run below unless a column says otherwise):
- Data: IceCube Kaggle batches 1–650 for training; dev = batch 651 (first 200k events); locked evaluation set = batches 652–655 (800k events); batches 656–659 never touched.
- Model: flat transformer over DOM tokens, d_model 256, 8 heads (head_dim 32), FFN 1024, max 128 DOMs per event, dropout 0.1, RMSNorm, QK-norm, ReLU², zero-init projections.
- DOM token: "raw" = first 84 pulses (time, charge, aux) + x,y,z,n_pulses = 256 numbers; "hybrid" = first 80 pulses + 12 aggregate statistics per DOM (the record hi-E model's tokenization). Input embedding: "none" = the 256-d token enters the residual stream directly; "learned linear" = a trained 256→256 linear map first.
- Head and loss: 3-component von Mises–Fisher mixture on the CLS token, unclamped κ (softplus), κ² penalty annealed 1e-4 → 0 over the first 30% of steps; loss = mixture NLL + 0.5 × angular distance of the κ-weighted mean direction.
- Optimizer: Muon (Newton–Schulz orthogonalized momentum SGD, momentum 0.95, weight decay 0.01) on all block weight matrices, AdamW (lr 3e-4, weight decay 0.01) on CLS/head/scalars/input map; OneCycle schedule with 2000 warmup steps; bf16 autocast; gradient clip 1.0.
- 130M-event runs: 10 epochs, effective batch 4096 (micro-batch by depth × gradient accumulation), 5 validations per epoch, epoch checkpoints, best checkpoint = lowest dev combined loss (for every finished run so far this was the final epoch).
- 10M-event runs (stage B): first 10M training events, 5 epochs, effective batch 1024, 2 validations per epoch.

Metric columns: "bulk" = mean angular error over all events; "≥1000" = mean angular error over events with ≥1000 pulses (the paper's hi-E headline); NLL = pure mixture NLL on the dev set.

## 130M-event runs (10 epochs, effective batch 4096)

| run | layers | params | DOM tokenization | input embedding | Muon lr | seed | status | dev at last epoch: NLL / bulk° / ≥1000° | locked 652–655: mean° / median° / ≥1000° |
|---|---|---|---|---|---|---|---|---|---|
| F02 | 6 | 5.0M | raw | none | 0.005 | 17 | finished | 0.4995 / 55.71 / 6.76 | 55.60 / 47.96 / 6.64 |
| F12 | 6 | 5.0M | raw | none | 0.005 | 41 | finished | 0.5044 / 55.73 / 6.93 | 55.65 / 48.06 / 6.63 |
| F14 | 6 | 5.0M | raw | none, **dropout 0** | 0.005 | 17 | finished | 0.4873 / 55.65 / 6.43 | 55.54 / 47.86 / 6.41 |
| F11 | 6 | 5.1M | raw | learned linear | 0.005 | 17 | finished | 0.4850 / 55.61 / 6.23 | 55.50 / 47.85 / 6.22 |
| F04 | 12 | 9.7M | raw | none | 0.005 | 17 | finished | 0.4730 / 55.63 / 6.47 | 55.52 / 47.83 / 6.24 |
| F01 | 24 | 19.2M | raw | none | 0.005 | 17 | finished | 0.4659 / 55.52 / 6.40 | 55.45 / 47.77 / 6.09 |
| F05 | 24 | 19.2M | raw | none | 0.005 | 41 | finished | 0.4676 / 55.62 / 6.24 | 55.50 / 47.83 / 6.11 |
| F09 | 24 | 19.2M | raw | none | 0.0035 | 17 | finished | 0.4655 / 55.46 / 6.15 | 55.43 / 47.74 / 6.12 |
| F03 | 24 | 19.2M | raw | learned linear | 0.005 | 17 | finished | 0.4546 / 55.46 / 5.90 | 55.36 / 47.63 / 5.85 |
| F06 | 36 | 28.6M | raw | none | 0.005 | 17 | finished | 0.4685 / 55.60 / 6.19 | 55.49 / 47.83 / 6.13 |
| F08 | 6 | 5.1M | hybrid | learned linear | 0.005 | 17 | finished | 0.4960 / 56.19 / 6.10 | 56.06 / 48.60 / 5.87 |
| F13 | 6 | 5.1M | hybrid | learned linear, **dropout 0** | 0.005 | 17 | finished | 0.4834 / 56.10 / 6.03 | 56.00 / 48.52 / 5.80 |
| F16 | 6 | 5.1M | hybrid | learned linear | 0.005 | 41 | finished | 0.4958 / 56.17 / 6.11 | 56.08 / 48.58 / 5.88 |
| F15 | 12 | 9.8M | hybrid | learned linear | 0.005 | 17 | finished | 0.4691 / 56.02 / 5.69 | 55.92 / 48.36 / 5.58 |
| F07 | 24 | 19.2M | hybrid | learned linear | 0.005 | 17 | finished | 0.4645 / 56.07 / 5.64 | 55.95 / 48.44 / 5.61 |
| F10 | 24 | 19.2M | hybrid | learned linear | 0.005 | 41 | finished | 0.4619 / 55.98 / 5.61 | 55.91 / 48.33 / 5.54 |
| F17 | 24 | 19.2M | hybrid | learned linear | 0.0035 | 17 | finished | 0.4636 / 56.00 / 5.77 | 55.91 / 48.37 / 5.54 |
| F18 | 24 | 19.2M | hybrid | learned linear, **dropout 0** | 0.005 | 17 | finished | 0.4624 / 55.99 / 5.81 | 55.90 / 48.37 / 5.61 |
| F19 | 24 | 19.2M | raw | learned linear, **dropout 0** | 0.005 | 17 | finished | 0.4496 / 55.48 / 5.85 | 55.38 / 47.66 / 5.79 |
| F20 | 24 | 19.2M | raw | learned linear | 0.005 | 41 | finished | 0.4466 / 55.42 / 5.76 | 55.32 / 47.58 / 5.72 |

## 10M-event runs (5 epochs, effective batch 1024), final epoch on dev

| run | layers | d_model | params | DOM tokenization | input embedding | Muon lr | seed | NLL | bulk° | median° | ≥1000° | train–dev gap | where |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B01 | 6 | 256 | 5.0M | raw | none | 0.005 | 17 | 0.649 | 56.67 | 49.68 | 9.28 | 0.016 | cluster |
| B10 | 6 | 256 | 5.0M | raw | none | 0.005 | 41 | 0.647 | 56.65 | 49.55 | 9.17 | 0.019 | cluster |
| B05 | 6 | 256 | 5.1M | raw | learned linear | 0.005 | 17 | 0.624 | 56.43 | 49.15 | 8.82 | 0.018 | cluster |
| B11 | 6 | 256 | 5.1M | raw | learned linear | 0.005 | 41 | 0.626 | 56.44 | 49.22 | 9.06 | 0.020 | cluster |
| B04 | 6 | 128 | 1.4M | raw | learned linear | 0.005 | 17 | 0.677 | 56.79 | 49.72 | 9.73 | 0.002 | cluster |
| B06 | 6 | 512 | 19.4M | raw | none (zero-padded) | 0.005 | 17 | 0.613 | 56.41 | 49.17 | 8.64 | 0.034 | cluster |
| B02 | 12 | 256 | 9.7M | raw | none | 0.005 | 17 | 0.596 | 56.32 | 49.18 | 8.57 | 0.022 | cluster |
| B08 | 12 | 256 | 9.8M | raw | learned linear | 0.005 | 17 | 0.577 | 56.12 | 48.77 | 8.29 | 0.022 | cluster |
| B03 | 24 | 256 | 19.2M | raw | none | 0.005 | 17 | 0.579 | 56.20 | 48.91 | 8.35 | 0.027 | cluster |
| B09 | 24 | 256 | 19.2M | raw | none | 0.005 | 41 | 0.578 | 56.11 | 48.82 | 8.23 | 0.030 | this box |
| B13 | 24 | 256 | 19.2M | raw | none | 0.0035 | 17 | 0.574 | 56.11 | 48.86 | 8.34 | 0.027 | this box |
| B12 | 24 | 256 | 19.2M | raw | none | 0.0071 | 17 | 0.594 | 56.37 | 49.27 | 8.61 | 0.026 | this box |
| B07 | 24 | 256 | 19.2M | raw | learned linear | 0.005 | 17 | 0.569 | 56.16 | 49.05 | 8.08 | 0.028 | cluster |
| B14 | 36 | 256 | 28.6M | raw | none | 0.005 | 17 | 0.582 | 56.27 | 49.15 | 8.34 | 0.032 | this box |
| B09a | 6 | 256 | 5.1M | hybrid | learned linear | 0.005 | 17 | 0.634 | 56.83 | 49.96 | 7.97 | 0.018 | this box |
| B09b | 24 | 256 | 19.2M | hybrid | learned linear | 0.005 | 17 | 0.585 | 56.70 | 49.79 | 7.58 | 0.030 | this box |

## Hi-E finetunes (continue from a 130M base on ≥1000-pulse events only: 1.46M training events, 512 DOMs per event, Muon lr 2.5e-4 / AdamW 3e-5, 500 warmup steps, 10 epochs, effective batch 512, κ penalty off; dev = the 2,258 bright events of batch 651)

Locked-set numbers are on the 8,996 events of batches 652–655 with ≥1000 pulses; "vs base" is the paired difference on those same events.

| run | base model | selected epoch (dev loss) | dev ≥1000 mean° / median° (selected / final) | locked ≥1000 mean° / median° (selected / final) | vs its base | vs hybrid L24 base (F07) |
|---|---|---|---|---|---|---|
| F03ft | F03: raw, 24 layers, linear input | 9 | 5.57 / 0.81 · 5.54 / 0.81 | 5.19 / 0.90 · 5.19 / 0.89 | −0.66 ± 0.05 | −0.42 ± 0.07 |
| F03ft (seed 41) | F03 base weights, finetune seed 41 | 9 | 5.51 / 0.81 · 5.50 / 0.81 | 5.17 / 0.89 · 5.17 / 0.90 | — | vs F03ft seed 17: −0.02 ± 0.02 (finetune seed noise ≈ 0.02°); vs F08ft (record recipe): −0.12 ± 0.05 |
| F19ft (≥1000) | F19: raw 24 + linear, **no dropout** | 9 | 5.48 / 0.83 · 5.51 / 0.82 | 5.23 / 0.88 · 5.21 / 0.87 | −0.56 ± 0.05 | vs F03ft (dropout-0.1 twin): +0.04 ± 0.04 selected, +0.03 ± 0.04 final → no dropout does not help the raw 24-layer finetune either |
| F19ft (≥500 set) | F19: raw 24 + linear, no dropout; final recipe (≥500) | 7 | 10.20 / 1.43 on ≥500 dev (≥1000 subset 5.55) | 5.24 / 0.89 · 5.24 / 0.88 | −0.55 ± 0.05 | vs F03ft ≥500 (dropout-0.1 twin): ≥1000 +0.06 ± 0.04, 200–999 +0.09 / +0.04 ± 0.06; vs F18ft ≥500 (hybrid): ≥1000 +0.06 ± 0.04, 200–999 −0.10 / −0.15 ± 0.07 |
| F20ft (≥500 set) | F20: raw 24 + linear, seed 41; final recipe (≥500) | 10 | 10.19 / 1.37 on ≥500 dev (≥1000 subset 5.39) | 5.15 / 0.85 | −0.57 ± 0.05 | vs F03ft ≥500 (seed-17 twin): ≥1000 −0.02 ± 0.04, 200–999 +0.01 ± 0.06 → seed pair of the headline two-model system agrees; vs hybrid ≥500 finetune: ≥1000 −0.03 ± 0.04, 200–999 −0.18 ± 0.07 |
| F03ft (Muon lr 5e-4) | F03 base, finetune LRs ×2 | 8 | 5.54 / 0.81 · 5.51 / 0.81 | 5.23 / 0.91 · 5.19 / 0.90 | — | vs F03ft at 2.5e-4: +0.04 ± 0.03 selected, 0.00 ± 0.03 final → finetune LR ×2 makes no difference |
| F03ft (Muon lr 1.25e-4) | F03 base, finetune LRs ×0.5 | 10 | 5.59 / 0.81 | 5.23 / 0.91 · 5.24 / 0.92 | — | vs F03ft at 2.5e-4: +0.04 ± 0.02 selected, +0.05 ± 0.02 final → finetune LR bracket 1.25e-4 / 2.5e-4 / 5e-4 is flat within 0.05°; 2.5e-4 is the (weak) optimum |
| F03ft (1024 DOMs) | F03 base, max_doms 1024 (bs 128×4) | 7 | 5.56 / 0.81 · 5.50 / 0.81 | 5.20 / 0.89 · 5.18 / 0.90 | — | vs F03ft at 512 DOMs: +0.01 ± 0.02 selected, −0.01 ± 0.02 final; on ≥5000-pulse events 0.00 ± 0.01 → 512 DOMs is enough |
| F03ft (≥500-pulse set) | F03 base, finetune on ≥500-pulse events (2.56M) | 9 | 10.17 / 1.47 on its ≥500 dev (≥1000 subset 5.55) | 5.18 / 0.90 · 5.17 / 0.89 | −0.67 ± 0.05 | vs F03ft (≥1000 set): ≥1000 −0.01 ± 0.03 (same), 200–999 −0.17 ± 0.05 (better) → the broader finetune set is free on ≥1000 and helps 200–999 |
| F03ft (20 epochs) | F03 base, 20-epoch finetune schedule | 15 | 5.67 / 0.81 · 5.69 / 0.81 | 5.26 / 0.92 · 5.24 / 0.90 | — | vs 10-epoch F03ft: +0.07 ± 0.03 selected, +0.05 ± 0.03 final; train–dev gap +0.45 vs +0.12 → the finetune starts to overfit its 1.46M events; 10 epochs is right |
| F07ft | F07: hybrid, 24 layers | 8 | 5.41 / 0.81 · 5.42 / 0.81 | 5.13 / 0.83 · 5.09 / 0.84 | −0.48 ± 0.06 (final −0.53) | (is the hybrid base's own finetune); vs F03ft: −0.05 ± 0.06 selected, −0.10 ± 0.05 final |
| F10ft | F10: hybrid, 24 layers, seed 41 | 8 | 5.33 / 0.81 · 5.36 / 0.81 | 5.07 / 0.82 · 5.08 / 0.82 | −0.47 ± 0.05 | vs F07ft (the seed-17 twin): −0.06 ± 0.04 selected, −0.01 ± 0.04 final; vs F03ft: −0.11 ± 0.05 |
| F18ft (≥1000) | F18: hybrid, 24 layers, **no dropout** | 9 | 5.49 / 0.81 · 5.46 / 0.81 | 5.20 / 0.86 · 5.14 / 0.83 | −0.41 ± 0.05 (final −0.47) | vs F07ft (dropout-0.1 twin): +0.07 ± 0.05 selected, +0.01 ± 0.05 final; vs F10ft: +0.13 / +0.07 → no dropout does not help the hybrid 24-layer finetune |
| F18ft (≥500 set) | F18: hybrid 24, no dropout; final recipe (≥500) | 7 | 10.40 / 1.41 on ≥500 dev (≥1000 subset 5.50) | 5.18 / 0.86 · 5.11 / 0.83 | −0.43 ± 0.05 (final −0.51) | vs F07ft (hybrid ≥1000): ≥1000 +0.05 / −0.03 ± 0.05, 200–999 0.00 ± 0.07; vs F03ft ≥500 (raw): ≥1000 0.00 / −0.07 ± 0.04, 200–999 +0.19 / +0.21 ± 0.07 |
| F08ft | F08: hybrid, 6 layers = **the July record recipe**, re-run | 10 | 5.75 / 0.82 (dev reproduces the July 5.76) | 5.29 / 0.91 | −0.58 ± 0.05 | F08ft is the old recipe; the new finetunes vs it on the same events: F10ft −0.22 ± 0.05, F07ft −0.16 ± 0.05, F03ft −0.11 ± 0.05 |
| F16ft | F16: hybrid, 6 layers, seed 41 (record recipe, other seed) | 10 | 5.62 / 0.82 | 5.27 / 0.93 | −0.61 ± 0.05 | vs F08ft (record recipe, seed 17): −0.03 ± 0.03 → record-recipe seed noise ≈ 0.03°; vs F03ft: +0.08 ± 0.05 |
| F02ft | F02: raw, 6 layers (the July raw finetune recipe, re-run) | 8 | 6.17 / 1.03 · 6.12 / 1.04 | 5.83 / 1.16 · 5.83 / 1.13 | −0.81 ± 0.06 | vs F08ft (record hybrid recipe): +0.54 ± 0.06; vs F03ft: +0.65 ± 0.05 |
| F13ft | F13: hybrid, 6 layers, **no dropout** | 8 | 5.74 / 0.81 · 5.73 / 0.82 | 5.35 / 0.91 · 5.32 / 0.90 | −0.45 ± 0.06 | vs F08ft (dropout-0.1 twin): +0.06 ± 0.04 selected, +0.03 ± 0.04 final → the no-dropout base's 0.07° advantage does not survive finetuning |
| F14ft | F14: raw, 6 layers, **no dropout** | 10 | 6.05 / 0.98 | 5.66 / 1.10 | −0.74 ± 0.06 | vs F02ft (dropout-0.1 twin): −0.17 ± 0.05 → on raw tokens the no-dropout base DOES finetune better; vs F13ft (hybrid no-dropout): +0.31 ± 0.06 |
| F11ft | F11: raw, 6 layers, linear input | 10 | 5.85 / 0.91 | 5.50 / 0.98 | −0.72 ± 0.05 | vs F02ft (no linear input): −0.33 ± 0.05; vs F14ft (no dropout, no linear): −0.16 ± 0.06; vs F03ft (24 layers + linear): +0.31 ± 0.05 |
| F01ft | F01: raw, 24 layers (no linear input) | 10 | 5.84 / 0.84 | 5.51 / 1.02 · 5.45 / 1.00 | −0.58 ± 0.06 (final −0.64) | vs F02ft (6 layers): −0.32 ± 0.05; vs F11ft (6 + linear): +0.01 ± 0.05; vs F03ft (24 + linear): +0.32 ± 0.05 |
| F05ft | F05: raw, 24 layers, seed 41 | 7 | 5.79 / 0.94 · 5.63 / 0.88 | 5.49 / 1.02 · 5.41 / 0.99 | −0.62 ± 0.05 (final −0.70) | vs F01ft (seed 17 twin): −0.02 ± 0.05 selected, −0.04 ± 0.05 final; vs F03ft: +0.30 / +0.23 |
| F04ft | F04: raw, 12 layers | 10 | 5.78 / 0.95 | 5.53 / 1.02 · 5.50 / 1.03 | −0.71 ± 0.06 | vs F02ft (6 layers): −0.30 ± 0.05; vs F01ft (24 layers): +0.02 ± 0.04 → raw finetune depth curve also saturates at 12 |
| F15ft | F15: hybrid, 12 layers | 8 | 5.45 / 0.81 · 5.47 / 0.81 | 5.13 / 0.85 · 5.10 / 0.84 | −0.45 ± 0.05 | vs F08ft (hybrid 6): −0.16 ± 0.04; vs F07ft / F10ft (hybrid 24): 0.00 / +0.06 ± 0.04 → hybrid finetune saturates at 12 layers too |

**Caution on absolute hi-E numbers.** Bright-event means are heavy-tailed: on the 8,996 locked events the sampling uncertainty of an absolute mean is about ±0.2°, and on the 2,258-event dev set about ±0.4°. The July record recipe re-run (F08ft) reproduces the July dev value (5.75° vs 5.76°) but scores 5.29° on the locked set, so **do not compare new locked-set numbers with the July 5.76°** (different events). The reliable statement is the paired one on identical events: the deep finetunes beat the record recipe by 0.11–0.22° (±0.05) on ≥1000-pulse events, and beat their own base models by 0.5–0.7°.

**What the hi-E finetune costs on the bulk** (full locked set, 800k events, finetuned model minus its own base, paired): F03ft +0.22 ± 0.01° overall (+0.26° on <200-pulse events, −0.31° on 200–999, −0.66° on ≥1000); F07ft +0.13 ± 0.01° overall (+0.16° on <200, −0.49° on 200–999, −0.48° on ≥1000). The finetuned models improve every event above 200 pulses and degrade only the dim majority, so the two-model setup (base for the bulk, finetune for ≥200) is the right deployment; the finetunes' own bulk numbers are 55.58° (F03ft) and 56.07° (F07ft).

## Final headline (locked set 652–655, two seeds each)

| system | bulk mean° | bulk median° | 200–999° | ≥1000° |
|---|---|---|---|---|
| Raw 5M control + its ≥1000 finetune (July raw recipe): F02/F12, F02ft | 55.60 / 55.65 | 47.96 / 48.06 | — | 5.83 |
| July record recipe re-run (hybrid 5M + finetune): F08/F16, F08ft/F16ft | 56.06 / 56.08 | 48.60 / 48.58 | — | 5.29 / 5.27 |
| **Raw 24 layers + linear input, + ≥500 finetune: F03/F20, F03ft/F20ft** | **55.36 / 55.32** | **47.63 / 47.58** | **26.64 / 26.65** | **5.18 / 5.15** |
| Hybrid 24 layers + finetune: F07/F10 (≥1000), F18 (≥500) | 55.95 / 55.91 | 48.44 / 48.33 | 26.83 | 5.07–5.18 |

Paired gains of the raw deep system over the July record recipe on identical events: bulk −0.7°, ≥1000 −0.11 to −0.14°; over the raw 5M control: bulk −0.25 to −0.33°, ≥1000 −0.65 to −0.68° (finetuned vs finetuned). Batches 656–659 remain untouched.

## Test-set selection rule (frozen 2026-09-18 before any evaluation on 656–659)

1. Bulk model: base run with the lowest locked-set (652–655) mean angular error over all events, final-epoch checkpoint → **F20** (raw, 24 layers, learned linear input, dropout 0.1, seed 41; 55.32°).
2. Hi-E model: finetune with the lowest locked-set mean angular error over all events with ≥200 pulses, dev-selected checkpoint → **F20ft (≥500-pulse finetune)** (22.47° on ≥200; F03ft 22.61, hybrid finetunes 22.6–22.7). By ≥1000-only mean the hybrid F10ft (5.07° vs 5.15°) would have been picked; it is deliberately not evaluated on the test set.
3. Deployment: bulk model for <200 pulses, hi-E model (512 DOMs) for ≥200 pulses. Exactly one evaluation of these two checkpoints on batches 656–659; no other model is scored on the test set.

## Test set 656–659 (single evaluation, 2026-09-18 07:37 UTC, rule above; predictions in evals/test_656_659/)

| | test 656–659 | validation 652–655 (same checkpoints) |
|---|---|---|
| Bulk model (F20), all 800,000 events: mean / median | 55.39 / 47.66 | 55.32 / 47.58 |
| Bulk model on <200 pulses (753,386 events) | 57.40 / 51.18 | 57.32 / — |
| Hi-E model (F20ft ≥500) on 200–999 pulses (37,540 events) | 26.63 / 4.54 | 26.65 / — |
| Hi-E model on ≥1000 pulses (9,074 events) | 5.53 / 0.87 | 5.15 / 0.85 |
| Bulk model on the same ≥1000 events (for reference) | 6.07 / 1.29 | 5.72 / 1.24 |
| Two-model system, all events | 55.36 / 47.62 | 55.30 / 47.54 |

Test tracks validation on the bulk to 0.07°. On ≥1000 pulses both models are ~0.35° worse on test than on validation (5.53 vs 5.15 finetuned; 6.07 vs 5.72 base), i.e. the test batches' bright events are intrinsically harder, within the ±0.2–0.3° sampling uncertainty of a 9k-event heavy-tailed mean; the finetune's gain over the base is the same on both sets (−0.54° test, −0.57° validation), so there is no sign of selection bias.

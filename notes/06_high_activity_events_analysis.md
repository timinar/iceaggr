# High-Activity Event Analysis (min_pulses ≥ 1000)

Date: 2026-03-05
Script: `scripts/analyze_high_activity_doms.py` (10 batch files sampled evenly from 650)

## Pulse Count Distribution (all 130M training events, batches 1–650)

| Percentile | Pulses |
|-----------|--------|
| P50       |     63 |
| P75       |     89 |
| P90       |    144 |
| P95       |    221 |
| P99       |  1,177 |
| P99.9     | 19,605 |
| Max       | 178,250 |

## High-Activity Events (≥ 1000 pulses)

- Count: **1,463,633 / 130,000,000 (1.13%)**

## DOM Count Distribution (events with ≥ 1000 pulses)

| Percentile | DOMs |
|-----------|------|
| P50       |  329 |
| P75       |  508 |
| P90       |  823 |
| P95       | 1097 |
| P99       | 1578 |
| P99.9     | 2050 |
| Max       | 2533 |
| Mean      |  425 |

## Design Decisions

**max_doms=512** chosen for the fine-tuning config because:
- Covers ~P75 of high-activity events (P75 = 508 DOMs)
- Matches baseline GPU attention memory budget: `batch_size × max_doms²` constant
  - Baseline: 4096 × 128² = 67M token-pairs
  - Fine-tune: 512 × 512² = 134M token-pairs (×2, still fits H100)
- H100 benchmark confirmed: batch_size=512, max_doms=512 → ~21 GB, 98% GPU util

**Data split for fine-tuning:**
- Train: batches 1–635 → 1,429,994 high-activity events (2,793 batches at bs=512)
- Val: batches 636–660 → 55,694 high-activity events, capped at 50,000

## Next Run Suggestions

- Increase batch_size (e.g. 1024) — H100 has 95 GB, currently using ~21 GB at bs=512
- Train for more epochs (20+) if loss is still decreasing
- Consider max_doms=1024 to cover P95 (1097 DOMs), with batch_size=128–256

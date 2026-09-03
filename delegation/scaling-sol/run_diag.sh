#!/bin/bash
# Like-for-like diagnostics for the model-scaling question (2026-09-02).
# 1) 19M MAE hi-E finetune on TEST 656-659 >=1000 (paper compares its VAL 7.43 vs 5M TEST 7.30)
# 2) 5M-MAE base and 19M-MAE base on full VAL 651-655 (1M events) -> per-pulse-count slices
# 3) same two bases on TRAIN batches 1-2 (400k events) -> eval-mode train loss by slice
set -e
cd /lustre/hpc/pheno/inar/iceaggr
OUT=paper/predictions/scaling_diag
B5=checkpoints/v2-none-K84-5M-proper-split-10ep/best.pt
B19=checkpoints/v2-d512-K169-18M-lr2e4-10ep/best.pt
FT19=checkpoints/v2-d512-K169-18M-finetune-hi1000/best.pt
echo "=== $(date) 19M ft test hi1000 ==="
uv run python scripts/inference.py --checkpoint $FT19 --output $OUT/mae19M_ft_hi1000_656_659.parquet --batch-range 656 659 --min-pulses 1000 --max-doms 512 --batch-size 256
echo "=== $(date) 5M base val ==="
uv run python scripts/inference.py --checkpoint $B5 --output $OUT/mae5M_base_651_655.parquet --batch-range 651 655 --batch-size 1024
echo "=== $(date) 19M base val ==="
uv run python scripts/inference.py --checkpoint $B19 --output $OUT/mae19M_base_651_655.parquet --batch-range 651 655 --batch-size 1024
echo "=== $(date) 5M base train 1-2 ==="
uv run python scripts/inference.py --checkpoint $B5 --output $OUT/mae5M_base_train_1_2.parquet --batch-range 1 2 --batch-size 1024
echo "=== $(date) 19M base train 1-2 ==="
uv run python scripts/inference.py --checkpoint $B19 --output $OUT/mae19M_base_train_1_2.parquet --batch-range 1 2 --batch-size 1024
echo "=== $(date) DONE ==="

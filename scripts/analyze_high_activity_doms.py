#!/usr/bin/env python3
"""
Analyze DOM count distribution for high-activity IceCube events (>1000 total pulses).

Run:
    uv run python scripts/analyze_high_activity_doms.py

Output: percentile table of DOM counts for events with >1000 pulses, to guide
the choice of max_doms for fine-tuning on high-activity events.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import yaml

from iceaggr.utils import get_logger

logger = get_logger(__name__)

MIN_PULSES = 1000
TRAIN_BATCH_MIN = 1
TRAIN_BATCH_MAX = 650
SAMPLE_BATCHES = 10  # number of batch files to sample for DOM count analysis


def load_data_config() -> dict:
    config_path = Path(__file__).parent.parent / "src" / "iceaggr" / "data" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_data_config()
    data_root = Path(cfg["data"]["root"])
    train_dir = Path(cfg["data"]["train"])

    # --- Load metadata ---
    meta_path = data_root / "train_meta.parquet"
    logger.info(f"Loading metadata from {meta_path}")
    meta = pl.read_parquet(meta_path)

    # Compute n_pulses from index columns
    meta = meta.with_columns(
        (pl.col("last_pulse_index") - pl.col("first_pulse_index") + 1).alias("n_pulses")
    )

    # Filter to training batches only
    meta = meta.filter(
        (pl.col("batch_id") >= TRAIN_BATCH_MIN) & (pl.col("batch_id") <= TRAIN_BATCH_MAX)
    )

    total_events = len(meta)
    logger.info(f"Total training events (batches {TRAIN_BATCH_MIN}-{TRAIN_BATCH_MAX}): {total_events:,}")

    # --- Pulse count distribution ---
    pulse_arr = meta["n_pulses"].to_numpy()
    print("\n=== Pulse count distribution (all events) ===")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  P{p:5.1f}: {np.percentile(pulse_arr, p):>8.0f} pulses")
    print(f"  Max:    {pulse_arr.max():>8.0f} pulses")

    # --- Filter to high-activity events ---
    hi = meta.filter(pl.col("n_pulses") >= MIN_PULSES)
    n_hi = len(hi)
    print(f"\n=== Events with >= {MIN_PULSES} total pulses ===")
    print(f"  Count:    {n_hi:,} / {total_events:,}  ({100 * n_hi / total_events:.2f}%)")

    if n_hi == 0:
        logger.error("No high-activity events found — check threshold or data paths.")
        return

    # --- Count unique DOMs per high-activity event ---
    # Load sensor_id column from each batch parquet (grouped to minimise file opens)
    # Sample a subset of batch files evenly across the range
    all_batch_ids = sorted(hi["batch_id"].unique().to_list())
    step = max(1, len(all_batch_ids) // SAMPLE_BATCHES)
    batch_ids_sorted = all_batch_ids[::step][:SAMPLE_BATCHES]
    hi_sample = hi.filter(pl.col("batch_id").is_in(batch_ids_sorted))
    logger.info(f"Sampling {len(batch_ids_sorted)} batch files → "
                f"{hi_sample.height:,} high-activity events …")

    dom_counts: list[int] = []

    for i, batch_id in enumerate(batch_ids_sorted):
        batch_path = train_dir / f"batch_{batch_id}.parquet"
        events_in_batch = hi_sample.filter(pl.col("batch_id") == batch_id)

        # Load only the sensor_id column as numpy (much faster than to_pylist)
        sensor_arr = (
            pq.read_table(str(batch_path), columns=["sensor_id"])
            .column("sensor_id")
            .to_numpy()
        )

        starts = events_in_batch["first_pulse_index"].to_numpy()
        ends = events_in_batch["last_pulse_index"].to_numpy()

        for start, end in zip(starts, ends):
            n_doms = np.unique(sensor_arr[start : end + 1]).size
            dom_counts.append(n_doms)

        logger.info(f"  [{i+1}/{len(batch_ids_sorted)}] batch {batch_id}: "
                    f"{len(dom_counts):,} events total")

    dom_arr = np.array(dom_counts)

    # --- DOM count distribution ---
    print(f"\n=== DOM count distribution for events with >= {MIN_PULSES} pulses ===")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  P{p:5.1f}: {np.percentile(dom_arr, p):>6.0f} DOMs")
    print(f"  Max:    {dom_arr.max():>6.0f} DOMs")
    print(f"  Mean:   {dom_arr.mean():>6.0f} DOMs")

    # Suggested max_doms values
    p95 = int(np.percentile(dom_arr, 95))
    p99 = int(np.percentile(dom_arr, 99))
    # Round up to next power of 2 or clean number
    def next_clean(n):
        for v in [64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 1024]:
            if v >= n:
                return v
        return n

    print(f"\n=== Suggested max_doms for fine-tuning config ===")
    print(f"  P95 = {p95}  → next clean value: {next_clean(p95)}")
    print(f"  P99 = {p99}  → next clean value: {next_clean(p99)}")
    print()
    print("Current baseline max_doms = 128 (covers most normal events).")
    print("For high-activity fine-tuning, use P95 or P99 rounded up.")


if __name__ == "__main__":
    main()

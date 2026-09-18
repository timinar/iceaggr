#!/usr/bin/env python3
"""Final results table for every run of the 2026-09 scaling fleet.

Reads configs/ladder/<run>.yaml, the locked-set predictions (evals/ladder/) and the test-set
predictions (evals/test_656_659/), and writes evals/ladder/final_table.{md,csv}.
Metrics: mean / median angular error (deg) over all events and over the <200, 200-999, >=1000
pulse slices. Base runs are scored on all events at 128 DOMs; finetunes on >=200-pulse events
at their finetune DOM cap (512 or 1024). Validation = batches 652-655, test = 656-659.
"""
import glob, os, sys
import numpy as np, polars as pl, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK, TEST = f"{ROOT}/evals/ladder", f"{ROOT}/evals/test_656_659"

def slices(path):
    if not path or not os.path.exists(path): return {}
    df = pl.read_parquet(path); e = df["angular_error_deg"].to_numpy(); n = df["n_pulses"].to_numpy()
    out = {}
    for tag, m in (("all", np.ones_like(n, bool)), ("lt200", n < 200), ("200_999", (n >= 200) & (n < 1000)), ("ge1000", n >= 1000)):
        if m.sum(): out[tag] = (float(e[m].mean()), float(np.median(e[m])), int(m.sum()))
    return out

def locked_path(run):
    if "ft_" in run:
        for c in (f"{LOCK}/{run}_best_652_655_ge200.parquet", f"{LOCK}/{run}_best_652_655_ALL.parquet",
                  f"{LOCK}/{run}_best_652_655_ge1000.parquet", f"{LOCK}/{run}_652_655_ge1000.parquet"):
            if os.path.exists(c): return c
        return None
    return f"{LOCK}/{run}_652_655.parquet"

def test_path(run):
    for c in (f"{TEST}/{run}_ge200.parquet", f"{TEST}/{run}_ALL.parquet",
              f"{TEST}/F20_bulk_656_659_ALL.parquet" if run == "F20_130M_d256L24lin_s41" else "",
              f"{TEST}/F20ft_hiE_656_659_ge200.parquet" if run == "F20ft_hi500_md512_mlr2p5em4" else ""):
        if c and os.path.exists(c): return c
    return None

def fmt(s, tag, what="mean"):
    if tag not in s: return "—"
    m, med, n = s[tag]; return f"{m:.2f}" if what == "mean" else f"{med:.2f}"

rows = []
for cfg_path in sorted(glob.glob(f"{ROOT}/configs/ladder/*.yaml")):
    run = os.path.basename(cfg_path)[:-5]
    if not os.path.isdir(f"{ROOT}/checkpoints/ladder/{run}"): continue
    c = yaml.safe_load(open(cfg_path)); m, t, d = c["model"], c["training"], c["data"]
    ft = bool(c.get("checkpoint", {}).get("finetune"))
    tok = "hybrid" if d.get("tokenization") == "hybrid" else "raw"
    emb = "linear" if m.get("input_mode") == "linear" else "none"
    ev = d.get("max_events", 0) / 1e6
    lk, ts = slices(locked_path(run)), slices(test_path(run))
    rows.append(dict(run=run, kind="finetune" if ft else "base", base=(c["checkpoint"].get("resume", "").split("/")[-2] if ft else ""),
        layers=m["num_layers"], params_M=round(c.get("_ladder", {}).get("n_params", 0) / 1e6, 1), tokenization=tok, embedding=emb,
        dropout=m.get("dropout"), muon_lr=t.get("muon_lr"), seed=t.get("seed"), events_M=ev if not ft else "", epochs=t["epochs"],
        max_doms=m["max_doms"], min_pulses=d.get("min_pulses", ""),
        val_all=fmt(lk, "all"), val_all_med=fmt(lk, "all", "med"), val_lt200=fmt(lk, "lt200"), val_200_999=fmt(lk, "200_999"), val_ge1000=fmt(lk, "ge1000"), val_ge1000_med=fmt(lk, "ge1000", "med"),
        test_all=fmt(ts, "all"), test_all_med=fmt(ts, "all", "med"), test_lt200=fmt(ts, "lt200"), test_200_999=fmt(ts, "200_999"), test_ge1000=fmt(ts, "ge1000"), test_ge1000_med=fmt(ts, "ge1000", "med")))

import csv
with open(f"{LOCK}/final_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
hdr = ["run", "kind", "base", "layers", "params (M)", "tokens", "input embedding", "dropout", "Muon lr", "seed", "train events (M)", "epochs", "max DOMs", "finetune min pulses",
       "VAL all mean", "VAL all median", "VAL <200", "VAL 200–999", "VAL ≥1000 mean", "VAL ≥1000 median",
       "TEST all mean", "TEST all median", "TEST <200", "TEST 200–999", "TEST ≥1000 mean", "TEST ≥1000 median"]
with open(f"{LOCK}/final_table.md", "w") as f:
    f.write("# All runs of the 2026-09 scaling fleet: validation (batches 652–655) and test (656–659) angular errors in degrees\n\n")
    f.write("Base runs are scored on all events with 128 DOMs per event; finetunes on events with ≥200 pulses at their finetune DOM cap. Checkpoint = the dev-selected one (best.pt), which is the final epoch for every base run. '—' = not applicable / not evaluated.\n\n")
    f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for r in rows: f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")
print(f"{len(rows)} rows -> evals/ladder/final_table.md / .csv; test columns filled for {sum(1 for r in rows if r['test_all'] != '—' or r['test_ge1000'] != '—')} runs so far")

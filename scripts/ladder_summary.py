#!/usr/bin/env python3
"""Summarize scaling-ladder runs from wandb (tag 'scaling-ladder') as one table.

Columns: run, params, events, epochs, bs, muon_lr, best val NLL (+epoch), val angular
error at that epoch (all / ≥200 / ≥1000 pulses, mean°), median°, final train loss, and
the train−val gap proxy. "best" = epoch with the lowest val/nll (vMF) or val/loss.

  uv run python scripts/ladder_summary.py            # all ladder runs
  uv run python scripts/ladder_summary.py --filter L1M --csv out.csv
"""

import argparse

import pandas as pd

import wandb


def summarize(run):
    cfg = run.config
    m, t, d = cfg.get("model", {}), cfg.get("training", {}), cfg.get("data", {})
    h = run.history(samples=100000, pandas=True)
    ep = h.dropna(subset=["epoch"]) if "epoch" in h else h
    key = "val/nll" if "val/nll" in ep and ep["val/nll"].notna().any() else "val/loss"
    row = {
        "run": run.name, "state": run.state,
        "params_M": round(cfg.get("_ladder", {}).get("n_params", float("nan")) / 1e6, 2),
        "d": m.get("d_model"), "L": m.get("num_layers"), "K": m.get("max_pulses_per_dom"),
        "in": m.get("input_mode"), "events_M": d.get("max_events", 0) / 1e6,
        "ep": t.get("epochs"), "bs": t.get("batch_size"),
        "muon_lr": t.get("muon_lr") if t.get("optimizer", "adamw") == "muon" else None,
        "lr": t.get("lr"), "seed": t.get("seed"), "drop": m.get("dropout"),
    }
    if len(ep) == 0 or key not in ep:
        return row
    i = ep[key].idxmin()
    b = ep.loc[i]
    row.update({
        "best_ep": int(b["epoch"]), "n_ep_done": int(ep["epoch"].max()),
        "nll": round(float(b.get("val/nll", float("nan"))), 4),
        "val_loss": round(float(b["val/loss"]), 4),
        "deg": round(float(b.get("val/angular_error_deg", float("nan"))), 3),
        "med": round(float(b.get("val/angular_error_median_deg", float("nan"))), 3),
        "ge200": round(float(b.get("val/angular_error_deg_ge200", float("nan"))), 3),
        "ge1000": round(float(b.get("val/angular_error_deg_ge1000", float("nan"))), 3),
        "train_last": round(float(ep["train/epoch_loss"].iloc[-1]), 4),
        "val_last": round(float(ep["val/loss"].iloc[-1]), 4),
        "runtime_h": round(run.summary.get("_runtime", 0) / 3600, 2),
    })
    # eval-mode train-eval set (fit probe): NLL at the best epoch and the gap
    if "train_eval/nll" in ep and ep["train_eval/nll"].notna().any():
        row["te_nll"] = round(float(b.get("train_eval/nll", float("nan"))), 4)
        row["gap_nll"] = round(float(b.get("train_eval/gap_nll", float("nan"))), 4)
        row["te_deg"] = round(float(b.get("train_eval/angular_error_deg", float("nan"))), 3)
        row["te_nll_last"] = round(float(ep["train_eval/nll"].iloc[-1]), 4)
        row["val_nll_last"] = round(float(ep["val/nll"].iloc[-1]), 4)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="iceaggr")
    p.add_argument("--tag", default="scaling-ladder")
    p.add_argument("--filter", default=None, help="substring of run name")
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    api = wandb.Api(timeout=60)
    runs = api.runs(a.project, filters={"tags": {"$in": [a.tag]}}, order="+created_at")
    rows = [summarize(r) for r in runs if not a.filter or a.filter in r.name]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()

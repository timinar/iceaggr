#!/usr/bin/env python3
"""Generate scaling-ladder configs (and optional queue jobs) from a base recipe.

Every ladder run = the current best base recipe
(configs/train_flat_v2_none_K84_muon_combined_10ep.yaml: vMF K=3 unclamped,
combined loss, Muon) with a handful of overrides: model shape, data budget,
batch size, LRs, seed. Configs land in configs/ladder/<name>.yaml; with
--enqueue <class> a one-line job script is dropped into queue/<class>/pending/
for scripts/queue_worker.sh.

Shape conventions
- head_dim fixed at 32 → num_heads = d_model // 32 (5M reference: 8 heads).
- hidden_dim = 4·d_model, head_hidden_dim = 1024 (as in both prior scale-ups).
- input_mode 'none' with d_model > 256 zero-pads the 256-dim raw token (K=84)
  instead of inflating K (the K=169 mistake); 'linear' adds a learned projection.

Examples
  uv run python scripts/scaling_ladder.py --name L1M_d256_L6 --max-events 1000000
  uv run python scripts/scaling_ladder.py --name L1M_d512_L6 --d-model 512 --max-events 1000000 \
      --muon-lr 0.005 --enqueue h100
  uv run python scripts/scaling_ladder.py --print-params --d-model 384 --layers 8
"""

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "train_flat_v2_none_K84_muon_combined_10ep.yaml"


def count_params(d_model: int, layers: int, hidden_dim: int, head_hidden: int,
                 input_mode: str, input_dim: int = 256, vmf_components: int = 3) -> int:
    """Parameter count of FlatTransformerV2 (vMF head), matching the module."""
    blocks = layers * (4 * d_model * d_model + 2 * d_model * hidden_dim)
    scalars = 2 * layers + d_model  # resid/x0 lambdas + CLS
    proj = 0
    if input_mode == "linear":
        proj = input_dim * d_model + d_model
    elif input_mode == "mlp":
        proj = input_dim * d_model + d_model + d_model * d_model + d_model
    # VMFMixtureHead: Linear(d, h) + Linear(h, 5K)  (see vmf_loss.py)
    head = d_model * head_hidden + head_hidden + head_hidden * 5 * vmf_components + 5 * vmf_components
    return blocks + scalars + proj + head


def build(args) -> dict:
    with open(args.base) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    m, t, d = cfg["model"], cfg["training"], cfg["data"]

    hidden = args.hidden_dim or 4 * args.d_model
    heads = args.heads or max(1, args.d_model // 32)
    assert args.d_model % heads == 0
    m.update({
        "d_model": args.d_model,
        "num_layers": args.layers,
        "hidden_dim": hidden,
        "num_heads": heads,
        "head_hidden_dim": args.head_hidden_dim,
        "max_pulses_per_dom": args.K,
        "input_mode": args.input_mode,
        "dropout": args.dropout,
    })
    if args.max_doms:
        m["max_doms"] = args.max_doms
    # Tokenization switch (defaults = flat raw tokens, as in the base config).
    # hybrid: 256-dim raw(K'=80)+12 aggregate tokens, learned Linear(256→d) as in
    # configs/hybrid_muon_base.yaml (the record hi-E model's tokenization).
    # npe15: 15 summary statistics per DOM, Linear(15→d).
    if args.tokenization == "hybrid":
        d["tokenization"] = "hybrid"
        d["hybrid_mode"] = "full"
        m["input_dim"] = 256
        m["input_mode"] = "linear"
    elif args.tokenization == "npe15":
        d["tokenization"] = "npe15"
        m["input_dim"] = 15
        m["input_mode"] = "linear"
    else:
        d.pop("tokenization", None)
    if args.kappa_reg0:
        # unregularized fit probe: no κ² penalty at all
        m["vmf_kappa_reg"] = 0.0
        m["vmf_kappa_reg_final"] = 0.0

    t.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "muon_lr": args.muon_lr,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
    })
    if args.seed is not None:
        t["seed"] = args.seed
    if args.adamw:
        t["optimizer"] = "adamw"
        t.pop("muon_lr", None)
        t.pop("muon_momentum", None)

    d.update({
        "max_events": args.max_events,
        "val_events": args.val_events,
        "val_per_epoch": args.val_per_epoch,
        "num_workers": args.workers,
    })
    if args.train_eval_events:
        d["train_eval_events"] = args.train_eval_events
    if args.grad_accum > 1:
        t["grad_accum_steps"] = args.grad_accum
    if args.main_threads:
        t["main_threads"] = args.main_threads
    if args.geometry_path:
        d["geometry_path"] = args.geometry_path

    n_params = count_params(args.d_model, args.layers, hidden, args.head_hidden_dim,
                            m["input_mode"], input_dim=m.get("input_dim", 4 + 3 * args.K),
                            vmf_components=m.get("vmf_components", 3))
    cfg["wandb"]["name"] = args.name
    cfg["wandb"]["tags"] = ["scaling-ladder", f"events-{args.max_events}",
                            f"d{args.d_model}-L{args.layers}", f"params-{n_params/1e6:.1f}M",
                            f"tok-{args.tokenization}",
                            f"muon-lr-{args.muon_lr}" if not args.adamw else f"adamw-lr-{args.lr}"]
    if args.tag:
        cfg["wandb"]["tags"].append(args.tag)
    cfg["checkpoint"]["dir"] = args.ckpt_dir
    cfg["checkpoint"]["save_every"] = args.save_every  # default: best.pt only
    cfg["_ladder"] = {"n_params": n_params, "note": args.note}
    return cfg, n_params


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=str(BASE))
    p.add_argument("--name", help="run name (wandb + checkpoint subdir + config filename)")
    p.add_argument("--note", default="")
    p.add_argument("--tag", default=None, help="extra wandb tag")
    # shape
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=None, help="FFN dim (default 4*d_model)")
    p.add_argument("--heads", type=int, default=None, help="default d_model//32 (head_dim 32)")
    p.add_argument("--head-hidden-dim", type=int, default=1024)
    p.add_argument("--K", type=int, default=84)
    p.add_argument("--input-mode", default="none", choices=["none", "linear", "mlp"])
    p.add_argument("--tokenization", default="flat", choices=["flat", "hybrid", "npe15"],
                   help="DOM tokenization; hybrid/npe15 force input_mode linear + input_dim")
    p.add_argument("--max-doms", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--muon-lr", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4, help="AdamW (side-group) LR")
    p.add_argument("--adamw", action="store_true", help="plain AdamW instead of Muon")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--kappa-reg0", action="store_true", help="vmf_kappa_reg = final = 0 (fit probe)")
    p.add_argument("--train-eval-events", type=int, default=None,
                   help="score this many fixed train events in eval mode each epoch")
    p.add_argument("--save-every", type=int, default=100, help="epoch checkpoint cadence (100 = best only)")
    p.add_argument("--main-threads", type=int, default=None,
                   help="cap torch threads in the main process (multi-trainer boxes: 4-8)")
    p.add_argument("--geometry-path", default=None, help="override data.geometry_path (other machines)")
    # data
    p.add_argument("--max-events", type=int, default=1_000_000)
    p.add_argument("--val-events", type=int, default=200_000)
    p.add_argument("--val-per-epoch", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    # output
    p.add_argument("--ckpt-dir", default="checkpoints/ladder")
    p.add_argument("--out-dir", default="configs/ladder")
    p.add_argument("--enqueue", default=None, help="queue class (h100|small) to drop a job into")
    p.add_argument("--print-params", action="store_true", help="only print the parameter count")
    args = p.parse_args()

    if args.print_params:
        hidden = args.hidden_dim or 4 * args.d_model
        n = count_params(args.d_model, args.layers, hidden, args.head_hidden_dim,
                         args.input_mode, input_dim=4 + 3 * args.K)
        print(f"d{args.d_model} L{args.layers} ff{hidden} {args.input_mode}: {n:,} params ({n/1e6:.2f}M)")
        return
    if not args.name:
        p.error("--name is required unless --print-params")

    cfg, n_params = build(args)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / f"{args.name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    try:
        cfg_ref = cfg_path.relative_to(ROOT)   # inside the repo: keep jobs relocatable
    except ValueError:
        cfg_ref = cfg_path                     # e.g. a scratch dir: absolute path
    cmd = f"uv run python scripts/train_flat.py --config {cfg_ref}"
    print(f"{cfg_ref}  ({n_params/1e6:.2f}M params)\n  {cmd}")

    if args.enqueue:
        qdir = ROOT / "queue" / args.enqueue / "pending"
        qdir.mkdir(parents=True, exist_ok=True)
        job = qdir / f"{args.name}.sh"
        # Self-locating: the job lives in <root>/queue/<class>/{pending,running}/,
        # so the repo root is three levels up from the script's own directory.
        job.write_text("#!/bin/bash\nset -euo pipefail\n"
                       'cd "$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"\n'
                       f"{cmd} 2>&1 | tee logs/ladder_{args.name}.log\n"
                       "exit ${PIPESTATUS[0]}\n")
        print(f"  queued -> {job.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

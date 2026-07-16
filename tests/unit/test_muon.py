"""Unit tests for the Muon optimizer and its train-script wiring.

Covers: the Newton–Schulz orthogonalization property, the (muon | adamw) param
split, that Muon actually decreases a simple objective, that MultiScheduler steps
every underlying group, and that ``build_optimizer_and_scheduler`` keeps the
AdamW path byte-identical while wiring Muon correctly when opted in.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from iceaggr.models.flat_transformer_v2 import FlatTransformerV2
from iceaggr.utils.muon import (
    Muon,
    MultiOptimizer,
    MultiScheduler,
    split_muon_params,
    zeropower_via_newtonschulz5,
)


# --- import build_optimizer_and_scheduler from the (non-package) scripts/ dir ---
def _load_train_flat():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "train_flat.py"
    spec = importlib.util.spec_from_file_location("train_flat", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_flat"] = module
    spec.loader.exec_module(module)
    return module


train_flat = _load_train_flat()


def _model_config(**overrides):
    cfg = {
        "version": "v2",
        "input_mode": "linear",
        "max_pulses_per_dom": 4,
        "d_model": 32,
        "max_doms": 16,
        "num_heads": 4,
        "num_layers": 2,
        "hidden_dim": 64,
        "head_hidden_dim": 32,
        "dropout": 0.0,
        "head_type": "directional",
    }
    cfg.update(overrides)
    return cfg


class TestNewtonSchulz:
    """The NS iteration semi-orthogonalizes: singular values pushed toward 1.

    The modded-nanogpt quintic is tuned for training speed, not precision, so it
    holds the singular values in a band around 1 (roughly [0.67, 1.13] for
    well-conditioned inputs) rather than converging them exactly to 1. The tests
    assert that banding and the resulting conditioning improvement, not σ == 1.
    """

    @pytest.mark.parametrize("shape", [(32, 96), (96, 32)])
    def test_singular_values_banded(self, shape):
        """Well-conditioned (wide/tall) input → every σ near 1 ⟺ semi-orthogonal."""
        for seed in range(5):
            torch.manual_seed(seed)
            G = torch.randn(*shape)
            Q = zeropower_via_newtonschulz5(G, steps=5).float()
            sv = torch.linalg.svdvals(Q)
            assert sv.max() < 1.4
            assert sv.min() > 0.6

    def test_improves_conditioning(self):
        """NS compresses the singular-value spectrum (orthogonalizes) on a
        near-singular square matrix, even where it can't lift a ~0 σ all the way
        to 1 in 5 steps."""
        for seed in range(5):
            torch.manual_seed(seed)
            G = torch.randn(64, 64)
            Q = zeropower_via_newtonschulz5(G, steps=5).float()
            assert torch.linalg.cond(Q) < torch.linalg.cond(G)
            assert torch.linalg.cond(Q) < 25.0  # worst observed ≈ 19

    def test_shape_and_finiteness_preserved(self):
        torch.manual_seed(2)
        G = torch.randn(48, 20)
        Q = zeropower_via_newtonschulz5(G, steps=5)
        assert Q.shape == G.shape
        assert torch.isfinite(Q.float()).all()


class TestParamSplit:
    """Every param lands in exactly one group; block 2D matrices go to Muon."""

    def test_partition_is_exact(self):
        model = FlatTransformerV2(_model_config())
        muon_params, adamw_params, muon_names = split_muon_params(model)

        all_ids = {id(p) for p in model.parameters()}
        muon_ids = {id(p) for p in muon_params}
        adamw_ids = {id(p) for p in adamw_params}

        assert muon_ids.isdisjoint(adamw_ids), "a param appears in both groups"
        assert muon_ids | adamw_ids == all_ids, "some param is in neither group"
        assert len(muon_ids) + len(adamw_ids) == len(all_ids)

    def test_muon_group_is_block_2d_weights(self):
        model = FlatTransformerV2(_model_config())
        muon_params, _, muon_names = split_muon_params(model)

        # Every Muon param is a 2D tensor named inside blocks.*
        assert all(n.startswith("blocks.") for n in muon_names)
        assert all(p.ndim == 2 for p in muon_params)

        # And it is exactly the set of block 2D weights (6 per layer × 2 layers).
        expected = {
            n for n, p in model.named_parameters()
            if n.startswith("blocks.") and p.ndim == 2
        }
        assert set(muon_names) == expected
        assert len(muon_names) == 12  # 4 attn + 2 ffn matrices per layer, 2 layers

    def test_compiled_module_names_stripped(self):
        """Split works on a torch.compile-wrapped module (``_orig_mod`` prefix)."""
        model = FlatTransformerV2(_model_config())
        compiled = torch.compile(model)
        _, _, muon_names = split_muon_params(compiled)
        assert all(n.startswith("blocks.") for n in muon_names)
        assert len(muon_names) == 12


class TestMuonStep:
    """Muon takes a well-formed, loss-decreasing step on 2D weights."""

    def test_decreases_frobenius_objective(self):
        torch.manual_seed(0)
        W = torch.nn.Parameter(torch.randn(24, 40))
        opt = Muon([W], lr=0.05, momentum=0.9)
        losses = []
        for _ in range(15):
            opt.zero_grad()
            loss = 0.5 * W.pow(2).sum()  # grad = W
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], "Muon did not decrease the objective"
        assert all(map(lambda x: x == x, losses)), "NaN in loss trajectory"

    def test_rejects_non_2d_params(self):
        v = torch.nn.Parameter(torch.randn(10))
        opt = Muon([v], lr=0.01)
        loss = v.pow(2).sum()
        loss.backward()
        with pytest.raises(AssertionError, match="2D"):
            opt.step()


class TestMultiScheduler:
    """MultiScheduler / MultiOptimizer drive every underlying group."""

    def test_steps_all_groups(self):
        a = torch.nn.Parameter(torch.randn(8, 8))
        b = torch.nn.Parameter(torch.randn(8, 8))
        adamw = torch.optim.AdamW([a], lr=1e-3)
        muon = Muon([b], lr=0.02)
        total_steps = 10
        sched = MultiScheduler([
            torch.optim.lr_scheduler.OneCycleLR(adamw, max_lr=1e-3, total_steps=total_steps, pct_start=0.3),
            torch.optim.lr_scheduler.OneCycleLR(muon, max_lr=0.02, total_steps=total_steps, pct_start=0.3),
        ])
        lr0 = sched.get_last_lr()
        assert len(lr0) == 2  # one per group
        # Warmup: both LRs increase on the first step.
        sched.step()
        lr1 = sched.get_last_lr()
        assert lr1[0] > lr0[0]
        assert lr1[1] > lr0[1]

    def test_multi_optimizer_state_roundtrip(self):
        a = torch.nn.Parameter(torch.randn(8, 8))
        b = torch.nn.Parameter(torch.randn(8, 8))
        opt = MultiOptimizer([torch.optim.AdamW([a], lr=1e-3), Muon([b], lr=0.02)])
        a.grad = torch.randn_like(a)
        b.grad = torch.randn_like(b)
        opt.step()
        sd = opt.state_dict()
        assert "optimizers" in sd and len(sd["optimizers"]) == 2
        # A fresh MultiOptimizer can load it back.
        opt2 = MultiOptimizer([torch.optim.AdamW([a], lr=1e-3), Muon([b], lr=0.02)])
        opt2.load_state_dict(sd)


class TestBuildOptimizer:
    """build_optimizer_and_scheduler: AdamW byte-identical, Muon opt-in wiring."""

    def _config(self, optimizer="adamw", **train_overrides):
        cfg = {
            "model": _model_config(),
            "training": {
                "lr": 3e-4,
                "weight_decay": 0.01,
                "optimizer": optimizer,
                "use_amp": True,
                "amp_dtype": "bf16",
            },
        }
        cfg["training"].update(train_overrides)
        return cfg

    def test_adamw_path_is_plain_adamw(self):
        model = FlatTransformerV2(_model_config())
        cfg = self._config("adamw")
        opt, sched = train_flat.build_optimizer_and_scheduler(model, cfg, total_steps=100, pct_start=0.1)

        assert type(opt) is torch.optim.AdamW
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)
        # One group over ALL params, at the configured lr / weight_decay.
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == pytest.approx(3e-4 / 25.0)  # OneCycle initial
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)
        n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        n_model = sum(p.numel() for p in model.parameters())
        assert n_opt == n_model

    def test_adamw_default_when_key_absent(self):
        model = FlatTransformerV2(_model_config())
        cfg = self._config("adamw")
        del cfg["training"]["optimizer"]  # absent → default adamw
        opt, _ = train_flat.build_optimizer_and_scheduler(model, cfg, total_steps=100, pct_start=0.1)
        assert type(opt) is torch.optim.AdamW

    def test_muon_path_builds_two_groups(self):
        model = FlatTransformerV2(_model_config())
        cfg = self._config("muon", muon_lr=0.02)
        opt, sched = train_flat.build_optimizer_and_scheduler(model, cfg, total_steps=100, pct_start=0.1)

        assert isinstance(opt, MultiOptimizer)
        assert isinstance(sched, MultiScheduler)
        assert len(opt.optimizers) == 2
        adamw_opt, muon_opt = opt.optimizers
        assert type(adamw_opt) is torch.optim.AdamW
        assert isinstance(muon_opt, Muon)

        # Muon holds exactly the block 2D weights; union covers the whole model.
        n_muon = sum(p.numel() for g in muon_opt.param_groups for p in g["params"])
        n_adamw = sum(p.numel() for g in adamw_opt.param_groups for p in g["params"])
        muon_params, _, _ = split_muon_params(model)
        assert n_muon == sum(p.numel() for p in muon_params)
        assert n_muon + n_adamw == sum(p.numel() for p in model.parameters())

    def test_muon_rejects_fp16(self):
        model = FlatTransformerV2(_model_config())
        cfg = self._config("muon", amp_dtype="fp16")
        with pytest.raises(ValueError, match="bf16"):
            train_flat.build_optimizer_and_scheduler(model, cfg, total_steps=100, pct_start=0.1)

    def test_unknown_optimizer_raises(self):
        model = FlatTransformerV2(_model_config())
        cfg = self._config("rmsprop")
        with pytest.raises(ValueError, match="Unknown training.optimizer"):
            train_flat.build_optimizer_and_scheduler(model, cfg, total_steps=100, pct_start=0.1)

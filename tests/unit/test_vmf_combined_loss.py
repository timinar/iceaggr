"""
Unit tests for the opt-in combined vMF loss (``vmf_angular_weight``).

The combined loss adds ``vmf_angular_weight · angular_distance(point_estimate,
truth)`` to the vMF NLL, where the point estimate is the κ-weighted mean already
used at inference. Guarantees under test:
- ``vmf_angular_weight`` absent is byte-identical to ``vmf_angular_weight=0.0``,
  which is byte-identical to the pure-NLL loss (so existing behavior/checkpoints
  are untouched).
- ``vmf_angular_weight=λ>0`` raises the loss by *exactly* λ times the angular
  term computed independently from the model's own point estimate.
- With λ>0 the loss backpropagates to finite gradients on every parameter (the
  angular term is differentiable through ``vmf_weighted_mean``).
"""

import torch

from iceaggr.models.flat_transformer_v2 import FlatTransformerV2


def _base_config(**overrides):
    """Small CPU-friendly vMF config; head_dim = d_model // num_heads = 8."""
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
        "head_type": "vmf",
        "vmf_components": 3,
    }
    cfg.update(overrides)
    return cfg


def _make_inputs(cfg, batch_size=3, seed=0):
    """Fixed seeded (dom_vectors, padding_mask, target) for a config."""
    g = torch.Generator().manual_seed(seed)
    input_dim = 4 + 3 * cfg["max_pulses_per_dom"]
    dom_vectors = torch.randn(batch_size, cfg["max_doms"], input_dim, generator=g)
    padding_mask = torch.ones(batch_size, cfg["max_doms"], dtype=torch.bool)
    padding_mask[:, -3:] = False  # exercise the mask path
    target = torch.randn(batch_size, 3, generator=g)
    target = target / target.norm(dim=-1, keepdim=True)  # unit vectors
    return dom_vectors, padding_mask, target


def _build_seeded(cfg, seed=1234):
    """Build a model under a fixed global seed (init is seed-sensitive)."""
    torch.manual_seed(seed)
    return FlatTransformerV2(cfg)


def _independent_angular_term(direction, target):
    """Recompute the angular term the same way the model does, independently."""
    dot = (direction * target).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(dot).mean()


class TestVMFCombinedLoss:
    def test_absent_equals_zero_weight(self):
        """Flag absent and vmf_angular_weight=0.0 give byte-identical loss."""
        m_absent = _build_seeded(_base_config())
        m_zero = _build_seeded(_base_config(vmf_angular_weight=0.0))
        assert m_absent.vmf_angular_weight == 0.0
        assert m_zero.vmf_angular_weight == 0.0

        dom_vectors, padding_mask, target = _make_inputs(_base_config())
        m_absent.eval()
        m_zero.eval()
        with torch.no_grad():
            loss_absent = m_absent(dom_vectors, padding_mask, target=target)["loss"]
            loss_zero = m_zero(dom_vectors, padding_mask, target=target)["loss"]

        # atol=0 / rtol=0 → exact equality: the λ=0 path must not perturb the NLL.
        assert torch.equal(loss_absent, loss_zero)

    def test_lambda_adds_exactly_angular_term(self):
        """loss(λ) − loss(0) == λ · angular_term, on the same parameters."""
        lam = 0.5
        model = _build_seeded(_base_config(vmf_angular_weight=lam))
        model.eval()
        dom_vectors, padding_mask, target = _make_inputs(_base_config())

        with torch.no_grad():
            out = model(dom_vectors, padding_mask, target=target)
            loss_combined = out["loss"]
            angular = _independent_angular_term(out["direction"], target)

            # Same params, only the weight changes → isolates the added term.
            model.vmf_angular_weight = 0.0
            loss_nll = model(dom_vectors, padding_mask, target=target)["loss"]

        assert torch.allclose(loss_combined, loss_nll + lam * angular, atol=1e-6, rtol=0.0)
        # And the term is actually non-trivial (random dirs → nonzero angle).
        assert angular.item() > 0.0

    def test_gradients_finite_with_lambda(self):
        """λ>0: forward+backward yield finite grads on every parameter."""
        model = _build_seeded(_base_config(vmf_angular_weight=0.5))
        model.train()
        dom_vectors, padding_mask, target = _make_inputs(_base_config())

        loss = model(dom_vectors, padding_mask, target=target)["loss"]
        assert torch.isfinite(loss).all()

        loss.backward()
        n_with_grad = 0
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
            n_with_grad += 1
        assert n_with_grad > 0

"""
Unit tests for FlatTransformerV2 opt-in flags.

Focus: the rotary position embedding (RoPE) opt-in flag (``use_rope``), mirroring
how ``use_spacetime_bias`` is threaded through the model. The key guarantees:
- ``use_rope=False`` is byte-identical to the baseline model (same params, exact
  numerics), so existing checkpoints load unchanged.
- ``use_rope=True`` adds zero learnable parameters and trains (finite forward +
  backward), and the even-head_dim requirement is asserted with a clear message.
"""

import pytest
import torch

from iceaggr.models.flat_transformer_v2 import FlatTransformerV2


def _base_config(**overrides):
    """Small CPU-friendly config; head_dim = d_model // num_heads = 8 (even)."""
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


def _make_inputs(cfg, batch_size=3, seed=0):
    """Fixed seeded (dom_vectors, padding_mask) for a config."""
    g = torch.Generator().manual_seed(seed)
    input_dim = 4 + 3 * cfg["max_pulses_per_dom"]
    dom_vectors = torch.randn(batch_size, cfg["max_doms"], input_dim, generator=g)
    padding_mask = torch.ones(batch_size, cfg["max_doms"], dtype=torch.bool)
    # A couple of padded DOMs to exercise the mask path.
    padding_mask[:, -3:] = False
    return dom_vectors, padding_mask


def _build_seeded(cfg, seed=1234):
    """Build a model under a fixed global seed (init is seed-sensitive)."""
    torch.manual_seed(seed)
    return FlatTransformerV2(cfg)


class TestRoPEOptIn:
    """RoPE opt-in flag mirrors use_spacetime_bias and is off by default."""

    def test_off_is_byte_identical(self):
        """use_rope=False output is exactly equal to the baseline (no flag)."""
        baseline = _build_seeded(_base_config())
        # Explicit use_rope=False must produce the SAME init + numerics.
        with_flag = _build_seeded(_base_config(use_rope=False))

        dom_vectors, padding_mask = _make_inputs(_base_config())
        baseline.eval()
        with_flag.eval()
        with torch.no_grad():
            out_base = baseline(dom_vectors, padding_mask)["direction"]
            out_flag = with_flag(dom_vectors, padding_mask)["direction"]

        # atol=0 / rtol=0 → exact byte-for-byte equality.
        assert torch.allclose(out_base, out_flag, atol=0.0, rtol=0.0)
        assert baseline.rope is None
        assert with_flag.rope is None

    def test_param_count_identical_when_off(self):
        """RoPE adds zero parameters and zero state_dict keys when off OR on."""
        baseline = _build_seeded(_base_config())
        rope_on = _build_seeded(_base_config(use_rope=True))

        n_base = sum(p.numel() for p in baseline.parameters())
        n_rope = sum(p.numel() for p in rope_on.parameters())
        assert n_base == n_rope, "RoPE must add zero learnable parameters"

        # state_dict keys must be identical (Rotary buffers are non-persistent).
        assert set(baseline.state_dict().keys()) == set(rope_on.state_dict().keys())

    def test_on_changes_output(self):
        """use_rope=True actually rotates q/k → output differs from baseline.

        Uses use_zero_init=False so the attention sublayer contributes a non-zero
        signal at initialization. (With the N2-T2 default zero-init output
        projections, the attention output is zeroed at init, so RoPE — which only
        affects q/k — has no effect on the *init-time* output even though it
        shapes attention scores and gradients during training.)
        """
        baseline = _build_seeded(_base_config(use_zero_init=False))
        rope_on = _build_seeded(_base_config(use_zero_init=False, use_rope=True))

        dom_vectors, padding_mask = _make_inputs(_base_config())
        baseline.eval()
        rope_on.eval()
        with torch.no_grad():
            out_base = baseline(dom_vectors, padding_mask)["direction"]
            out_rope = rope_on(dom_vectors, padding_mask)["direction"]
        assert not torch.allclose(out_base, out_rope, atol=1e-5)

    def test_on_forward_backward_finite(self):
        """use_rope=True: forward + backward run with finite outputs/grads."""
        model = _build_seeded(_base_config(use_rope=True))
        model.train()
        dom_vectors, padding_mask = _make_inputs(_base_config())

        out = model(dom_vectors, padding_mask)["direction"]
        assert torch.isfinite(out).all()

        loss = out.square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_odd_head_dim_assertion(self):
        """The even-head_dim requirement fires with a clear message."""
        # d_model=30, num_heads=10 → head_dim=3 (odd) but d_model % num_heads == 0.
        cfg = _base_config(d_model=30, num_heads=10, use_rope=True)
        with pytest.raises(AssertionError, match="even head_dim"):
            FlatTransformerV2(cfg)

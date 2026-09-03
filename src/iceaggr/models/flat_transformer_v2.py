"""
Flat transformer v2 (the N2-T2 backbone) for IceCube direction prediction.

Architecture improvements inspired by nanochat/GPT (Karpathy):
- Functional RMSNorm (no learnable params, simpler and more stable)
- Post-embedding normalization (norm after input projection)
- Per-layer residual scaling + skip connection to initial embedding
- ReLU² activation in FFN (sharper gating than GELU)
- Zero-init output projections (attention c_proj, FFN c_proj)
- No bias in transformer linear layers
- QK norm in attention for stability

Supports configurable input projection mode:
- "none":   identity (d_model must equal input_dim)
- "linear": single Linear(input_dim, d_model)
- "mlp":    Linear → GELU → Linear (learns nonlinear feature interactions)

PER-CHANGE ABLATION FLAGS (all default to the N2-T2 behavior, so existing
checkpoints are byte-identical and load unchanged):
- use_rmsnorm      (default True):  False → LayerNorm in place of RMSNorm
- use_qknorm       (default True):  False → no QK normalization
- use_relu2        (default True):  False → GELU activation in the FFN
- use_zero_init    (default True):  False → uniform-init output projections
- use_resid_scaling(default True):  False → plain pre-norm residual (no x0 skip)
- use_bias         (default False): True  → biases on the transformer linears
Setting all six to the vanilla side reconstructs a standard pre-LN transformer,
which is the control for the per-change ablation (\\cref{sec:ablations}).

OPT-IN POSITIONAL ENCODING (default off, byte-identical to N2-T2 when off):
- use_rope (default False): True → rotary position embedding (RoPE) applied to
  the query/key vectors in every attention layer, indexed by token position
  (CLS at 0, DOM i at i+1). Adds ZERO learnable parameters. Mirrors how
  use_spacetime_bias is threaded through; the two are independent and may both
  be on (RoPE rotates q/k, the spacetime bias is added to the scores).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional


import math

from .directional_head import DirectionalHead
from .vmf_loss import VMFMixtureHead, VMFMixtureLoss, kappa_bias_for_target, vmf_weighted_mean


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Functional RMSNorm with no learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    """Rotary position embedding (RoPE), parameter-free.

    Standard RoPE: the head_dim is split into head_dim/2 pairs, each rotated by
    an angle that grows linearly with token position and geometrically across
    pairs (base ``theta``). Applied per-head to the query and key vectors, so it
    encodes *relative* position through the dot product without touching values.

    Requires an even head_dim (paired rotation). The cos/sin tables are
    registered as non-persistent buffers (not learnable, not in the state_dict),
    so a model with ``use_rope=True`` adds zero parameters and zero state_dict
    keys versus the same model with it off. Built only when ``use_rope=True``.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, (
            f"RoPE requires an even head_dim (d_model // num_heads), got {head_dim}. "
            "Pick num_heads so that d_model // num_heads is even."
        )
        # Per-pair inverse frequencies: theta^(-2i/head_dim), i = 0 .. head_dim/2-1
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.outer(pos, inv_freq)  # (max_seq_len, head_dim/2)
        # Duplicate to full head_dim so cos/sin line up with [first half, second half].
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the two halves of the last dim: [-x2, x1] for x = [x1, x2]."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Rotate q and k of shape (B, num_heads, T, head_dim) by token position.

        Token positions are 0..T-1 (CLS at 0, DOM i at i+1), matching the order
        in which tokens are concatenated in the forward pass.
        """
        T = q.shape[-2]
        cos = self.cos[:T].to(q.dtype)  # (T, head_dim)
        sin = self.sin[:T].to(q.dtype)
        # Broadcast over (B, num_heads): (1, 1, T, head_dim)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class Norm(nn.Module):
    """Selectable normalization: parameter-free RMSNorm (default) or LayerNorm.

    With use_rms=True there are no parameters, so the module adds no state_dict
    keys and the forward is identical to the old inline ``rms_norm`` calls.
    """

    def __init__(self, d_model: int, use_rms: bool = True):
        super().__init__()
        self.use_rms = use_rms
        self.ln = None if use_rms else nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x) if self.use_rms else self.ln(x)


class Attention(nn.Module):
    """Multi-head self-attention with optional QK norm, optional bias, zero-init output."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_qknorm: bool = True, use_bias: bool = False,
                 rope: Optional["Rotary"] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0
        self.use_qknorm = use_qknorm
        # Shared Rotary module (None when use_rope is off). Not a submodule with
        # parameters — it only holds non-persistent cos/sin buffers — but we keep
        # the reference so attention can rotate q/k.
        self.rope = rope

        self.c_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.c_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.c_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # QK norm for stability (from nanochat)
        if self.use_qknorm:
            q = rms_norm(q)
            k = rms_norm(k)

        # Rotary position embedding (opt-in). Apply AFTER QK-norm so we rotate
        # the normalized q/k. Off → this block is skipped and q/k are untouched.
        if self.rope is not None:
            q, k = self.rope(q, k)

        if attn_bias is None:
            # Boolean mask path (default, byte-identical to the original model):
            # (B, 1, 1, T). SDPA expects True = attend, False = mask out.
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            # Additive float-mask path: relative spacetime bias on valid pairs,
            # -inf on padded keys. attn_bias is (B, num_heads, T, T).
            neg_inf = torch.finfo(attn_bias.dtype).min
            key_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T), True = valid key
            attn_mask = attn_bias.masked_fill(~key_mask, neg_inf)

        # Scaled dot-product attention (uses Flash Attention when available)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FFN(nn.Module):
    """Feed-forward network with ReLU² (or GELU) activation, optional bias."""

    def __init__(self, d_model: int, hidden_dim: int,
                 use_relu2: bool = True, use_bias: bool = False):
        super().__init__()
        self.use_relu2 = use_relu2
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=use_bias)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square() if self.use_relu2 else F.gelu(x)  # ReLU² (nanochat) or GELU
        return self.c_proj(x)


class SpacetimeBias(nn.Module):
    """Relative spacetime-interval attention bias (DOM-level, shared across layers).

    Each DOM token carries a spacetime coordinate (t1, x, y, z) read straight
    from the input dom_vectors (cols 4, 0, 1, 2 — already normalized). For every
    pair (i, j) we form a small relative feature vector

        [Δt1, Δx, Δy, Δz, |Δr|, ds²]

    where Δ = pos_i - pos_j, |Δr| is the spatial distance, and
    ds² = (c·Δt1)² - |Δr|² is the Minkowski interval with a learnable speed
    scalar ``c`` (units are normalized, so the MLP/c learn the right scales).
    A tiny shared MLP maps these 6 features → num_heads additive logits, giving
    an (B, num_heads, T, T) bias added to the attention scores in every layer.

    The CLS token (sequence position 0) has no position. Any pair that touches
    CLS — row 0 or column 0 of the T×T grid — gets a single learned per-head
    scalar instead of the MLP output, so undefined CLS coordinates never feed
    the relative features.

    NOTE: the bias is computed once from the input positions and reused across
    all layers (simplest first). Per-layer biases are a future TODO.

    This module exists only when ``use_spacetime_bias=True``; when the flag is
    off it is never constructed, so it adds no parameters / state_dict keys and
    existing checkpoints load unchanged.
    """

    _N_FEATURES = 6  # [Δt1, Δx, Δy, Δz, |Δr|, ds²]

    def __init__(self, num_heads: int, hidden_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(self._N_FEATURES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        # Learnable "speed of light" for the Minkowski interval (normalized units).
        self.log_c = nn.Parameter(torch.zeros(()))
        # Learned per-head scalar for any CLS-involving pair (no position).
        self.cls_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, dom_vectors: torch.Tensor) -> torch.Tensor:
        """Build the (B, num_heads, T, T) additive bias from DOM positions.

        Args:
            dom_vectors: (B, max_doms, input_dim) — the raw model input, where
                col 0/1/2 = x/y/z and col 4 = t1 (first-pulse normalized time).

        Returns:
            bias: (B, num_heads, T, T) with T = max_doms + 1 (CLS at index 0).
        """
        B, D, _ = dom_vectors.shape
        # DOM spacetime coords: (B, D, 4) as [t1, x, y, z]
        t1 = dom_vectors[:, :, 4]
        xyz = dom_vectors[:, :, 0:3]

        # Pairwise separations over the DOM grid (B, D, D, ·)
        dt = t1[:, :, None] - t1[:, None, :]                 # (B, D, D)
        dr = xyz[:, :, None, :] - xyz[:, None, :, :]         # (B, D, D, 3)
        dist = torch.linalg.vector_norm(dr, dim=-1)          # (B, D, D)
        c = torch.exp(self.log_c)
        ds2 = (c * dt) ** 2 - dist ** 2                      # Minkowski interval

        feats = torch.cat(
            [dt[..., None], dr, dist[..., None], ds2[..., None]], dim=-1
        )  # (B, D, D, 6)

        dom_bias = self.mlp(feats)                           # (B, D, D, num_heads)
        dom_bias = dom_bias.permute(0, 3, 1, 2)              # (B, num_heads, D, D)

        # Assemble the full (B, H, T, T) grid with the CLS row/col (index 0).
        T = D + 1
        bias = dom_bias.new_empty(B, self.num_heads, T, T)
        cls = self.cls_bias.view(1, self.num_heads, 1)  # broadcasts over (B, ·, len)
        bias[:, :, 0, :] = cls            # CLS as query (row 0)
        bias[:, :, :, 0] = cls            # CLS as key (col 0)
        bias[:, :, 1:, 1:] = dom_bias     # DOM-DOM block
        return bias


class Block(nn.Module):
    """Pre-norm transformer block (RMSNorm or LayerNorm)."""

    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout: float = 0.1,
                 use_rmsnorm: bool = True, use_qknorm: bool = True,
                 use_relu2: bool = True, use_bias: bool = False,
                 rope: Optional["Rotary"] = None):
        super().__init__()
        self.norm1 = Norm(d_model, use_rms=use_rmsnorm)
        self.norm2 = Norm(d_model, use_rms=use_rmsnorm)
        self.attn = Attention(d_model, num_heads, dropout, use_qknorm=use_qknorm,
                              use_bias=use_bias, rope=rope)
        self.ffn = FFN(d_model, hidden_dim, use_relu2=use_relu2, use_bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), padding_mask, attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class FlatTransformerV2(nn.Module):
    """
    Flat transformer v2 (N2-T2) with nanochat-style improvements.

    Args:
        config: Dictionary with:
            - max_pulses_per_dom: K (default: 16)
            - d_model: transformer dim (default: 128)
            - max_doms: max DOMs per event (default: 128)
            - num_heads: attention heads (default: 8)
            - num_layers: transformer layers (default: 4)
            - hidden_dim: FFN hidden dim (default: 512)
            - head_hidden_dim: direction head hidden dim (default: 128)
            - dropout: dropout rate (default: 0.1)
            - input_mode: "none", "linear", or "mlp" (default: "mlp")
            - use_rmsnorm / use_qknorm / use_relu2 / use_zero_init /
              use_resid_scaling / use_bias: per-change ablation flags (see module docstring)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if config is None:
            config = {}

        self.max_pulses_per_dom = config.get('max_pulses_per_dom', 16)
        self.d_model = config.get('d_model', 128)
        self.max_doms = config.get('max_doms', 128)
        dropout = config.get('dropout', 0.1)
        num_layers = config.get('num_layers', 4)
        num_heads = config.get('num_heads', 8)
        hidden_dim = config.get('hidden_dim', 512)
        input_mode = config.get('input_mode', 'mlp')

        # Per-change ablation flags (defaults reproduce the N2-T2 architecture)
        self.use_rmsnorm = bool(config.get('use_rmsnorm', True))
        self.use_qknorm = bool(config.get('use_qknorm', True))
        self.use_relu2 = bool(config.get('use_relu2', True))
        self.use_zero_init = bool(config.get('use_zero_init', True))
        self.use_resid_scaling = bool(config.get('use_resid_scaling', True))
        self.use_bias = bool(config.get('use_bias', False))
        # Relative spacetime-interval attention bias (opt-in; default off so the
        # model is byte-identical to the current N2-T2 and existing checkpoints
        # load unchanged). The bias module is constructed only when on.
        self.use_spacetime_bias = bool(config.get('use_spacetime_bias', False))
        # Rotary position embedding (opt-in; default off, same byte-identical
        # guarantee). The Rotary module is parameter-free and built only when on.
        self.use_rope = bool(config.get('use_rope', False))
        self.rope_theta = float(config.get('rope_theta', 10000.0))

        # Input dimension: 3 (xyz) + 1 (n_pulses) + 3*K (pulse features) for the
        # flat pulse-concat tokenization. Overridable via config for alternative
        # tokenizations (e.g. the 15-dim NPE summary-statistics tokens), which do
        # not follow the 4+3*K layout. Default preserves the flat layout exactly.
        self.input_dim = int(config.get('input_dim', 4 + 3 * self.max_pulses_per_dom))

        # Input projection (configurable)
        self._pad_input = 0
        if input_mode == 'none':
            if self.input_dim < self.d_model:
                self._pad_input = self.d_model - self.input_dim
            elif self.input_dim > self.d_model:
                raise ValueError(
                    f"input_mode='none' requires d_model >= input_dim={self.input_dim}, got {self.d_model}"
                )
            self.input_proj = nn.Identity()
        elif input_mode == 'linear':
            self.input_proj = nn.Linear(self.input_dim, self.d_model)
        elif input_mode == 'mlp':
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        # Post-embedding norm and final norm (RMSNorm or LayerNorm per flag)
        self.post_norm = Norm(self.d_model, use_rms=self.use_rmsnorm)
        self.final_norm = Norm(self.d_model, use_rms=self.use_rmsnorm)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Per-layer residual scaling + skip connection to initial embedding
        # (from nanochat: resid_lambdas scale residual, x0_lambdas blend in initial embedding)
        self.resid_lambdas = nn.Parameter(torch.ones(num_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))

        # Rotary position embedding (shared across layers), built only when on.
        # max_seq_len = max_doms + 1 (CLS token at position 0, DOM i at i+1).
        # Parameter-free: holds only non-persistent cos/sin buffers.
        self.rope = (
            Rotary(
                head_dim=self.d_model // num_heads,
                max_seq_len=self.max_doms + 1,
                theta=self.rope_theta,
            )
            if self.use_rope else None
        )

        # Transformer blocks (the same Rotary instance is shared by every layer)
        self.blocks = nn.ModuleList([
            Block(self.d_model, num_heads, hidden_dim, dropout,
                  use_rmsnorm=self.use_rmsnorm, use_qknorm=self.use_qknorm,
                  use_relu2=self.use_relu2, use_bias=self.use_bias,
                  rope=self.rope)
            for _ in range(num_layers)
        ])

        # Relative spacetime bias (shared across layers), built only when on.
        self.spacetime_bias = (
            SpacetimeBias(
                num_heads=num_heads,
                hidden_dim=config.get('spacetime_bias_hidden_dim', 32),
            )
            if self.use_spacetime_bias else None
        )

        # Output head: directional or vMF mixture (default mixture=1)
        self.head_type = config.get('head_type', 'directional')
        if self.head_type == 'directional':
            self.direction_head = DirectionalHead(
                embed_dim=self.d_model,
                hidden_dim=config.get('head_hidden_dim', 128),
            )
            self.vmf_head = None
            self.vmf_loss = None
        elif self.head_type == 'vmf':
            self.direction_head = None
            self.vmf_head = VMFMixtureHead(
                d_model=self.d_model,
                hidden_dim=config.get('head_hidden_dim', 128),
                n_components=config.get('vmf_components', 1),
            )
            self.vmf_loss = VMFMixtureLoss(
                kappa_min=config.get('vmf_kappa_min', 1.0),
                kappa_max=config.get('vmf_kappa_max', 10000.0),
                kappa_reg=config.get('vmf_kappa_reg', 1e-4),
                kappa_param=config.get('vmf_kappa_param', 'softplus'),
                kappa_temperature=config.get('vmf_kappa_temperature', 1.0),
            )
            # Opt-in combined loss: NLL + vmf_angular_weight · angular_distance(
            # point_estimate, truth). Competition-metric term alongside the vMF
            # NLL (the 2nd-place Kaggle solution found this worth ~55 bps). The
            # point estimate is the same κ-weighted mean used at inference, and
            # the gradient flows through it into μ/κ/weights (see forward). At
            # the default 0.0 the term is skipped entirely, so the loss and every
            # trained checkpoint stay byte-identical to the pure-NLL path.
            self.vmf_angular_weight = float(config.get('vmf_angular_weight', 0.0))

            # Bias the κ-channel of the head's output so κ_init matches a
            # target value (default ≈ κ_min + log(2), the natural softplus
            # zero-bias init). This matters for the sigmoid parameterization,
            # where zero bias gives κ_init ≈ (κ_min + κ_max) / 2 — wildly
            # overconfident — and forces the model to fight its way down. With
            # this bias, softplus and sigmoid start at the same κ, so any
            # difference in trained behavior is the parameterization itself.
            target_init = config.get('vmf_kappa_init_target')
            if target_init is None:
                target_init = config.get('vmf_kappa_min', 1.0) + math.log(2.0)
            bias_value = kappa_bias_for_target(
                target=float(target_init),
                kappa_min=float(config.get('vmf_kappa_min', 1.0)),
                kappa_max=float(config.get('vmf_kappa_max', 10000.0)),
                param=config.get('vmf_kappa_param', 'softplus'),
                kappa_temperature=float(config.get('vmf_kappa_temperature', 1.0)),
            )
            K = config.get('vmf_components', 1)
            with torch.no_grad():
                head_bias = self.vmf_head.net[-1].bias  # (K*5,) — 3 mu + 1 κ + 1 logw per component
                for k in range(K):
                    head_bias[k * 5 + 3] = bias_value
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        """Initialize weights following nanochat conventions."""
        # CLS token: small init to match projected DOM norms
        nn.init.normal_(self.cls_token, std=0.02)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Transformer blocks: uniform init; zero-init output projections (if enabled)
        s = 3**0.5 * self.d_model**-0.5
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.uniform_(block.ffn.c_fc.weight, -s, s)
            if self.use_zero_init:
                nn.init.zeros_(block.attn.c_proj.weight)  # Zero-init output projection
                nn.init.zeros_(block.ffn.c_proj.weight)
            else:
                nn.init.uniform_(block.attn.c_proj.weight, -s, s)
                nn.init.uniform_(block.ffn.c_proj.weight, -s, s)
            # Zero any biases (present only when use_bias=True)
            for lin in (block.attn.c_q, block.attn.c_k, block.attn.c_v,
                        block.attn.c_proj, block.ffn.c_fc, block.ffn.c_proj):
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

        # Input projection (if not identity)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_normal_(self.input_proj.weight)
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)
        elif isinstance(self.input_proj, nn.Sequential):
            for m in self.input_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        dom_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            dom_vectors: (B, max_doms, input_dim) flat DOM features
            padding_mask: (B, max_doms) True = valid DOM

        Returns:
            Predicted directions (B, 3) as unit vectors
        """
        B = dom_vectors.shape[0]
        device = dom_vectors.device

        # Relative spacetime-interval bias, computed once from the raw input
        # positions (cols x/y/z/t1) and reused in every layer. None when off.
        attn_bias = (
            self.spacetime_bias(dom_vectors) if self.spacetime_bias is not None else None
        )

        # Project to model dimension
        if self._pad_input > 0:
            dom_vectors = F.pad(dom_vectors, (0, self._pad_input))
        x = self.input_proj(dom_vectors)  # (B, max_doms, d_model)

        # Post-embedding norm (from nanochat: norm after token embedding)
        x = self.post_norm(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, max_doms+1, d_model)

        # Extend mask for CLS (always valid)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Save initial embedding for x0 skip connection
        x0 = x

        # Transformer with optional per-layer residual scaling + x0 skip
        for i, block in enumerate(self.blocks):
            if self.use_resid_scaling:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask, attn_bias)

        # Final norm
        x = self.final_norm(x)

        # CLS token embedding
        embedding = x[:, 0, :]

        if self.head_type == 'directional':
            direction = self.direction_head(embedding)
            return {'direction': direction, 'embedding': embedding}

        # vMF mixture head path
        mu, raw_kappa, log_weights = self.vmf_head(embedding)
        direction = vmf_weighted_mean(
            mu, raw_kappa, log_weights,
            kappa_min=self.vmf_loss.kappa_min,
            kappa_max=self.vmf_loss.kappa_max,
            kappa_param=self.vmf_loss.kappa_param,
            kappa_temperature=self.vmf_loss.kappa_temperature,
        )
        out = {
            'mu': mu,
            'raw_kappa': raw_kappa,
            'log_weights': log_weights,
            'direction': direction,
            'embedding': embedding,
        }

        if target is not None and self.vmf_loss is not None:
            loss = self.vmf_loss(mu, raw_kappa, log_weights, target)
            # Opt-in competition-metric term. self.vmf_angular_weight is a plain
            # Python float read from config (not a tensor), so this branch is a
            # trace-time constant under torch.compile — no data-dependent control
            # flow. When it is 0.0 (default) the else branch reproduces the pure
            # NLL loss exactly, byte-for-byte.
            if self.vmf_angular_weight > 0.0:
                out['loss'] = loss + self.vmf_angular_weight * self._angular_term(
                    direction, target
                )
            else:
                out['loss'] = loss

        return out

    @staticmethod
    def _angular_term(direction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean angular distance (radians) between point estimate and truth.

        Differentiable: ``direction`` is the κ-weighted mean returned by
        ``vmf_weighted_mean`` (no detach), so gradients flow back into μ/κ/
        weights and the whole backbone. Computed in fp32 with autocast off —
        arccos' gradient diverges as |dot|→1, so the dot product is clamped to
        (-1+1e-7, 1-1e-7) before arccos to keep both the value and the gradient
        finite.
        """
        with torch.amp.autocast("cuda", enabled=False):
            dot = (direction.float() * target.float()).sum(dim=-1)
            dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            return torch.arccos(dot).mean()

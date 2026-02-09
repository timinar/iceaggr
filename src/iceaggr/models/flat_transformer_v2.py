"""
Flat transformer v2 for IceCube direction prediction.

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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional


from .directional_head import DirectionalHead


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Functional RMSNorm with no learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    """Multi-head self-attention with QK norm, no bias, zero-init output."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.c_q = nn.Linear(d_model, d_model, bias=False)
        self.c_k = nn.Linear(d_model, d_model, bias=False)
        self.c_v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # QK norm for stability (from nanochat)
        q = rms_norm(q)
        k = rms_norm(k)

        # Create attention mask from padding mask: (B, 1, 1, T) for broadcasting
        # SDPA expects: True = attend, False = mask out (opposite of nn.MHA)
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        # Scaled dot-product attention (uses Flash Attention when available)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FFN(nn.Module):
    """Feed-forward network with ReLU² activation, no bias."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² (from nanochat)
        return self.c_proj(x)


class Block(nn.Module):
    """Pre-norm transformer block with RMSNorm."""

    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = Attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, hidden_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), padding_mask)
        x = x + self.ffn(rms_norm(x))
        return x


class FlatTransformerV2(nn.Module):
    """
    Flat transformer v2 with nanochat-style improvements.

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

        # Input dimension: 3 (xyz) + 1 (n_pulses) + 3*K (pulse features)
        self.input_dim = 4 + 3 * self.max_pulses_per_dom

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

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Per-layer residual scaling + skip connection to initial embedding
        # (from nanochat: resid_lambdas scale residual, x0_lambdas blend in initial embedding)
        self.resid_lambdas = nn.Parameter(torch.ones(num_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(self.d_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Direction prediction head
        self.direction_head = DirectionalHead(
            embed_dim=self.d_model,
            hidden_dim=config.get('head_hidden_dim', 128),
        )

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        """Initialize weights following nanochat conventions."""
        # CLS token: small init to match projected DOM norms
        nn.init.normal_(self.cls_token, std=0.02)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Transformer blocks: uniform init, zero-init output projections
        s = 3**0.5 * self.d_model**-0.5
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)  # Zero-init output projection
            nn.init.uniform_(block.ffn.c_fc.weight, -s, s)
            nn.init.zeros_(block.ffn.c_proj.weight)  # Zero-init output projection

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
    ) -> torch.Tensor:
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

        # Project to model dimension
        if self._pad_input > 0:
            dom_vectors = F.pad(dom_vectors, (0, self._pad_input))
        x = self.input_proj(dom_vectors)  # (B, max_doms, d_model)

        # Post-embedding norm (from nanochat: norm after token embedding)
        x = rms_norm(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, max_doms+1, d_model)

        # Extend mask for CLS (always valid)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Save initial embedding for x0 skip connection
        x0 = x

        # Transformer with per-layer residual scaling + x0 skip
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, full_mask)

        # Final norm
        x = rms_norm(x)

        # CLS token → direction
        return self.direction_head(x[:, 0, :])

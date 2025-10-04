"""
T1: DOM-level transformer for pulse aggregation using FlexAttention.

Aggregates variable-length pulse sequences within each DOM to produce
fixed-size DOM embeddings for downstream event-level processing.
"""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Dict, Optional

from iceaggr.utils import get_logger

logger = get_logger(__name__)


class DOMTransformer(nn.Module):
    """
    DOM-level transformer that aggregates pulses within each DOM.

    Uses FlexAttention with document masking to process variable-length
    pulse sequences efficiently in a single batched forward pass.

    Args:
        d_model: Model dimension (default: 128)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 4)
        d_ff: Feedforward dimension (default: 4 * d_model)
        dropout: Dropout rate (default: 0.1)
        pulse_features: Number of input pulse features (default: 4)
            [time, charge, sensor_id, auxiliary]
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        pulse_features: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model

        # Input projection
        self.input_projection = nn.Linear(pulse_features, d_model)

        # Transformer layers with FlexAttention
        self.layers = nn.ModuleList([
            DOMTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=self.d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        logger.info(
            f"Initialized DOMTransformer: d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, d_ff={self.d_ff}"
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: pulse features → DOM embeddings.

        Args:
            batch: Dict with keys from collate_with_dom_grouping():
                - pulse_features: (total_pulses, 4)
                - pulse_to_dom_idx: (total_pulses,)
                - dom_pulse_counts: (total_doms,)
                - ... (other fields for T2)

        Returns:
            dom_embeddings: (total_doms, d_model) - one embedding per DOM
        """
        pulse_features = batch['pulse_features']  # (total_pulses, 4)
        pulse_to_dom_idx = batch['pulse_to_dom_idx']  # (total_pulses,)
        total_doms = batch['total_doms']

        # Project pulse features to model dimension
        x = self.input_projection(pulse_features)  # (total_pulses, d_model)

        # Create document mask for FlexAttention
        # Pulses attend only to other pulses in the same DOM
        def document_mask(score, b, h, q_idx, kv_idx):
            # Return -inf for pulses from different DOMs, else return score unchanged
            return torch.where(
                pulse_to_dom_idx[q_idx] == pulse_to_dom_idx[kv_idx],
                score,
                float('-inf')
            )

        # Apply transformer layers with FlexAttention
        for layer in self.layers:
            x = layer(x, document_mask, pulse_to_dom_idx)

        x = self.norm(x)  # (total_pulses, d_model)

        # Aggregate pulses to DOM embeddings via mean pooling
        dom_embeddings = self._aggregate_to_doms(
            x, pulse_to_dom_idx, total_doms
        )

        return dom_embeddings  # (total_doms, d_model)

    def _aggregate_to_doms(
        self,
        pulse_embeddings: torch.Tensor,
        pulse_to_dom_idx: torch.Tensor,
        total_doms: int,
    ) -> torch.Tensor:
        """
        Aggregate pulse embeddings to DOM embeddings using mean pooling.

        Args:
            pulse_embeddings: (total_pulses, d_model)
            pulse_to_dom_idx: (total_pulses,) - which DOM each pulse belongs to
            total_doms: Number of unique DOMs in batch

        Returns:
            dom_embeddings: (total_doms, d_model)
        """
        device = pulse_embeddings.device

        # Use scatter_add for efficient aggregation
        dom_embeddings = torch.zeros(
            total_doms, self.d_model, device=device, dtype=pulse_embeddings.dtype
        )

        # Sum embeddings per DOM
        dom_embeddings.scatter_add_(
            dim=0,
            index=pulse_to_dom_idx.unsqueeze(1).expand(-1, self.d_model),
            src=pulse_embeddings,
        )

        # Count pulses per DOM for averaging
        dom_pulse_counts = torch.zeros(total_doms, device=device)
        dom_pulse_counts.scatter_add_(
            dim=0,
            index=pulse_to_dom_idx,
            src=torch.ones_like(pulse_to_dom_idx, dtype=torch.float),
        )

        # Mean pooling (avoid division by zero)
        dom_embeddings = dom_embeddings / dom_pulse_counts.unsqueeze(1).clamp(min=1)

        return dom_embeddings


class DOMTransformerLayer(nn.Module):
    """Single transformer layer with FlexAttention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head attention projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask_fn,
        pulse_to_dom_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with FlexAttention.

        Args:
            x: (total_pulses, d_model)
            mask_fn: Document mask function for FlexAttention
            pulse_to_dom_idx: (total_pulses,) - which DOM each pulse belongs to

        Returns:
            x: (total_pulses, d_model)
        """
        # Multi-head attention with residual
        attn_out = self._flex_attention(x, mask_fn)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feedforward with residual
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x

    def _flex_attention(
        self,
        x: torch.Tensor,
        mask_fn,
    ) -> torch.Tensor:
        """
        Multi-head FlexAttention.

        Args:
            x: (total_pulses, d_model)
            mask_fn: Document mask function

        Returns:
            out: (total_pulses, d_model)
        """
        batch_size = x.shape[0]

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (total_pulses, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (total_pulses, d_model)

        # Reshape for multi-head attention
        # FlexAttention expects: (batch, heads, seq_len, head_dim)
        # We treat total_pulses as seq_len, batch=1
        head_dim = self.d_model // self.n_heads

        q = q.view(1, batch_size, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(1, batch_size, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(1, batch_size, self.n_heads, head_dim).transpose(1, 2)
        # Shape: (1, n_heads, total_pulses, head_dim)

        # Apply FlexAttention with document masking
        out = flex_attention(q, k, v, score_mod=mask_fn)
        # Shape: (1, n_heads, total_pulses, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, self.d_model)

        # Output projection
        out = self.out_proj(out)

        return out

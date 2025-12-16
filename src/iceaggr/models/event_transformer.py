"""
Event-level transformer for aggregating DOM embeddings.

This module processes DOM embeddings across an event to produce
an event-level representation for direction prediction.
"""

import torch
import torch.nn as nn
from typing import Optional


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block.

    Uses pre-layer normalization (LayerNorm before attention/FFN)
    which is more stable for training deep transformers.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        hidden_dim: Feed-forward hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply transformer block.

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            padding_mask: Boolean mask (batch_size, seq_len), True = valid

        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)

        # Create attention mask (True = ignore for MultiheadAttention)
        attn_mask = ~padding_mask

        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=attn_mask,
        )
        x = x + attn_out

        # Feed-forward with pre-norm
        x = x + self.ff(self.norm2(x))

        return x


class EventTransformer(nn.Module):
    """
    Transformer operating on DOM embeddings to produce event representation.

    Uses a CLS token to aggregate information across all DOMs in an event.
    DOM positions are integrated via a learnable projection.

    Args:
        embed_dim: Embedding dimension (default: 64)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer blocks (default: 4)
        hidden_dim: Feed-forward hidden dimension (default: 256)
        max_doms: Maximum DOMs per event (default: 128)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> transformer = EventTransformer(embed_dim=64, max_doms=128)
        >>> dom_embeds = torch.randn(32, 128, 64)  # batch of 32 events
        >>> dom_positions = torch.randn(32, 128, 3)
        >>> mask = torch.ones(32, 128, dtype=torch.bool)
        >>> event_embed = transformer(dom_embeds, dom_positions, mask)
        >>> event_embed.shape  # (32, 64)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        hidden_dim: int = 256,
        max_doms: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_doms = max_doms

        # CLS token for event-level aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Learnable positional encoding
        # +1 for CLS token position
        self.pos_encoding = nn.Parameter(torch.randn(1, max_doms + 1, embed_dim))

        # Geometry integration: project (x,y,z) to embedding space
        self.geometry_proj = nn.Linear(3, embed_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize positional encoding and CLS token."""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_encoding, std=0.02)

    def forward(
        self,
        dom_embeddings: torch.Tensor,
        dom_positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process DOM embeddings to produce event embedding.

        Args:
            dom_embeddings: (batch_size, max_doms, embed_dim)
            dom_positions: (batch_size, max_doms, 3) - x, y, z coordinates
            padding_mask: (batch_size, max_doms) - True for valid DOMs

        Returns:
            Event embeddings (batch_size, embed_dim)
        """
        batch_size = dom_embeddings.shape[0]
        device = dom_embeddings.device

        # Add geometry information
        geom_embed = self.geometry_proj(dom_positions)
        x = dom_embeddings + geom_embed

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, max_doms+1, embed_dim)

        # Update padding mask for CLS (always valid)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, full_mask)

        # Final normalization
        x = self.final_norm(x)

        # Return CLS token embedding
        return x[:, 0, :]

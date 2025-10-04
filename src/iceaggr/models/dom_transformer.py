"""
T1: DOM-level transformer for pulse aggregation using FlexAttention.

Aggregates variable-length pulse sequences within each DOM to produce
fixed-size DOM embeddings for downstream event-level processing.

Uses DOM-level batching with packing: multiple sparse DOMs are packed into
fixed-length sequences to maximize GPU efficiency while controlling memory.
"""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from typing import Dict, Optional, Tuple

from iceaggr.utils import get_logger

logger = get_logger(__name__)


class DOMTransformer(nn.Module):
    """
    DOM-level transformer that aggregates pulses within each DOM.

    Uses FlexAttention with DOM-boundary masking to process packed sequences
    where multiple DOMs may be packed into a single sequence.

    Args:
        d_model: Model dimension (default: 128)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 4)
        d_ff: Feedforward dimension (default: 4 * d_model)
        dropout: Dropout rate (default: 0.1)
        pulse_features: Number of input pulse features (default: 4)
            [time, charge, sensor_id, auxiliary]
        max_seq_len: Maximum sequence length for packing (default: 512)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        pulse_features: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_len = max_seq_len

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
            f"n_layers={n_layers}, d_ff={self.d_ff}, max_seq_len={max_seq_len}"
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        max_batch_size: int = 64
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: packed pulse sequences → DOM embeddings.

        Automatically splits large batches into mini-batches to avoid OOM on extreme events.

        Args:
            batch: Dict with keys from collate_dom_packing():
                - packed_sequences: (bsz, max_seq_len, 4) - packed pulse features
                - dom_boundaries: (bsz, max_seq_len) - DOM ID at each position
                - dom_mask: (bsz, max_seq_len) - 1 for valid pulses, 0 for padding
                - metadata: Dict with DOM-to-event mapping info
            max_batch_size: Maximum number of sequences to process at once (default: 64)

        Returns:
            dom_embeddings: (total_doms, d_model) - one embedding per DOM
            metadata: Dict with reassembly information for T2
        """
        packed_sequences = batch['packed_sequences']  # (bsz, max_seq_len, 4)
        dom_boundaries = batch['dom_boundaries']  # (bsz, max_seq_len)
        dom_mask = batch['dom_mask']  # (bsz, max_seq_len)

        bsz, seq_len, _ = packed_sequences.shape

        # If batch is too large, process in chunks
        if bsz > max_batch_size:
            logger.debug(f"Splitting batch of {bsz} sequences into mini-batches of {max_batch_size}")
            return self._forward_chunked(batch, max_batch_size)

        # Normal forward pass for manageable batches
        # Project pulse features to model dimension
        x = self.input_projection(packed_sequences)  # (bsz, max_seq_len, d_model)

        # Create DOM boundary mask function for FlexAttention
        # Pulses attend only to other pulses in the same DOM
        def dom_boundary_mask(score, b, h, q_idx, kv_idx):
            # Same DOM if dom_boundaries match AND both are valid (not padding)
            same_dom = dom_boundaries[b, q_idx] == dom_boundaries[b, kv_idx]
            both_valid = dom_mask[b, q_idx] * dom_mask[b, kv_idx]
            return torch.where(
                same_dom & both_valid.bool(),
                score,
                float('-inf')
            )

        # Apply transformer layers with FlexAttention
        for layer in self.layers:
            x = layer(x, dom_boundary_mask)

        x = self.norm(x)  # (bsz, max_seq_len, d_model)

        # Aggregate pulses to DOM embeddings
        dom_embeddings = self._aggregate_to_doms(
            x, dom_boundaries, dom_mask, batch['metadata']
        )

        return dom_embeddings, batch['metadata']

    def _forward_chunked(
        self,
        batch: Dict[str, torch.Tensor],
        chunk_size: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process large batch in chunks to avoid OOM.

        Args:
            batch: Full batch dict
            chunk_size: Number of sequences per chunk

        Returns:
            dom_embeddings: (total_doms, d_model)
            metadata: Original metadata
        """
        packed_sequences = batch['packed_sequences']
        dom_boundaries = batch['dom_boundaries']
        dom_mask = batch['dom_mask']
        metadata = batch['metadata']

        bsz = packed_sequences.shape[0]
        device = packed_sequences.device

        # Process in chunks
        all_embeddings = []

        for start_idx in range(0, bsz, chunk_size):
            end_idx = min(start_idx + chunk_size, bsz)

            # Create mini-batch
            mini_batch_seqs = packed_sequences[start_idx:end_idx]
            mini_batch_boundaries = dom_boundaries[start_idx:end_idx]
            mini_batch_mask = dom_mask[start_idx:end_idx]
            mini_batch_global_ids = metadata['global_dom_ids'][start_idx:end_idx]

            # Forward pass on mini-batch
            x = self.input_projection(mini_batch_seqs)

            # Note: dom_boundaries and dom_mask are sliced, so indices b are relative to mini-batch
            def dom_boundary_mask(score, b, h, q_idx, kv_idx):
                same_dom = mini_batch_boundaries[b, q_idx] == mini_batch_boundaries[b, kv_idx]
                both_valid = mini_batch_mask[b, q_idx] * mini_batch_mask[b, kv_idx]
                return torch.where(
                    same_dom & both_valid.bool(),
                    score,
                    float('-inf')
                )

            for layer in self.layers:
                x = layer(x, dom_boundary_mask)

            x = self.norm(x)

            # Store embeddings for this chunk
            all_embeddings.append(x)

        # Concatenate all chunk embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (bsz, seq_len, d_model)

        # Now aggregate to DOMs using full batch metadata
        dom_embeddings = self._aggregate_to_doms(
            all_embeddings, dom_boundaries, dom_mask, metadata
        )

        return dom_embeddings, metadata

    def _aggregate_to_doms(
        self,
        pulse_embeddings: torch.Tensor,
        dom_boundaries: torch.Tensor,
        dom_mask: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Aggregate pulse embeddings to DOM embeddings using mean pooling.

        Args:
            pulse_embeddings: (bsz, max_seq_len, d_model)
            dom_boundaries: (bsz, max_seq_len) - DOM ID at each position
            dom_mask: (bsz, max_seq_len) - 1 for valid pulses, 0 for padding
            metadata: Dict containing 'total_doms' and 'global_dom_ids'

        Returns:
            dom_embeddings: (total_doms, d_model)
        """
        bsz, seq_len, d_model = pulse_embeddings.shape
        total_doms = metadata['total_doms']
        global_dom_ids = metadata['global_dom_ids']  # (bsz, max_seq_len) - global DOM index
        device = pulse_embeddings.device

        # Flatten for scatter
        pulse_embeddings_flat = pulse_embeddings.view(-1, d_model)  # (bsz * seq_len, d_model)
        global_dom_ids_flat = global_dom_ids.view(-1)  # (bsz * seq_len,)
        dom_mask_flat = dom_mask.view(-1)  # (bsz * seq_len,)

        # Initialize DOM embeddings
        dom_embeddings = torch.zeros(total_doms, d_model, device=device, dtype=pulse_embeddings.dtype)
        dom_counts = torch.zeros(total_doms, device=device, dtype=torch.float32)

        # Only aggregate valid pulses (dom_mask == 1)
        valid_mask = dom_mask_flat.bool()
        valid_embeddings = pulse_embeddings_flat[valid_mask]
        valid_dom_ids = global_dom_ids_flat[valid_mask]

        # Sum embeddings per DOM
        dom_embeddings.scatter_add_(
            dim=0,
            index=valid_dom_ids.unsqueeze(1).expand(-1, d_model),
            src=valid_embeddings,
        )

        # Count pulses per DOM
        dom_counts.scatter_add_(
            dim=0,
            index=valid_dom_ids,
            src=torch.ones_like(valid_dom_ids, dtype=torch.float32),
        )

        # Mean pooling (avoid division by zero)
        dom_embeddings = dom_embeddings / dom_counts.unsqueeze(1).clamp(min=1)

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
    ) -> torch.Tensor:
        """
        Forward pass with FlexAttention.

        Args:
            x: (bsz, seq_len, d_model)
            mask_fn: DOM boundary mask function for FlexAttention

        Returns:
            x: (bsz, seq_len, d_model)
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
            x: (bsz, seq_len, d_model)
            mask_fn: DOM boundary mask function

        Returns:
            out: (bsz, seq_len, d_model)
        """
        bsz, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (bsz, seq_len, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (bsz, seq_len, d_model)

        # Reshape for multi-head attention
        # FlexAttention expects: (batch, heads, seq_len, head_dim)
        head_dim = self.d_model // self.n_heads

        q = q.view(bsz, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, head_dim).transpose(1, 2)
        # Shape: (bsz, n_heads, seq_len, head_dim)

        # Apply FlexAttention with DOM boundary masking
        out = flex_attention(q, k, v, score_mod=mask_fn)
        # Shape: (bsz, n_heads, seq_len, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(bsz, seq_len, self.d_model)

        # Output projection
        out = self.out_proj(out)

        return out

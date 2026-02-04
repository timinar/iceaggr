"""
DOM encoder modules for aggregating pulse embeddings per DOM.

This module provides different strategies for encoding variable-length
sequences of pulses within each DOM into a fixed-size DOM embedding.
"""

import torch
import torch.nn as nn
from typing import Optional


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
    """
    Scatter mean using native PyTorch operations.

    Args:
        src: Source tensor (N, D)
        index: Index tensor (N,) mapping each row to an output row
        dim: Dimension to scatter along (must be 0)
        dim_size: Size of output dimension

    Returns:
        Output tensor (dim_size, D) with mean values
    """
    if dim_size is None:
        dim_size = index.max().item() + 1

    # Expand index for broadcasting: (N,) -> (N, D)
    index_expanded = index.unsqueeze(-1).expand_as(src)

    # Sum values
    out = torch.zeros(dim_size, src.shape[1], dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index_expanded, src)

    # Count occurrences
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))

    # Compute mean (avoid division by zero)
    count = count.clamp(min=1).unsqueeze(-1)
    return out / count


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> tuple:
    """
    Scatter max using native PyTorch operations.

    Args:
        src: Source tensor (N, D)
        index: Index tensor (N,) mapping each row to an output row
        dim: Dimension to scatter along (must be 0)
        dim_size: Size of output dimension

    Returns:
        Tuple of (values, indices) where values is (dim_size, D)
    """
    if dim_size is None:
        dim_size = index.max().item() + 1

    # Initialize with very negative values
    out = torch.full((dim_size, src.shape[1]), float('-inf'), dtype=src.dtype, device=src.device)
    out_idx = torch.zeros(dim_size, src.shape[1], dtype=torch.long, device=src.device)

    # Expand index for broadcasting: (N,) -> (N, D)
    index_expanded = index.unsqueeze(-1).expand_as(src)

    # Use scatter_reduce with max (PyTorch 2.0+)
    out = out.scatter_reduce(0, index_expanded, src, reduce='amax', include_self=False)

    # Replace -inf with 0 for empty groups
    out = torch.where(out == float('-inf'), torch.zeros_like(out), out)

    return out, out_idx


class DOMPooling(nn.Module):
    """
    Pooling-based DOM encoder with physics feature preservation.

    Aggregates pulse embeddings within each DOM using scatter operations,
    then concatenates explicit physics features (min_time, sum_charge, std_time)
    that are critical for direction reconstruction but destroyed by pooling.

    Args:
        embed_dim: Embedding dimension (default: 64)
        pool_method: Pooling method - "mean", "max", "mean_max", or "first" (default: "mean_max")
        use_physics_features: Whether to use min_time, sum_charge, std_time (default: True)

    Example:
        >>> encoder = DOMPooling(embed_dim=64, pool_method="mean_max")
        >>> pulse_embeds = torch.randn(1000, 64)  # 1000 pulses
        >>> pulse_to_dom_idx = torch.randint(0, 50, (1000,))  # 50 DOMs
        >>> dom_embeds = encoder(pulse_embeds, pulse_to_dom_idx, num_doms=50)
        >>> dom_embeds.shape  # (50, 64)
    """

    # Number of physics features: min_time, sum_charge, std_time, n_pulses
    N_PHYSICS_FEATURES = 4

    def __init__(
        self,
        embed_dim: int = 64,
        pool_method: str = "mean_max",
        use_physics_features: bool = True,
    ):
        super().__init__()

        assert pool_method in ["mean", "max", "mean_max", "first"], \
            f"pool_method must be 'mean', 'max', 'mean_max', or 'first', got {pool_method}"

        self.embed_dim = embed_dim
        self.pool_method = pool_method
        self.use_physics_features = use_physics_features

        # Determine pooling output dimension
        if pool_method == "mean_max":
            pool_out_dim = embed_dim * 2
        else:
            pool_out_dim = embed_dim

        # Input to projection: pooled embeddings + optional physics features
        proj_in_dim = pool_out_dim + (self.N_PHYSICS_FEATURES if use_physics_features else 0)
        self.projection = nn.Linear(proj_in_dim, embed_dim)

    def forward(
        self,
        pulse_embeddings: torch.Tensor,
        pulse_to_dom_idx: torch.Tensor,
        num_doms: int,
        pulse_idx_in_dom: Optional[torch.Tensor] = None,
        dom_physics: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Aggregate pulse embeddings to DOM embeddings.

        Args:
            pulse_embeddings: (total_pulses, embed_dim)
            pulse_to_dom_idx: (total_pulses,) - DOM index for each pulse
            num_doms: Total number of DOMs in the batch
            pulse_idx_in_dom: (total_pulses,) - Index within DOM (needed for "first" mode)
            dom_physics: Optional dict with 'dom_min_time', 'dom_sum_charge',
                        'dom_std_time', 'dom_pulse_counts' (each (num_doms,))

        Returns:
            DOM embeddings (num_doms, embed_dim)
        """
        # 1. Compute pooled embeddings
        if self.pool_method == "first":
            if pulse_idx_in_dom is None:
                raise ValueError("pulse_idx_in_dom required for 'first' pool_method")
            first_pulse_mask = pulse_idx_in_dom == 0
            first_pulse_embeds = pulse_embeddings[first_pulse_mask]
            first_pulse_dom_idx = pulse_to_dom_idx[first_pulse_mask]
            pooled = torch.zeros(num_doms, self.embed_dim,
                                 dtype=pulse_embeddings.dtype,
                                 device=pulse_embeddings.device)
            pooled[first_pulse_dom_idx] = first_pulse_embeds

        elif self.pool_method == "mean":
            pooled = scatter_mean(
                pulse_embeddings, pulse_to_dom_idx, dim=0, dim_size=num_doms
            )

        elif self.pool_method == "max":
            pooled = scatter_max(
                pulse_embeddings, pulse_to_dom_idx, dim=0, dim_size=num_doms
            )[0]

        else:  # mean_max
            mean_pool = scatter_mean(
                pulse_embeddings, pulse_to_dom_idx, dim=0, dim_size=num_doms
            )
            max_pool = scatter_max(
                pulse_embeddings, pulse_to_dom_idx, dim=0, dim_size=num_doms
            )[0]
            pooled = torch.cat([mean_pool, max_pool], dim=-1)

        # 2. Add physics features if enabled and available
        if self.use_physics_features and dom_physics is not None:
            # Normalize physics features to similar scale as embeddings
            # min_time: (time - 1e4) / 3e4  (same as pulse normalization)
            min_time_norm = (dom_physics['dom_min_time'] - 1e4) / 3e4

            # sum_charge: log10(charge) / 3.0
            sum_charge_norm = torch.log10(dom_physics['dom_sum_charge'].clamp(min=1e-6)) / 3.0

            # std_time: / 3e4 (same scale as time)
            std_time_norm = dom_physics['dom_std_time'] / 3e4

            # n_pulses: log / 3.0 - 1.0 (same as pulse normalization)
            n_pulses_norm = torch.log1p(dom_physics['dom_pulse_counts'].float()) / 3.0 - 1.0

            physics_features = torch.stack([
                min_time_norm, sum_charge_norm, std_time_norm, n_pulses_norm
            ], dim=-1)  # (num_doms, 4)

            # Match dtype with pooled embeddings
            physics_features = physics_features.to(pooled.dtype)

            # Concatenate: pooled + physics
            pooled = torch.cat([pooled, physics_features], dim=-1)

        # 3. Project to embed_dim
        return self.projection(pooled)


class DOMTransformerEncoder(nn.Module):
    """
    Transformer-based DOM encoder for richer pulse aggregation.

    Uses a small transformer to process pulses within each DOM,
    extracting a CLS token as the DOM representation.

    Note: This is more complex and memory-intensive than DOMPooling.
    Use DOMPooling as baseline, then try this for potential improvements.

    Args:
        embed_dim: Embedding dimension (default: 64)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        max_pulses_per_dom: Maximum pulses per DOM for positional encoding (default: 256)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_pulses_per_dom: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_pulses_per_dom = max_pulses_per_dom

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))

        # Positional encoding for pulse order within DOM
        self.pos_encoding = nn.Embedding(max_pulses_per_dom + 1, embed_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fallback pooling for DOMs with too many pulses
        self.fallback_pool = DOMPooling(embed_dim, pool_method="mean_max")

    def forward(
        self,
        pulse_embeddings: torch.Tensor,
        pulse_to_dom_idx: torch.Tensor,
        pulse_idx_in_dom: torch.Tensor,
        dom_pulse_counts: torch.Tensor,
        num_doms: int,
    ) -> torch.Tensor:
        """
        Aggregate pulse embeddings to DOM embeddings using transformer.

        For DOMs with <= max_pulses_per_dom pulses, uses transformer.
        For larger DOMs, falls back to simple pooling.

        Args:
            pulse_embeddings: (total_pulses, embed_dim)
            pulse_to_dom_idx: (total_pulses,) - DOM index for each pulse
            pulse_idx_in_dom: (total_pulses,) - Pulse index within DOM
            dom_pulse_counts: (num_doms,) - Number of pulses per DOM
            num_doms: Total number of DOMs

        Returns:
            DOM embeddings (num_doms, embed_dim)
        """
        device = pulse_embeddings.device

        # Initialize output tensor
        dom_embeddings = torch.zeros(num_doms, self.embed_dim, device=device)

        # Identify small and large DOMs
        small_dom_mask = dom_pulse_counts <= self.max_pulses_per_dom
        large_dom_indices = torch.where(~small_dom_mask)[0]
        small_dom_indices = torch.where(small_dom_mask)[0]

        # Process large DOMs with fallback pooling
        if len(large_dom_indices) > 0:
            # Find pulses belonging to large DOMs
            large_dom_set = set(large_dom_indices.tolist())
            large_pulse_mask = torch.tensor(
                [idx.item() in large_dom_set for idx in pulse_to_dom_idx],
                device=device,
            )

            if large_pulse_mask.any():
                large_embeds = self.fallback_pool(
                    pulse_embeddings[large_pulse_mask],
                    pulse_to_dom_idx[large_pulse_mask],
                    num_doms,
                )
                dom_embeddings[large_dom_indices] = large_embeds[large_dom_indices]

        # Process small DOMs with transformer (batched)
        if len(small_dom_indices) > 0:
            # Group pulses by DOM and process in batches
            max_len = dom_pulse_counts[small_dom_indices].max().item()

            # Pad sequences for batch processing
            n_small_doms = len(small_dom_indices)
            padded_seqs = torch.zeros(n_small_doms, max_len + 1, self.embed_dim, device=device)
            attention_mask = torch.ones(n_small_doms, max_len + 1, dtype=torch.bool, device=device)

            # Add CLS token to all sequences
            padded_seqs[:, 0, :] = self.cls_token

            # Fill in pulse embeddings for each small DOM
            for batch_idx, dom_idx in enumerate(small_dom_indices):
                pulse_mask = pulse_to_dom_idx == dom_idx
                dom_pulses = pulse_embeddings[pulse_mask]
                dom_positions = pulse_idx_in_dom[pulse_mask]

                n_pulses = dom_pulses.shape[0]

                # Add positional encoding (offset by 1 for CLS token)
                positions = torch.clamp(dom_positions, max=self.max_pulses_per_dom - 1)
                pos_embeds = self.pos_encoding(positions + 1)  # +1 to skip CLS position

                padded_seqs[batch_idx, 1:n_pulses + 1, :] = dom_pulses + pos_embeds
                attention_mask[batch_idx, n_pulses + 1:] = False

            # CLS token position encoding
            padded_seqs[:, 0, :] = padded_seqs[:, 0, :] + self.pos_encoding(torch.zeros(1, dtype=torch.long, device=device))

            # Apply transformer (key_padding_mask expects True for ignored positions)
            transformer_out = self.transformer(
                padded_seqs,
                src_key_padding_mask=~attention_mask,
            )

            # Extract CLS token embeddings
            dom_embeddings[small_dom_indices] = transformer_out[:, 0, :]

        return dom_embeddings

"""
Full hierarchical DOM aggregation model for IceCube direction prediction.

This module assembles all components into a complete model:
1. PulseEmbedder: Embeds raw pulse features
2. DOMEncoder: Aggregates pulse embeddings per DOM
3. EventTransformer: Processes DOM embeddings
4. DirectionalHead: Predicts neutrino direction
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .pulse_embedder import PulseEmbedder, build_pulse_features
from .dom_encoder import DOMPooling, DOMTransformerEncoder
from .event_transformer import EventTransformer
from .directional_head import DirectionalHead


class HierarchicalDOMModel(nn.Module):
    """
    Full hierarchical model for neutrino direction prediction.

    Architecture:
        Pulses (N, 8) → PulseEmbedder → (N, embed_dim)
        → DOMEncoder (scatter pooling) → (D, embed_dim)
        → Pad/subsample to (B, max_doms, embed_dim)
        → EventTransformer → (B, embed_dim)
        → DirectionalHead → (B, 3) unit vectors

    Args:
        config: Configuration dictionary with model hyperparameters

    Config options:
        - embed_dim: Embedding dimension (default: 64)
        - max_doms: Maximum DOMs per event (default: 128)
        - pulse_hidden_dims: Hidden dims for pulse MLP (default: [64, 64])
        - dom_encoder_type: "pooling" or "transformer" (default: "pooling")
        - pool_method: "mean", "max", or "mean_max" (default: "mean_max")
        - event_num_heads: Attention heads (default: 8)
        - event_num_layers: Transformer layers (default: 4)
        - event_hidden_dim: FFN hidden dim (default: 256)
        - head_hidden_dim: Direction head hidden dim (default: 128)
        - dropout: Dropout rate (default: 0.1)

    Example:
        >>> config = {"embed_dim": 64, "max_doms": 128}
        >>> model = HierarchicalDOMModel(config)
        >>> batch = {...}  # from collate_with_dom_grouping + geometry
        >>> directions = model(batch)
        >>> directions.shape  # (batch_size, 3)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        if config is None:
            config = {}

        self.config = config

        # Core hyperparameters
        self.embed_dim = config.get('embed_dim', 64)
        self.max_doms = config.get('max_doms', 128)
        dropout = config.get('dropout', 0.1)

        # Pulse embedder
        self.pulse_embedder = PulseEmbedder(
            input_dim=8,
            embed_dim=self.embed_dim,
            hidden_dims=config.get('pulse_hidden_dims', [64, 64]),
            dropout=dropout,
        )

        # DOM encoder
        dom_encoder_type = config.get('dom_encoder_type', 'pooling')
        if dom_encoder_type == 'pooling':
            self.dom_encoder = DOMPooling(
                embed_dim=self.embed_dim,
                pool_method=config.get('pool_method', 'mean_max'),
            )
            self._use_transformer_dom = False
        else:
            self.dom_encoder = DOMTransformerEncoder(
                embed_dim=self.embed_dim,
                num_heads=config.get('dom_num_heads', 4),
                num_layers=config.get('dom_num_layers', 2),
                max_pulses_per_dom=config.get('max_pulses_per_dom', 256),
                dropout=dropout,
            )
            self._use_transformer_dom = True

        # Event transformer
        self.event_transformer = EventTransformer(
            embed_dim=self.embed_dim,
            num_heads=config.get('event_num_heads', 8),
            num_layers=config.get('event_num_layers', 4),
            hidden_dim=config.get('event_hidden_dim', 256),
            max_doms=self.max_doms,
            dropout=dropout,
        )

        # Direction head
        self.direction_head = DirectionalHead(
            embed_dim=self.embed_dim,
            hidden_dim=config.get('head_hidden_dim', 128),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the full model.

        Args:
            batch: Dictionary from collate_with_geometry containing:
                - pulse_features: (total_pulses, 4)
                - pulse_to_dom_idx: (total_pulses,)
                - pulse_idx_in_dom: (total_pulses,)
                - n_pulses_in_dom: (total_pulses,)
                - dom_positions: (total_doms, 3)
                - dom_to_event_idx: (total_doms,)
                - dom_pulse_counts: (total_doms,)
                - event_dom_counts: (batch_size,)
                - total_doms: int
                - batch_size: int
                - targets: (batch_size, 2) - optional

        Returns:
            Predicted directions (batch_size, 3) as unit vectors
        """
        # 1. Build full pulse features (8 dims)
        pulse_features = build_pulse_features(
            raw_features=batch['pulse_features'],
            dom_positions=batch['dom_positions'],
            pulse_to_dom_idx=batch['pulse_to_dom_idx'],
            n_pulses_in_dom=batch['n_pulses_in_dom'],
            pulse_idx_in_dom=batch['pulse_idx_in_dom'],
        )

        # 2. Embed pulses
        pulse_embeds = self.pulse_embedder(pulse_features)

        # 3. Aggregate to DOM level
        if self._use_transformer_dom:
            dom_embeds = self.dom_encoder(
                pulse_embeds,
                batch['pulse_to_dom_idx'],
                batch['pulse_idx_in_dom'],
                batch['dom_pulse_counts'],
                batch['total_doms'],
            )
        else:
            dom_embeds = self.dom_encoder(
                pulse_embeds,
                batch['pulse_to_dom_idx'],
                batch['total_doms'],
                pulse_idx_in_dom=batch['pulse_idx_in_dom'],
            )

        # 4. Reshape to (batch_size, max_doms, embed_dim) with padding/subsampling
        dom_embeds_padded, dom_positions_padded, padding_mask = self._pad_to_batch(
            dom_embeds,
            batch['dom_positions'],
            batch['dom_to_event_idx'],
            batch['event_dom_counts'],
            batch['batch_size'],
        )

        # 5. Event transformer
        event_embeds = self.event_transformer(
            dom_embeds_padded,
            dom_positions_padded,
            padding_mask,
        )

        # 6. Predict direction
        directions = self.direction_head(event_embeds)

        return directions

    def _pad_to_batch(
        self,
        dom_embeds: torch.Tensor,
        dom_positions: torch.Tensor,
        dom_to_event_idx: torch.Tensor,
        event_dom_counts: torch.Tensor,
        batch_size: int,
    ) -> tuple:
        """
        Pad/subsample DOMs to fixed (batch_size, max_doms, embed_dim).

        If an event has more than max_doms DOMs, randomly subsample.
        If fewer, zero-pad.

        Args:
            dom_embeds: (total_doms, embed_dim)
            dom_positions: (total_doms, 3)
            dom_to_event_idx: (total_doms,) - event index for each DOM
            event_dom_counts: (batch_size,) - DOMs per event
            batch_size: Number of events

        Returns:
            Tuple of:
                - dom_embeds_padded: (batch_size, max_doms, embed_dim)
                - dom_positions_padded: (batch_size, max_doms, 3)
                - padding_mask: (batch_size, max_doms) - True for valid
        """
        device = dom_embeds.device

        # Initialize padded tensors
        dom_embeds_padded = torch.zeros(
            batch_size, self.max_doms, self.embed_dim, device=device
        )
        dom_positions_padded = torch.zeros(
            batch_size, self.max_doms, 3, device=device
        )
        padding_mask = torch.zeros(
            batch_size, self.max_doms, dtype=torch.bool, device=device
        )

        # Fill in for each event
        dom_offset = 0
        for event_idx in range(batch_size):
            n_doms = event_dom_counts[event_idx].item()

            # Get this event's DOM data
            event_dom_embeds = dom_embeds[dom_offset:dom_offset + n_doms]
            event_dom_positions = dom_positions[dom_offset:dom_offset + n_doms]

            if n_doms > self.max_doms:
                # Random subsample
                indices = torch.randperm(n_doms, device=device)[:self.max_doms]
                event_dom_embeds = event_dom_embeds[indices]
                event_dom_positions = event_dom_positions[indices]
                n_doms = self.max_doms

            dom_embeds_padded[event_idx, :n_doms] = event_dom_embeds
            dom_positions_padded[event_idx, :n_doms] = event_dom_positions
            padding_mask[event_idx, :n_doms] = True

            dom_offset += event_dom_counts[event_idx].item()

        return dom_embeds_padded, dom_positions_padded, padding_mask


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

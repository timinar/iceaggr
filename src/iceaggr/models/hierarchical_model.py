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
        Vectorized pad/subsample DOMs to fixed (batch_size, max_doms, embed_dim).

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
        total_doms = dom_embeds.shape[0]

        # Compute within-event DOM index using cumsum trick
        # dom_starts[i] = index of first DOM in event i
        dom_starts = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        dom_starts[1:] = event_dom_counts.cumsum(0)

        # For each DOM, compute its index within its event (0, 1, 2, ...)
        dom_idx_in_event = torch.arange(total_doms, device=device) - dom_starts[dom_to_event_idx]

        # Clamp counts to max_doms for output sizing
        clamped_counts = event_dom_counts.clamp(max=self.max_doms)

        # Determine which DOMs to keep (handle subsampling for large events)
        # For events with n_doms <= max_doms: keep all
        # For events with n_doms > max_doms: keep random subset
        needs_subsample = event_dom_counts > self.max_doms

        if needs_subsample.any():
            # Generate random priorities for subsampling
            random_priority = torch.rand(total_doms, device=device)

            # For DOMs in events needing subsampling, rank by random priority
            # Keep DOM if its rank < max_doms
            keep_mask = torch.ones(total_doms, dtype=torch.bool, device=device)

            # Process events needing subsampling
            for event_idx in needs_subsample.nonzero(as_tuple=True)[0]:
                start = dom_starts[event_idx]
                end = dom_starts[event_idx + 1]
                event_priorities = random_priority[start:end]
                # Get indices of top max_doms by priority
                _, top_indices = event_priorities.topk(self.max_doms, largest=True)
                # Mark all as not kept, then mark top ones as kept
                keep_mask[start:end] = False
                keep_mask[start + top_indices] = True

            # Filter to kept DOMs
            kept_indices = keep_mask.nonzero(as_tuple=True)[0]
            dom_embeds = dom_embeds[kept_indices]
            dom_positions = dom_positions[kept_indices]
            dom_to_event_idx = dom_to_event_idx[kept_indices]

            # Recompute within-event indices for kept DOMs
            # Use cumsum trick again
            kept_dom_starts = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
            kept_dom_starts[1:] = clamped_counts.cumsum(0)
            dom_idx_in_event = torch.arange(dom_embeds.shape[0], device=device) - kept_dom_starts[dom_to_event_idx]
        else:
            # No subsampling needed - dom_idx_in_event already computed
            pass

        # Now scatter into padded tensors
        # Valid positions are where dom_idx_in_event < max_doms
        valid_mask = dom_idx_in_event < self.max_doms
        valid_event_idx = dom_to_event_idx[valid_mask]
        valid_dom_idx = dom_idx_in_event[valid_mask]
        valid_embeds = dom_embeds[valid_mask]
        valid_positions = dom_positions[valid_mask]

        # Initialize output tensors (match input dtype for AMP compatibility)
        dom_embeds_padded = torch.zeros(
            batch_size, self.max_doms, self.embed_dim, device=device, dtype=valid_embeds.dtype
        )
        dom_positions_padded = torch.zeros(
            batch_size, self.max_doms, 3, device=device, dtype=valid_positions.dtype
        )
        padding_mask = torch.zeros(
            batch_size, self.max_doms, dtype=torch.bool, device=device
        )

        # Scatter embeddings and positions
        dom_embeds_padded[valid_event_idx, valid_dom_idx] = valid_embeds
        dom_positions_padded[valid_event_idx, valid_dom_idx] = valid_positions
        padding_mask[valid_event_idx, valid_dom_idx] = True

        return dom_embeds_padded, dom_positions_padded, padding_mask


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

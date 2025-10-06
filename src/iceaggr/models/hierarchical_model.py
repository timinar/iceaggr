"""
Hierarchical neutrino reconstruction model for IceCube.

Two-stage architecture:
1. DeepSets DOM Encoder: Aggregates pulses within each DOM
2. Event Transformer: Aggregates DOM embeddings to predict direction

This is the main model combining all components.
"""

import torch
import torch.nn as nn
from typing import Dict

from .deepsets_dom_encoder import DeepSetsDOMEncoder
from .event_transformer import EventTransformer
from iceaggr.utils import get_logger

logger = get_logger(__name__)


class HierarchicalIceCubeModel(nn.Module):
    """
    Hierarchical model for IceCube neutrino direction reconstruction.

    Architecture:
        Pulses → DeepSets DOM Encoder → DOM embeddings → Event Transformer → Direction

    Args:
        # DOM encoder (DeepSets) params
        d_pulse: Dimension of pulse features (default: 4)
        d_dom_embedding: Output dimension of DOM embeddings
        dom_latent_dim: Hidden dimension in DeepSets encoder
        dom_hidden_dim: Hidden dimension in DeepSets MLPs

        # Event transformer params
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension (default: 4 * d_model)

        # Shared params
        dropout: Dropout probability
        use_geometry: Whether to use DOM geometry in T2
    """

    def __init__(
        self,
        # DOM encoder params
        d_pulse: int = 4,
        d_dom_embedding: int = 128,
        dom_latent_dim: int = 128,
        dom_hidden_dim: int = 256,

        # Event transformer params
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = None,

        # Shared params
        dropout: float = 0.1,
        use_geometry: bool = True
    ):
        super().__init__()

        self.d_pulse = d_pulse
        self.d_dom_embedding = d_dom_embedding
        self.use_geometry = use_geometry

        # Stage 1: DOM-level encoder (DeepSets)
        self.dom_encoder = DeepSetsDOMEncoder(
            d_pulse=d_pulse,
            d_relative=6,  # Fixed: 6 relative features
            d_latent=dom_latent_dim,
            d_output=d_dom_embedding,
            hidden_dim=dom_hidden_dim,
            dropout=dropout
        )

        # Stage 2: Event-level transformer
        self.event_transformer = EventTransformer(
            d_input=d_dom_embedding,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_geometry=use_geometry
        )

        # Geometry extraction and normalization
        # IceCube detector spans ~500m in each direction
        self.geometry_scale = 500.0

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"HierarchicalIceCubeModel initialized")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  DOM encoder: {sum(p.numel() for p in self.dom_encoder.parameters()):,}")
        logger.info(f"  Event transformer: {sum(p.numel() for p in self.event_transformer.parameters()):,}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through hierarchical model.

        Args:
            batch: Dict with keys:
                - pulse_features: (total_pulses, 4) pulse features [time, charge, sensor_id, auxiliary]
                - pulse_to_dom_idx: (total_pulses,) which DOM each pulse belongs to
                - num_doms: int, total number of DOMs in batch
                - dom_to_event_idx: (num_doms,) which event each DOM belongs to
                - dom_ids: (num_doms,) original sensor IDs
                - batch_size: int, number of events in batch
                - (optional) dom_geometry: (num_doms, 3) pre-computed geometry

        Returns:
            predictions: (batch_size, 2) neutrino directions [azimuth, zenith]
        """
        pulse_features = batch['pulse_features']
        pulse_to_dom_idx = batch['pulse_to_dom_idx']
        num_doms = batch['num_doms']
        dom_to_event_idx = batch['dom_to_event_idx']
        batch_size = batch['batch_size']

        # Stage 1: Encode pulses to DOM embeddings
        dom_embeddings = self.dom_encoder(
            pulse_features=pulse_features,
            pulse_to_dom_idx=pulse_to_dom_idx,
            num_doms=num_doms
        )  # (num_doms, d_dom_embedding)

        # Extract geometry if needed
        geometry = None
        if self.use_geometry:
            if 'dom_geometry' in batch:
                geometry = batch['dom_geometry']
            else:
                # We need to provide geometry lookup
                # For now, we'll require it to be in the batch
                logger.warning("use_geometry=True but no dom_geometry in batch. Proceeding without geometry.")

        # Stage 2: Aggregate DOM embeddings to predict event direction
        predictions = self.event_transformer(
            dom_embeddings=dom_embeddings,
            dom_to_event_idx=dom_to_event_idx,
            batch_size=batch_size,
            geometry=geometry
        )  # (batch_size, 2)

        return predictions

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute angular error loss.

        Uses mean absolute error on (azimuth, zenith) as a simple baseline.
        Can be replaced with more sophisticated angular distance metrics.

        Args:
            predictions: (batch_size, 2) predicted [azimuth, zenith]
            targets: (batch_size, 2) true [azimuth, zenith]
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: scalar or (batch_size,) depending on reduction
        """
        # Simple L1 loss on angles
        # TODO: Use proper angular distance (great circle distance)
        loss = torch.abs(predictions - targets).sum(dim=1)  # (batch_size,)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

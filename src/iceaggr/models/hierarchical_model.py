"""
Hierarchical neutrino reconstruction model for IceCube.

Two-stage architecture:
1. DeepSets DOM Encoder: Aggregates pulses within each DOM
2. Event Transformer: Aggregates DOM embeddings to predict direction

This is the main model combining all components.
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml
import pandas as pd

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
        use_geometry: bool = True,
        config_path: str | None = None
    ):
        super().__init__()

        self.d_pulse = d_pulse
        self.d_dom_embedding = d_dom_embedding
        self.use_geometry = use_geometry

        # Load geometry lookup table
        self.geometry_table = self._load_geometry(config_path)

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

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"HierarchicalIceCubeModel initialized")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  DOM encoder: {sum(p.numel() for p in self.dom_encoder.parameters()):,}")
        logger.info(f"  Event transformer: {sum(p.numel() for p in self.event_transformer.parameters()):,}")

    def _load_geometry(self, config_path: str | None = None) -> torch.Tensor:
        """
        Load sensor geometry lookup table.

        Returns:
            geometry_table: (5160, 3) tensor of normalized (x, y, z) coordinates
        """
        # Load config to get data root
        if config_path is None:
            config_path = Path(__file__).parent.parent / "data" / "data_config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        data_root = Path(config["data"]["root"])
        geometry_path = data_root / "sensor_geometry.csv"

        logger.info(f"Loading geometry from {geometry_path}")
        geometry_df = pd.read_csv(geometry_path)

        # Convert to tensor and normalize
        # Assumes columns are [sensor_id, x, y, z] or [x, y, z]
        if 'sensor_id' in geometry_df.columns:
            coords = geometry_df[['x', 'y', 'z']].values
        else:
            coords = geometry_df.values[:, :3]

        geometry_tensor = torch.from_numpy(coords).float()
        geometry_tensor = geometry_tensor / 500.0  # Normalize by detector scale

        logger.info(f"  Loaded geometry for {len(geometry_tensor)} sensors")
        logger.info(f"  Shape: {geometry_tensor.shape}")

        return geometry_tensor

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through hierarchical model.

        Args:
            batch: Dict with keys:
                - pulse_features: (total_pulses, 4) RAW pulse features [time, charge, sensor_id, auxiliary]
                - pulse_to_dom_idx: (total_pulses,) which DOM each pulse belongs to
                - num_doms: int, total number of DOMs in batch
                - dom_to_event_idx: (num_doms,) which event each DOM belongs to
                - dom_ids: (num_doms,) original sensor IDs
                - batch_size: int, number of events in batch

        Returns:
            predictions: (batch_size, 2) neutrino directions [azimuth, zenith]
        """
        pulse_features = batch['pulse_features']  # (total_pulses, 4) - RAW
        pulse_to_dom_idx = batch['pulse_to_dom_idx']
        num_doms = batch['num_doms']
        dom_to_event_idx = batch['dom_to_event_idx']
        dom_ids = batch['dom_ids']  # (num_doms,) sensor IDs
        batch_size = batch['batch_size']

        # Normalize pulse features (CRITICAL for training stability!)
        pulse_features_normalized = self._normalize_pulse_features(pulse_features)

        # Stage 1: Encode pulses to DOM embeddings
        dom_embeddings = self.dom_encoder(
            pulse_features=pulse_features_normalized,
            pulse_to_dom_idx=pulse_to_dom_idx,
            num_doms=num_doms
        )  # (num_doms, d_dom_embedding)

        # Lookup geometry from sensor IDs
        geometry = None
        if self.use_geometry:
            geometry = self._lookup_geometry(dom_ids)  # (num_doms, 3)

        # Stage 2: Aggregate DOM embeddings to predict event direction
        predictions = self.event_transformer(
            dom_embeddings=dom_embeddings,
            dom_to_event_idx=dom_to_event_idx,
            batch_size=batch_size,
            geometry=geometry
        )  # (batch_size, 2)

        return predictions

    def _normalize_pulse_features(self, pulse_features: torch.Tensor) -> torch.Tensor:
        """
        Normalize pulse features using fixed normalization scheme.

        Args:
            pulse_features: (n_pulses, 4) [time, charge, sensor_id, auxiliary]

        Returns:
            normalized: (n_pulses, 4) normalized features
        """
        time = pulse_features[:, 0]
        charge = pulse_features[:, 1]
        sensor_id = pulse_features[:, 2]
        auxiliary = pulse_features[:, 3]

        # Fixed normalization (from CLAUDE.md)
        time_norm = (time - 1e4) / 3e4
        charge_norm = torch.log10(charge + 1e-8) / 3.0
        sensor_id_norm = sensor_id / 5160.0
        auxiliary_norm = auxiliary  # auxiliary is already 0 or 1

        normalized = torch.stack([time_norm, charge_norm, sensor_id_norm, auxiliary_norm], dim=1)
        return normalized

    def _lookup_geometry(self, dom_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup geometry coordinates for given sensor IDs.

        Args:
            dom_ids: (num_doms,) sensor IDs

        Returns:
            geometry: (num_doms, 3) normalized (x, y, z) coordinates
        """
        # Move geometry table to same device as dom_ids
        if self.geometry_table.device != dom_ids.device:
            self.geometry_table = self.geometry_table.to(dom_ids.device)

        # Lookup coordinates by sensor ID (direct indexing)
        geometry = self.geometry_table[dom_ids.long()]  # (num_doms, 3)

        return geometry

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

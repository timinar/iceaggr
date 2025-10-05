"""
T2-only model: Skip T1 DOM aggregation, use first pulse from each DOM.

This is a diagnostic model to test if T2 can train independently.
Instead of aggregating pulses with T1, we simply use the features of the
first pulse in each DOM as input to T2.
"""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from iceaggr.utils import get_logger

logger = get_logger(__name__)


class T2FirstPulseModel(nn.Module):
    """
    Simplified model that skips T1 and uses first pulse features per DOM.

    For each DOM with pulses, we take the first pulse's features
    (time, charge, sensor_id, auxiliary) and use them directly as
    "DOM embeddings" for T2.

    This allows testing if T2 can train independently of T1.

    Args:
        d_model: Model dimension (default: 128)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 4)
        d_ff: Feedforward dimension (default: 4 * d_model)
        dropout: Dropout rate (default: 0.1)
        pulse_features: Number of input pulse features (default: 4)
        geometry_path: Path to sensor_geometry.csv (default: auto-detect)
        max_doms: Maximum DOMs per event for batching (default: 2048)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        pulse_features: int = 4,
        geometry_path: Optional[str] = None,
        max_doms: int = 2048,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_doms = max_doms

        # Load sensor geometry
        self.sensor_geometry = self._load_geometry(geometry_path)

        # Project combined features to d_model
        # Input: [time, charge, auxiliary, x, y, z] (6 features)
        # Note: sensor_id is NOT normalized - replaced with actual geometry!
        self.pulse_projection = nn.Linear(6, d_model)

        # Transformer layers (same as T2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=self.d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Prediction head: aggregated embedding → unit vector (x, y, z)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),  # Outputs (x, y, z) before normalization
        )

        logger.info(
            f"Initialized T2FirstPulseModel: d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, d_ff={self.d_ff}, max_doms={max_doms}"
        )

    def _load_geometry(self, geometry_path: Optional[str] = None) -> torch.Tensor:
        """Load sensor geometry (x,y,z) for all DOMs."""
        if geometry_path is None:
            # Auto-detect from data config
            config_path = Path(__file__).parent.parent / "data" / "data_config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Could not find data_config.yaml at {config_path}. "
                    f"Please provide geometry_path explicitly."
                )

            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

            data_root = Path(config["data"]["root"])
            geometry_path = data_root / "sensor_geometry.csv"

        logger.info(f"Loading sensor geometry from {geometry_path}")
        df = pd.read_csv(geometry_path)

        # Extract (x,y,z) coordinates, indexed by sensor_id
        geometry = torch.from_numpy(df[['x', 'y', 'z']].values).float()

        logger.info(f"Loaded geometry for {len(geometry)} sensors")
        return geometry

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass: first pulse features per DOM → event predictions.

        Args:
            batch: Dict with keys:
                - packed_sequences: (bsz, max_seq_len, 4) - packed pulse features
                - dom_boundaries: (bsz, max_seq_len) - DOM ID at each position
                - dom_mask: (bsz, max_seq_len) - 1 for valid pulses, 0 for padding
                - metadata: Dict with DOM-to-event mapping info

        Returns:
            predictions: (batch_size, 3) - unit vector (x, y, z) predictions
        """
        packed_sequences = batch['packed_sequences']  # (bsz, max_seq_len, 4)
        dom_boundaries = batch['dom_boundaries']  # (bsz, max_seq_len)
        dom_mask = batch['dom_mask']  # (bsz, max_seq_len)
        metadata = batch['metadata']

        device = packed_sequences.device

        # Extract first pulse per DOM (now returns 6 features: time, charge, aux, x, y, z)
        dom_features_combined = self._extract_first_pulses(
            packed_sequences, dom_boundaries, dom_mask, metadata
        )
        # dom_features_combined: (total_doms, 6) - [time, charge, auxiliary, x, y, z] all normalized

        # Project combined features to d_model
        dom_embeddings = self.pulse_projection(dom_features_combined)  # (total_doms, d_model)

        # Pack DOMs into batched sequences (one event per sequence)
        dom_to_event_idx = metadata['dom_to_event_idx']
        batch_size = metadata['event_ids'].shape[0]

        event_sequences, padding_mask = self._pack_events(
            dom_embeddings, dom_to_event_idx, batch_size
        )

        # Apply transformer
        transformed = self.transformer(
            event_sequences,
            src_key_padding_mask=padding_mask
        )  # (batch_size, max_doms, d_model)

        # Global aggregation: mean pool over DOMs (excluding padding)
        mask = ~padding_mask.unsqueeze(-1)  # (batch_size, max_doms, 1)
        masked_features = transformed * mask

        event_embedding = masked_features.sum(dim=1)  # (batch_size, d_model)
        valid_doms = mask.sum(dim=1).clamp(min=1)  # (batch_size, 1)
        event_embedding = event_embedding / valid_doms  # Mean pooling

        # Predict direction as unit vector
        vector = self.prediction_head(event_embedding)  # (batch_size, 3)

        # Normalize to unit sphere
        norm = torch.sqrt(torch.sum(vector**2, dim=1, keepdim=True))
        unit_vector = vector / (norm + 1e-8)  # (batch_size, 3)

        return unit_vector

    def _extract_first_pulses(
        self,
        packed_sequences: torch.Tensor,
        dom_boundaries: torch.Tensor,
        dom_mask: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract first pulse features from each DOM, combined with geometry.

        Args:
            packed_sequences: (bsz, max_seq_len, 4) - packed pulse features
            dom_boundaries: (bsz, max_seq_len) - DOM ID at each position
            dom_mask: (bsz, max_seq_len) - 1 for valid pulses, 0 for padding
            metadata: Dict containing 'total_doms', 'global_dom_ids', 'sensor_ids'

        Returns:
            combined_features: (total_doms, 6) - [time, charge, auxiliary, x, y, z] normalized
        """
        bsz, seq_len, _ = packed_sequences.shape
        total_doms = metadata['total_doms']
        global_dom_ids = metadata['global_dom_ids']  # (bsz, max_seq_len)
        sensor_ids = metadata['sensor_ids']  # (total_doms,)
        device = packed_sequences.device

        # Normalize pulse features (CRITICAL for stable training!)
        time = packed_sequences[..., 0]
        charge = packed_sequences[..., 1]
        # Note: sensor_id (feature 2) is NOT used - we use geometry instead!
        auxiliary = packed_sequences[..., 3]

        time_normalized = (time - 1e4) / 3e4
        charge_normalized = torch.log10(charge + 1e-8) / 3.0

        normalized_sequences = torch.stack([
            time_normalized,
            charge_normalized,
            auxiliary
        ], dim=-1)  # (bsz, max_seq_len, 3)

        # Initialize first pulse features for each DOM
        first_pulses = torch.zeros(total_doms, 3, device=device, dtype=normalized_sequences.dtype)
        dom_found = torch.zeros(total_doms, device=device, dtype=torch.bool)

        # Flatten for easier processing
        normalized_flat = normalized_sequences.view(-1, 3)  # (bsz * seq_len, 3)
        global_dom_ids_flat = global_dom_ids.view(-1)  # (bsz * seq_len,)
        dom_mask_flat = dom_mask.view(-1)  # (bsz * seq_len,)

        # For each valid pulse, check if it's the first for its DOM
        valid_mask = dom_mask_flat.bool()
        valid_features = normalized_flat[valid_mask]  # (n_valid_pulses, 3)
        valid_dom_ids = global_dom_ids_flat[valid_mask]  # (n_valid_pulses,)

        # Process pulses in order (first occurrence = first pulse)
        for pulse_idx in range(len(valid_dom_ids)):
            dom_id = valid_dom_ids[pulse_idx]
            if not dom_found[dom_id]:
                first_pulses[dom_id] = valid_features[pulse_idx]
                dom_found[dom_id] = True

        # Get geometry for each DOM
        if self.sensor_geometry.device != device:
            self.sensor_geometry = self.sensor_geometry.to(device)
        dom_geometry = self.sensor_geometry[sensor_ids]  # (total_doms, 3)
        dom_geometry_normalized = dom_geometry / 500.0  # Normalize by detector scale

        # Combine: [time, charge, auxiliary, x, y, z]
        combined_features = torch.cat([first_pulses, dom_geometry_normalized], dim=1)  # (total_doms, 6)

        return combined_features

    def _pack_events(
        self,
        dom_features: torch.Tensor,
        dom_to_event_idx: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack variable-length DOM sequences into batched tensor with padding.

        Same as EventTransformer._pack_events()
        """
        device = dom_features.device
        d_model = dom_features.shape[1]

        # Count DOMs per event
        doms_per_event = torch.zeros(batch_size, dtype=torch.long, device=device)
        for event_idx in range(batch_size):
            doms_per_event[event_idx] = (dom_to_event_idx == event_idx).sum()

        max_doms = doms_per_event.max().item()

        if max_doms > self.max_doms:
            logger.warning(
                f"Event has {max_doms} DOMs, exceeding max_doms={self.max_doms}. "
                f"Consider increasing max_doms."
            )

        # Initialize padded tensors
        packed_sequences = torch.zeros(
            batch_size, max_doms, d_model, device=device, dtype=dom_features.dtype
        )
        padding_mask = torch.ones(batch_size, max_doms, device=device, dtype=torch.bool)

        # Pack each event
        for event_idx in range(batch_size):
            event_mask = (dom_to_event_idx == event_idx)
            event_doms = dom_features[event_mask]  # (n_doms_in_event, d_model)
            n_doms = event_doms.shape[0]

            if n_doms > 0:
                packed_sequences[event_idx, :n_doms] = event_doms
                padding_mask[event_idx, :n_doms] = False

        return packed_sequences, padding_mask

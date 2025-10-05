"""
T2-only model with VECTORIZED operations (no Python loops).

This is an optimized version of T2FirstPulseModel that replaces Python loops
with scatter/gather operations for better performance.
"""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from iceaggr.utils import get_logger

logger = get_logger(__name__)


class T2FirstPulseModelVectorized(nn.Module):
    """
    Vectorized version of T2FirstPulseModel - no Python loops!

    Optimizations:
    1. scatter_reduce for extracting first pulses (replaces loop over pulses)
    2. bincount + cumsum for packing events (replaces loops over batch)

    Same architecture as T2FirstPulseModel, just faster.
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
        # Hidden layer matches transformer MLP dimension (d_ff)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, 3),  # Outputs (x, y, z) before normalization
        )

        logger.info(
            f"Initialized T2FirstPulseModelVectorized: d_model={d_model}, n_heads={n_heads}, "
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

        # Extract first pulse per DOM (VECTORIZED!)
        dom_features_combined = self._extract_first_pulses_vectorized(
            packed_sequences, dom_boundaries, dom_mask, metadata
        )
        # dom_features_combined: (total_doms, 6) - [time, charge, auxiliary, x, y, z] all normalized

        # Project combined features to d_model
        dom_embeddings = self.pulse_projection(dom_features_combined)  # (total_doms, d_model)

        # Pack DOMs into batched sequences (VECTORIZED!)
        dom_to_event_idx = metadata['dom_to_event_idx']
        batch_size = metadata['event_ids'].shape[0]

        event_sequences, padding_mask = self._pack_events_vectorized(
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

    def _extract_first_pulses_vectorized(
        self,
        packed_sequences: torch.Tensor,
        dom_boundaries: torch.Tensor,
        dom_mask: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        VECTORIZED: Extract first pulse features from each DOM using scatter_reduce.

        This replaces the Python loop with GPU operations.

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
        auxiliary = packed_sequences[..., 3]

        time_normalized = (time - 1e4) / 3e4
        charge_normalized = torch.log10(charge + 1e-8) / 3.0

        normalized_sequences = torch.stack([
            time_normalized,
            charge_normalized,
            auxiliary
        ], dim=-1)  # (bsz, max_seq_len, 3)

        # Flatten for scatter operations
        normalized_flat = normalized_sequences.reshape(-1, 3)  # (bsz * seq_len, 3)
        global_dom_ids_flat = global_dom_ids.reshape(-1)  # (bsz * seq_len,)
        dom_mask_flat = dom_mask.reshape(-1)  # (bsz * seq_len,)

        # Filter to valid pulses only
        valid_mask = dom_mask_flat.bool()
        valid_features = normalized_flat[valid_mask]  # (n_valid_pulses, 3)
        valid_dom_ids = global_dom_ids_flat[valid_mask]  # (n_valid_pulses,)

        # VECTORIZED: Use scatter to get first pulse per DOM
        # We need to find the FIRST occurrence of each DOM ID
        # Strategy: Create position indices, scatter with min reduction

        # Create position index for each pulse
        pulse_positions = torch.arange(len(valid_dom_ids), device=device)

        # Scatter positions to find first occurrence (min position) per DOM
        first_positions = torch.full((total_doms,), fill_value=len(valid_dom_ids),
                                     dtype=torch.long, device=device)
        first_positions.scatter_reduce_(
            dim=0,
            index=valid_dom_ids,
            src=pulse_positions,
            reduce='min',
            include_self=False
        )

        # Gather features at first positions
        # For DOMs with no pulses, first_positions will be len(valid_dom_ids) (out of bounds)
        # We need to handle this safely
        first_pulses = torch.zeros(total_doms, 3, device=device, dtype=normalized_flat.dtype)

        # Mask for DOMs that actually have pulses
        has_pulses = first_positions < len(valid_dom_ids)

        # Gather first pulse features for DOMs that have pulses
        if has_pulses.any():
            valid_first_positions = first_positions[has_pulses]
            first_pulses[has_pulses] = valid_features[valid_first_positions]

        # Get geometry for each DOM
        if self.sensor_geometry.device != device:
            self.sensor_geometry = self.sensor_geometry.to(device)
        dom_geometry = self.sensor_geometry[sensor_ids]  # (total_doms, 3)
        dom_geometry_normalized = dom_geometry / 500.0  # Normalize by detector scale

        # Combine: [time, charge, auxiliary, x, y, z]
        combined_features = torch.cat([first_pulses, dom_geometry_normalized], dim=1)  # (total_doms, 6)

        return combined_features

    def _pack_events_vectorized(
        self,
        dom_features: torch.Tensor,
        dom_to_event_idx: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VECTORIZED: Pack variable-length DOM sequences into batched tensor.

        This replaces the Python loops with bincount and advanced indexing.

        Events with more than max_doms are clipped to max_doms.
        """
        device = dom_features.device
        d_model = dom_features.shape[1]

        # Count DOMs per event using bincount (VECTORIZED!)
        doms_per_event = torch.bincount(
            dom_to_event_idx,
            minlength=batch_size
        )  # (batch_size,)

        # Clip to max_doms
        max_doms_in_batch = min(doms_per_event.max().item(), self.max_doms)

        # Initialize padded tensors
        packed_sequences = torch.zeros(
            batch_size, max_doms_in_batch, d_model, device=device, dtype=dom_features.dtype
        )
        padding_mask = torch.ones(batch_size, max_doms_in_batch, device=device, dtype=torch.bool)

        # Create indices for scatter operation
        # For each DOM, we need (event_idx, position_in_event)

        # Sort DOMs by event index for easier processing
        sorted_indices = torch.argsort(dom_to_event_idx)
        sorted_dom_features = dom_features[sorted_indices]
        sorted_event_ids = dom_to_event_idx[sorted_indices]

        # Compute cumulative sum to get starting index for each event
        cumsum_doms = torch.cat([
            torch.tensor([0], device=device),
            doms_per_event.cumsum(dim=0)[:-1]
        ])  # (batch_size,) - starting index for each event

        # Create position within event for each DOM
        # This is like: [0,1,2, 0,1,2,3,4, 0,1, ...] for events with [3, 5, 2, ...] DOMs
        positions = torch.arange(len(sorted_dom_features), device=device) - cumsum_doms[sorted_event_ids]

        # Clip positions to max_doms (discard DOMs beyond max)
        valid_mask = positions < self.max_doms

        if valid_mask.any():
            valid_event_ids = sorted_event_ids[valid_mask]
            valid_positions = positions[valid_mask]
            valid_features = sorted_dom_features[valid_mask]

            # Pack using advanced indexing (VECTORIZED!)
            packed_sequences[valid_event_ids, valid_positions] = valid_features
            padding_mask[valid_event_ids, valid_positions] = False

        return packed_sequences, padding_mask

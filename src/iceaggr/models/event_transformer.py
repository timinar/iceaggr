"""
T2: Event-level transformer for DOM embedding aggregation.

Aggregates DOM embeddings across the entire event to predict neutrino direction.
Uses standard transformer with geometry-aware positional encoding.
"""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from iceaggr.utils import get_logger

logger = get_logger(__name__)


class EventTransformer(nn.Module):
    """
    Event-level transformer that aggregates DOM embeddings to predict direction.

    Uses standard self-attention across all DOMs in an event, with geometry-aware
    positional encoding based on DOM (x,y,z) positions.

    Args:
        d_model: Model dimension (should match T1 output)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 4)
        d_ff: Feedforward dimension (default: 4 * d_model)
        dropout: Dropout rate (default: 0.1)
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

        # Geometry encoder: (x,y,z) → d_model
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Transformer layers
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
        # Predicting 3D unit vector is more stable than angles
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),  # Outputs (x, y, z) before normalization
        )

        logger.info(
            f"Initialized EventTransformer: d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, d_ff={self.d_ff}, max_doms={max_doms}"
        )

    def _load_geometry(self, geometry_path: Optional[str] = None) -> torch.Tensor:
        """
        Load sensor geometry (x,y,z) for all DOMs.

        Args:
            geometry_path: Path to sensor_geometry.csv. If None, auto-detect from data config.

        Returns:
            geometry: (n_sensors, 3) tensor of (x,y,z) positions
        """
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
        dom_embeddings: torch.Tensor,
        dom_ids: torch.Tensor,
        dom_to_event_idx: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Forward pass: DOM embeddings → event predictions.

        Args:
            dom_embeddings: (total_doms, d_model) - DOM embeddings from T1
            dom_ids: (total_doms,) - sensor IDs for each DOM
            dom_to_event_idx: (total_doms,) - which event each DOM belongs to
            batch_size: Number of events in batch

        Returns:
            predictions: (batch_size, 2) - (azimuth, zenith) predictions
        """
        device = dom_embeddings.device
        total_doms = dom_embeddings.shape[0]

        # Get geometry for each DOM (move sensor_geometry to device if needed)
        if self.sensor_geometry.device != device:
            self.sensor_geometry = self.sensor_geometry.to(device)
        dom_geometry = self.sensor_geometry[dom_ids]  # (total_doms, 3)

        # Normalize geometry coordinates by 500 (from reference implementation)
        dom_geometry_normalized = dom_geometry / 500.0

        # Encode geometry as positional information
        geo_encoding = self.geometry_encoder(dom_geometry_normalized)  # (total_doms, d_model)

        # Add geometry encoding to DOM embeddings
        dom_features = dom_embeddings + geo_encoding  # (total_doms, d_model)

        # Pack DOMs into batched sequences (one event per sequence)
        # This requires padding to max_doms_in_batch
        event_sequences, padding_mask = self._pack_events(
            dom_features, dom_to_event_idx, batch_size
        )

        # Apply transformer
        # event_sequences: (batch_size, max_doms, d_model)
        # padding_mask: (batch_size, max_doms) - True for padding positions
        transformed = self.transformer(
            event_sequences,
            src_key_padding_mask=padding_mask
        )  # (batch_size, max_doms, d_model)

        # Global aggregation: mean pool over DOMs (excluding padding)
        # Expand padding mask to match d_model
        mask = ~padding_mask.unsqueeze(-1)  # (batch_size, max_doms, 1)
        masked_features = transformed * mask

        # Sum and normalize by number of valid DOMs
        event_embedding = masked_features.sum(dim=1)  # (batch_size, d_model)
        valid_doms = mask.sum(dim=1).clamp(min=1)  # (batch_size, 1)
        event_embedding = event_embedding / valid_doms  # Mean pooling

        # Predict direction as unit vector
        vector = self.prediction_head(event_embedding)  # (batch_size, 3)

        # Normalize to unit sphere
        norm = torch.sqrt(torch.sum(vector**2, dim=1, keepdim=True))
        unit_vector = vector / (norm + 1e-8)  # (batch_size, 3)

        return unit_vector  # Return unit vectors directly, not angles

    def _pack_events(
        self,
        dom_features: torch.Tensor,
        dom_to_event_idx: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack variable-length DOM sequences into batched tensor with padding.

        Args:
            dom_features: (total_doms, d_model)
            dom_to_event_idx: (total_doms,) - which event each DOM belongs to
            batch_size: Number of events

        Returns:
            packed_sequences: (batch_size, max_doms, d_model) - padded sequences
            padding_mask: (batch_size, max_doms) - True for padding positions
        """
        device = dom_features.device
        d_model = dom_features.shape[1]

        # Count DOMs per event
        doms_per_event = torch.zeros(batch_size, dtype=torch.long, device=device)
        for event_idx in range(batch_size):
            doms_per_event[event_idx] = (dom_to_event_idx == event_idx).sum()

        max_doms = doms_per_event.max().item()

        # Check if we need to warn about large events
        if max_doms > self.max_doms:
            logger.warning(
                f"Event has {max_doms} DOMs, exceeding max_doms={self.max_doms}. "
                f"Consider increasing max_doms or implementing DOM subsampling."
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


class EventAccumulator:
    """
    Accumulates DOM embeddings across multiple T1 batches to form complete events.

    Used when extreme events are split across multiple T1 batches. Collects all
    DOM embeddings for each event before passing to T2.

    Example usage:
        accumulator = EventAccumulator()

        # Process T1 batches
        for t1_batch in t1_dataloader:
            dom_embeddings, metadata = t1_model(t1_batch)
            accumulator.add_batch(dom_embeddings, metadata)

        # Get complete events for T2
        t2_batches = accumulator.get_complete_events(batch_size=32)
        for t2_batch in t2_batches:
            predictions = t2_model(**t2_batch)
    """

    def __init__(self):
        self.events = {}  # event_id -> {'dom_embeddings': list, 'dom_ids': list, 'target': tensor}

    def add_batch(
        self,
        dom_embeddings: torch.Tensor,
        metadata: Dict[str, torch.Tensor]
    ):
        """
        Add DOM embeddings from a T1 batch.

        Args:
            dom_embeddings: (total_doms, d_model) from T1
            metadata: Dict with:
                - dom_to_event_idx: (total_doms,) - which event each DOM belongs to (local to T1 batch)
                - sensor_ids: (total_doms,) - sensor IDs for each DOM
                - event_ids: (n_events,) - global event IDs
                - targets: (n_events, 2) - azimuth, zenith (optional)
        """
        dom_to_event_idx = metadata['dom_to_event_idx']  # (total_doms,)
        sensor_ids = metadata['sensor_ids']  # (total_doms,)
        event_ids = metadata['event_ids']  # (n_events,)
        targets = metadata.get('targets')  # (n_events, 2) or None

        # Group DOMs by event
        unique_event_indices = torch.unique(dom_to_event_idx)

        for local_event_idx in unique_event_indices:
            # Get global event ID
            event_id = event_ids[local_event_idx].item()

            # Get DOMs belonging to this event
            event_dom_mask = (dom_to_event_idx == local_event_idx)
            event_dom_embeddings = dom_embeddings[event_dom_mask]  # (n_doms_in_event, d_model)
            event_sensor_ids = sensor_ids[event_dom_mask]  # (n_doms_in_event,)

            # Initialize event if not seen before
            if event_id not in self.events:
                self.events[event_id] = {
                    'dom_embeddings': [],
                    'dom_ids': [],
                }
                # Add target if available
                if targets is not None:
                    self.events[event_id]['target'] = targets[local_event_idx]

            # Append DOMs to event
            self.events[event_id]['dom_embeddings'].append(event_dom_embeddings)
            self.events[event_id]['dom_ids'].append(event_sensor_ids)

    def get_complete_events(self, batch_size: int = 32):
        """
        Get batches of complete events for T2 processing.

        Yields batches of complete events (all DOMs collected).

        Args:
            batch_size: Number of events per T2 batch

        Yields:
            Dict with:
                - dom_embeddings: (total_doms_in_batch, d_model)
                - dom_ids: (total_doms_in_batch,)
                - dom_to_event_idx: (total_doms_in_batch,)
                - batch_size: int
                - targets: (batch_size, 2) if available
        """
        complete_event_ids = list(self.events.keys())

        for i in range(0, len(complete_event_ids), batch_size):
            batch_event_ids = complete_event_ids[i:i+batch_size]

            # Collect embeddings for this batch
            all_dom_embeddings = []
            all_dom_ids = []
            all_dom_to_event = []
            all_targets = []

            for local_event_idx, event_id in enumerate(batch_event_ids):
                event_data = self.events[event_id]

                # Concatenate all chunks for this event
                event_dom_embeddings = torch.cat(event_data['dom_embeddings'], dim=0)
                event_dom_ids = torch.cat(event_data['dom_ids'], dim=0)

                all_dom_embeddings.append(event_dom_embeddings)
                all_dom_ids.append(event_dom_ids)

                n_doms = event_dom_ids.shape[0]
                all_dom_to_event.extend([local_event_idx] * n_doms)

                if 'target' in event_data:
                    all_targets.append(event_data['target'])

            yield {
                'dom_embeddings': torch.cat(all_dom_embeddings, dim=0),
                'dom_ids': torch.cat(all_dom_ids, dim=0),
                'dom_to_event_idx': torch.tensor(all_dom_to_event),
                'batch_size': len(batch_event_ids),
                'targets': torch.stack(all_targets) if all_targets else None,
            }

    def clear(self):
        """Clear all accumulated events."""
        self.events.clear()

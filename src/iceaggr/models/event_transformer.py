"""
Event-level transformer (T2) for IceCube neutrino reconstruction.

This module implements the second stage of the hierarchical model:
- Input: DOM embeddings + geometry (x, y, z positions)
- Output: Neutrino direction prediction (azimuth, zenith)

The transformer aggregates information across all active DOMs in an event
to predict the neutrino direction.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from iceaggr.utils import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for geometry (x, y, z).

    Uses the standard transformer positional encoding but applied to
    3D spatial coordinates instead of sequence positions.
    """

    def __init__(self, d_model: int, max_freq: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq

        # Create frequency bands
        num_bands = d_model // 6  # 6 because we have 3 coords × (sin + cos)
        self.freqs = torch.logspace(0, math.log10(max_freq), num_bands)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D geometry coordinates.

        Args:
            geometry: (batch_size, 3) or (num_doms, 3) coordinates [x, y, z]

        Returns:
            pos_enc: (batch_size, d_model) or (num_doms, d_model) position encodings
        """
        device = geometry.device
        freqs = self.freqs.to(device)

        # geometry: (N, 3)
        x, y, z = geometry[:, 0:1], geometry[:, 1:2], geometry[:, 2:3]  # (N, 1) each

        # Compute sin/cos for each frequency
        encodings = []
        for coord in [x, y, z]:
            for freq in freqs:
                encodings.append(torch.sin(2 * math.pi * freq * coord))
                encodings.append(torch.cos(2 * math.pi * freq * coord))

        # Concatenate and pad if needed
        pos_enc = torch.cat(encodings, dim=1)  # (N, num_bands * 6)

        # Pad to d_model if needed
        if pos_enc.shape[1] < self.d_model:
            padding = torch.zeros(pos_enc.shape[0], self.d_model - pos_enc.shape[1], device=device)
            pos_enc = torch.cat([pos_enc, padding], dim=1)
        elif pos_enc.shape[1] > self.d_model:
            pos_enc = pos_enc[:, :self.d_model]

        return pos_enc


class EventTransformer(nn.Module):
    """
    Event-level transformer (T2) for neutrino direction prediction.

    Takes DOM embeddings and geometry as input, aggregates across DOMs,
    and predicts neutrino direction (azimuth, zenith).

    Args:
        d_input: Input DOM embedding dimension
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension (default: 4 * d_model)
        dropout: Dropout probability
        use_geometry: Whether to add geometry positional encoding
    """

    def __init__(
        self,
        d_input: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_geometry: bool = True
    ):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_geometry = use_geometry

        if d_ff is None:
            d_ff = 4 * d_model

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Geometry encoding
        if use_geometry:
            self.geometry_encoder = PositionalEncoding(d_model)
            self.geometry_proj = nn.Linear(d_model, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head for direction prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # azimuth, zenith
        )

        # Learnable CLS token for global pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        logger.info(f"EventTransformer: {d_input} → {d_model}")
        logger.info(f"  Architecture: {n_layers} layers, {n_heads} heads, {d_ff} ff_dim")
        logger.info(f"  Geometry encoding: {use_geometry}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        dom_embeddings: torch.Tensor,
        dom_to_event_idx: torch.Tensor,
        batch_size: int,
        geometry: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of event transformer.

        Args:
            dom_embeddings: (num_doms, d_input) DOM-level embeddings
            dom_to_event_idx: (num_doms,) which event each DOM belongs to
            batch_size: Number of events in batch
            geometry: (num_doms, 3) DOM positions [x, y, z] (optional)

        Returns:
            predictions: (batch_size, 2) neutrino directions [azimuth, zenith]
        """
        num_doms = dom_embeddings.shape[0]

        # Project DOM embeddings to model dimension
        x = self.input_proj(dom_embeddings)  # (num_doms, d_model)

        # Add geometry encoding if available
        if self.use_geometry and geometry is not None:
            geo_enc = self.geometry_encoder(geometry)  # (num_doms, d_model)
            geo_proj = self.geometry_proj(geo_enc)  # (num_doms, d_model)
            x = x + geo_proj

        # Pack into batch format
        # We need to create padded sequences per event
        # Count DOMs per event
        dom_counts = torch.bincount(dom_to_event_idx, minlength=batch_size)
        max_doms = dom_counts.max().item()

        # Create padded batch tensor and attention mask
        batch_tensor = torch.zeros(batch_size, max_doms + 1, self.d_model, device=x.device)  # +1 for CLS token
        attention_mask = torch.ones(batch_size, max_doms + 1, dtype=torch.bool, device=x.device)

        # Add CLS token
        batch_tensor[:, 0, :] = self.cls_token

        # Fill in DOM embeddings
        current_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        for i in range(num_doms):
            event_id = dom_to_event_idx[i].item()
            pos = current_idx[event_id].item() + 1  # +1 to skip CLS token
            batch_tensor[event_id, pos, :] = x[i]
            attention_mask[event_id, pos] = False  # False = not masked (attend to this)
            current_idx[event_id] += 1

        # Apply transformer
        # PyTorch attention mask: True = ignore, False = attend
        transformed = self.transformer(batch_tensor, src_key_padding_mask=attention_mask)  # (batch_size, max_doms+1, d_model)

        # Extract CLS token (global representation)
        cls_output = transformed[:, 0, :]  # (batch_size, d_model)

        # Predict direction
        predictions = self.output_head(cls_output)  # (batch_size, 2)

        return predictions

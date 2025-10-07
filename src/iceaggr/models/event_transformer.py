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

    Takes DOM embeddings (which already contain geometry information from T1),
    aggregates across DOMs, and predicts neutrino direction (azimuth, zenith).

    Args:
        d_input: Input DOM embedding dimension
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension (default: 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_input: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_geometry: bool = True  # Kept for backward compatibility, but ignored
    ):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        if d_ff is None:
            d_ff = 4 * d_model

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

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

        # Output head for direction prediction (unit vectors)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)  # x, y, z unit vector
        )

        # Learnable CLS token for global pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        logger.info(f"EventTransformer: {d_input} → {d_model}")
        logger.info(f"  Architecture: {n_layers} layers, {n_heads} heads, {d_ff} ff_dim")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        dom_embeddings: torch.Tensor,
        dom_to_event_idx: torch.Tensor,
        batch_size: int,
        geometry: Optional[torch.Tensor] = None  # Kept for backward compatibility but ignored
    ) -> torch.Tensor:
        """
        Forward pass of event transformer.

        Args:
            dom_embeddings: (num_doms, d_input) DOM-level embeddings
            dom_to_event_idx: (num_doms,) which event each DOM belongs to
            batch_size: Number of events in batch
            geometry: (num_doms, 3) DOM positions [x, y, z] (IGNORED - already in embeddings)

        Returns:
            predictions: (batch_size, 3) neutrino direction unit vectors [x, y, z]
        """
        num_doms = dom_embeddings.shape[0]

        # Project DOM embeddings to model dimension
        x = self.input_proj(dom_embeddings)  # (num_doms, d_model)

        # Pack into batch format
        # Count DOMs per event
        dom_counts = torch.bincount(dom_to_event_idx, minlength=batch_size)
        max_doms = dom_counts.max().item()

        # Create padded batch tensor and attention mask
        batch_tensor = torch.zeros(batch_size, max_doms + 1, self.d_model, device=x.device)  # +1 for CLS token
        attention_mask = torch.ones(batch_size, max_doms + 1, dtype=torch.bool, device=x.device)

        # Add CLS token to all events
        batch_tensor[:, 0, :] = self.cls_token

        # Vectorized packing: compute positions for each DOM
        # For each DOM, compute its position within its event (0, 1, 2, ...)
        # We'll use scatter_add to count how many DOMs we've seen for each event so far

        # Create a running counter for each event
        position_counter = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        positions = torch.zeros(num_doms, dtype=torch.long, device=x.device)

        # Compute position for each DOM sequentially (this is still a small loop over num_doms, not num_pulses)
        for i in range(num_doms):
            event_id = dom_to_event_idx[i]
            positions[i] = position_counter[event_id]
            position_counter[event_id] += 1

        # Pack embeddings into batch tensor (offset by 1 for CLS token)
        batch_indices = dom_to_event_idx
        position_indices = positions + 1  # +1 to skip CLS token

        # Use advanced indexing to pack all DOMs at once
        batch_tensor[batch_indices, position_indices] = x
        attention_mask[batch_indices, position_indices] = False  # False = not masked (attend to this)

        # Apply transformer
        # PyTorch attention mask: True = ignore, False = attend
        transformed = self.transformer(batch_tensor, src_key_padding_mask=attention_mask)  # (batch_size, max_doms+1, d_model)

        # Extract CLS token (global representation)
        cls_output = transformed[:, 0, :]  # (batch_size, d_model)

        # Predict direction as unit vector
        predictions = self.output_head(cls_output)  # (batch_size, 3)

        # Normalize to unit vectors (L2 norm)
        norm = torch.sqrt(torch.sum(predictions**2, dim=1, keepdim=True))
        predictions = predictions / norm

        return predictions

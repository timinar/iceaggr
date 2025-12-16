"""
Pulse embedder module for IceCube hierarchical model.

This module embeds raw pulse features into a learned embedding space
using a multi-layer perceptron (MLP).
"""

import torch
import torch.nn as nn
from typing import List, Optional


class PulseEmbedder(nn.Module):
    """
    Simple MLP to embed raw pulse features into embedding space.

    Input features (8 dimensions):
        - time: normalized timestamp
        - charge: photoelectron count (log-transformed internally)
        - auxiliary: 0/1 flag for auxiliary pulses
        - x, y, z: DOM position (from geometry, already normalized)
        - n_pulses_in_dom: number of pulses in the DOM (log-transformed internally)
        - pulse_idx_in_dom: index of pulse within DOM (normalized internally)

    Args:
        input_dim: Number of input features (default: 8)
        embed_dim: Output embedding dimension (default: 64)
        hidden_dim: Hidden layer dimension (default: 64)

    Example:
        >>> embedder = PulseEmbedder(input_dim=8, embed_dim=64)
        >>> pulse_features = torch.randn(1000, 8)  # 1000 pulses
        >>> embeddings = embedder(pulse_features)  # (1000, 64)
    """

    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,  # kept for backward compat, ignored
        dropout: float = 0.1,  # kept for backward compat, ignored
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Simple MLP: input -> hidden -> output (no LayerNorm, no Dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, pulse_features: torch.Tensor) -> torch.Tensor:
        """
        Embed pulse features.

        Args:
            pulse_features: Raw pulse features (total_pulses, input_dim)

        Returns:
            Pulse embeddings (total_pulses, embed_dim)
        """
        return self.mlp(pulse_features)


def build_pulse_features(
    raw_features: torch.Tensor,
    dom_positions: torch.Tensor,
    pulse_to_dom_idx: torch.Tensor,
    n_pulses_in_dom: torch.Tensor,
    pulse_idx_in_dom: torch.Tensor,
) -> torch.Tensor:
    """
    Build full pulse feature tensor from batch components.

    This combines raw pulse features with geometry and DOM metadata
    to create the 8-dimensional input for PulseEmbedder.

    Args:
        raw_features: (total_pulses, 4) - [time, charge, sensor_id, auxiliary]
        dom_positions: (total_doms, 3) - DOM x, y, z positions
        pulse_to_dom_idx: (total_pulses,) - DOM index for each pulse
        n_pulses_in_dom: (total_pulses,) - Number of pulses in each pulse's DOM
        pulse_idx_in_dom: (total_pulses,) - Index of each pulse within its DOM

    Returns:
        Full feature tensor (total_pulses, 8)
    """
    # Get DOM positions for each pulse
    pulse_positions = dom_positions[pulse_to_dom_idx]  # (total_pulses, 3)

    # Extract raw features
    time_raw = raw_features[:, 0:1]
    charge_raw = raw_features[:, 1:2]
    auxiliary = raw_features[:, 3:4]  # Skip sensor_id (column 2)

    # Normalize using established IceCube conventions
    # time: (time - 1e4) / 3e4  (centered around typical event time)
    time_norm = (time_raw - 1e4) / 3e4

    # charge: log10(charge) / 3.0  (log10, not ln!)
    charge_norm = torch.log10(charge_raw.clamp(min=1e-6)) / 3.0

    # aux: center around 0
    aux_norm = auxiliary - 0.5

    # Preprocess DOM metadata
    n_pulses_log = torch.log1p(n_pulses_in_dom.float()).unsqueeze(1)
    # Normalize: typical range is log(1) to log(500) ≈ 0 to 6
    n_pulses_norm = n_pulses_log / 3.0 - 1.0  # Roughly center around 0

    # Normalize pulse index by DOM size (avoid division by zero)
    # This is already in [0, 1) range, shift to center at 0
    pulse_idx_norm = (pulse_idx_in_dom.float() / n_pulses_in_dom.float().clamp(min=1)).unsqueeze(1) - 0.5

    # Concatenate all features: [time, charge, aux, x, y, z, n_pulses, pulse_idx]
    features = torch.cat([
        time_norm,               # (N, 1) - normalized time
        charge_norm,             # (N, 1) - normalized log charge
        aux_norm,                # (N, 1) - centered around 0
        pulse_positions,         # (N, 3) - already ~[-1, 1]
        n_pulses_norm,           # (N, 1) - normalized
        pulse_idx_norm,          # (N, 1) - centered around 0
    ], dim=1)

    return features

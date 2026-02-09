"""
Directional prediction head for neutrino direction.

This module predicts the neutrino direction as a 3D unit vector
from the event embedding.
"""

import torch
import torch.nn as nn


class DirectionalHead(nn.Module):
    """
    Prediction head for neutrino direction as unit vector.

    Takes an event embedding and predicts a 3D direction vector,
    normalized to unit length for the angular loss function.

    Args:
        embed_dim: Input embedding dimension (default: 64)
        hidden_dim: Hidden layer dimension (default: 128)

    Example:
        >>> head = DirectionalHead(embed_dim=64, hidden_dim=128)
        >>> event_embed = torch.randn(32, 64)
        >>> direction = head(event_embed)
        >>> direction.shape  # (32, 3)
        >>> torch.allclose(direction.norm(dim=1), torch.ones(32))  # True (unit vectors)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()  # GELU instead of ReLU to avoid dead neurons
        self.fc2 = nn.Linear(hidden_dim, 3)

        # Xavier normal gives std ≈ 0.124 for Linear(128→3), producing diverse
        # initial predictions. (std=0.01 caused mode collapse — all predictions
        # collapsed to the same direction.)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, event_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict neutrino direction.

        Args:
            event_embedding: (batch_size, embed_dim)

        Returns:
            Direction unit vectors (batch_size, 3)
        """
        x = self.fc1(event_embedding)
        x = self.activation(x)
        x = self.fc2(x)

        # Normalize to unit vector
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        x = x / (norm + 1e-8)

        return x

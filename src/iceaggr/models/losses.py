"""
Loss functions for IceCube direction prediction.

Functions from PolarBERT for consistency.
"""

import torch
import torch.nn as nn


# From PolarBERT loss_functions.py
def angles_to_unit_vector(azimuth: torch.Tensor, zenith: torch.Tensor) -> torch.Tensor:
    """Convert spherical angles to 3D unit vector."""
    return torch.stack([
        torch.cos(azimuth) * torch.sin(zenith),
        torch.sin(azimuth) * torch.sin(zenith),
        torch.cos(zenith)
    ], dim=1)


def unit_vector_to_angles(n: torch.Tensor) -> torch.Tensor:
    """Convert 3D unit vectors to spherical angles."""
    norm = torch.sqrt(torch.sum(n**2, dim=1))
    x = n[:, 0]
    y = n[:, 1]
    z = n[:, 2]
    azimuth = torch.atan2(y, x)
    zenith = torch.arccos(z / norm)
    return torch.stack([azimuth, zenith], dim=1)


def angular_dist_score_unit_vectors(
    n_true: torch.Tensor,
    n_pred: torch.Tensor,
    epsilon: float = 0,
) -> torch.Tensor:
    """Compute angular distance between unit vectors (from PolarBERT)."""
    assert torch.all(torch.isfinite(n_true))
    assert torch.all(torch.isfinite(n_pred))
    scalar_prod = torch.sum(n_true * n_pred, dim=1)
    scalar_prod = torch.clip(scalar_prod, -1 + epsilon, 1 - epsilon)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))


def angular_distance_loss(
    y_pred: torch.Tensor,
    y_true_angles: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    Compute mean angular distance between predicted and true directions.

    Args:
        y_pred: Predicted unit vectors (batch_size, 3)
        y_true_angles: True angles [azimuth, zenith] in radians (batch_size, 2)
        epsilon: Small value for numerical stability

    Returns:
        Scalar loss (mean angular distance in radians)
    """
    # Convert true angles to unit vector (like PolarBERT does)
    y_true = angles_to_unit_vector(y_true_angles[:, 0], y_true_angles[:, 1])

    # Use PolarBERT's loss function
    return angular_dist_score_unit_vectors(y_true, y_pred, epsilon=epsilon)


class AngularDistanceLoss(nn.Module):
    """
    Angular distance loss as a PyTorch module.

    This computes the mean angular distance between predicted unit vectors
    and target directions given as (azimuth, zenith) angles.
    """

    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true_angles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute angular distance loss.

        Args:
            y_pred: Predicted unit vectors (batch_size, 3)
            y_true_angles: True angles [azimuth, zenith] (batch_size, 2)

        Returns:
            Scalar loss in radians
        """
        return angular_distance_loss(y_pred, y_true_angles, self.epsilon)

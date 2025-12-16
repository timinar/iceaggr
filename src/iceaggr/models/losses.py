"""
Loss functions for IceCube direction prediction.

This module provides angular distance loss functions for training
directional prediction models using unit vector representations.
"""

import torch
import torch.nn as nn


def angles_to_unit_vector(azimuth: torch.Tensor, zenith: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical angles to 3D unit vector.

    Args:
        azimuth: Azimuth angle in radians (batch_size,)
        zenith: Zenith angle in radians (batch_size,)

    Returns:
        Unit vectors of shape (batch_size, 3)
    """
    return torch.stack([
        torch.cos(azimuth) * torch.sin(zenith),
        torch.sin(azimuth) * torch.sin(zenith),
        torch.cos(zenith)
    ], dim=1)


def unit_vector_to_angles(unit_vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D unit vectors to spherical angles.

    Args:
        unit_vectors: Unit vectors of shape (batch_size, 3)

    Returns:
        Angles of shape (batch_size, 2) as [azimuth, zenith]
    """
    x, y, z = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]

    azimuth = torch.atan2(y, x)
    zenith = torch.acos(torch.clamp(z, -1.0, 1.0))

    return torch.stack([azimuth, zenith], dim=1)


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
    # Convert true angles to unit vector
    y_true = angles_to_unit_vector(y_true_angles[:, 0], y_true_angles[:, 1])

    # Dot product
    dot_product = torch.sum(y_pred * y_true, dim=1)

    # Clip for numerical stability
    dot_product = torch.clamp(dot_product, -1 + epsilon, 1 - epsilon)

    # Angular distance
    angular_dist = torch.acos(dot_product)

    return angular_dist.mean()


def angular_distance_from_unit_vectors(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    Compute angular distance between two sets of unit vectors.

    Args:
        y_pred: Predicted unit vectors (batch_size, 3)
        y_true: True unit vectors (batch_size, 3)
        epsilon: Small value for numerical stability

    Returns:
        Scalar loss (mean angular distance in radians)
    """
    # Dot product
    dot_product = torch.sum(y_pred * y_true, dim=1)

    # Clip for numerical stability
    dot_product = torch.clamp(dot_product, -1 + epsilon, 1 - epsilon)

    # Angular distance
    angular_dist = torch.acos(dot_product)

    return angular_dist.mean()


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

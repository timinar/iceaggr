"""
Loss functions for angular reconstruction.

Based on the Kaggle IceCube competition metric.
"""

import torch
import torch.nn as nn


def angular_dist_score(
    az_true: torch.Tensor,
    zen_true: torch.Tensor,
    az_pred: torch.Tensor,
    zen_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate mean angular distance between predicted and true directions.

    Args:
        az_true: True azimuth angles (radians), shape (N,)
        zen_true: True zenith angles (radians), shape (N,)
        az_pred: Predicted azimuth angles (radians), shape (N,)
        zen_pred: Predicted zenith angles (radians), shape (N,)

    Returns:
        Mean angular distance in radians, shape ()
    """
    assert torch.all(torch.isfinite(az_pred)), "Non-finite values in azimuth predictions"
    assert torch.all(torch.isfinite(zen_pred)), "Non-finite values in zenith predictions"

    # Pre-compute sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)

    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    # Scalar product of cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # Clip to [-1, 1] for numerical stability
    scalar_prod = torch.clip(scalar_prod, -1, 1)

    # Convert to angle (radians)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))


def angles_to_unit_vector(azimuth: torch.Tensor, zenith: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical angles to unit vectors.

    Args:
        azimuth: Azimuth angles (radians), shape (N,)
        zenith: Zenith angles (radians), shape (N,)

    Returns:
        Unit vectors, shape (N, 3)
    """
    return torch.stack(
        [
            torch.cos(azimuth) * torch.sin(zenith),
            torch.sin(azimuth) * torch.sin(zenith),
            torch.cos(zenith),
        ],
        dim=1,
    )


def unit_vector_to_angles(
    n: torch.Tensor, check_unit_norm: bool = False
) -> torch.Tensor:
    """
    Convert unit vectors to spherical angles.

    Args:
        n: Unit vectors, shape (N, 3)
        check_unit_norm: Whether to assert unit norm

    Returns:
        Angles (azimuth, zenith) in radians, shape (N, 2)
    """
    norm = torch.sqrt(torch.sum(n**2, dim=1))
    if check_unit_norm:
        assert torch.allclose(
            norm, torch.ones_like(norm)
        ), "Input vectors are not unit vectors"

    x = n[:, 0]
    y = n[:, 1]
    z = n[:, 2]
    azimuth = torch.atan2(y, x)
    zenith = torch.arccos(z / norm)
    return torch.stack([azimuth, zenith], dim=1)


def angular_dist_score_unit_vectors(
    n_true: torch.Tensor, n_pred: torch.Tensor, epsilon: float = 0.0
) -> torch.Tensor:
    """
    Calculate angular distance using unit vectors directly.

    Args:
        n_true: True direction unit vectors, shape (N, 3)
        n_pred: Predicted direction unit vectors, shape (N, 3)
        epsilon: Small value for numerical stability

    Returns:
        Mean angular distance in radians, shape ()
    """
    assert torch.all(torch.isfinite(n_true)), "Non-finite values in true vectors"
    assert torch.all(torch.isfinite(n_pred)), "Non-finite values in predicted vectors"

    scalar_prod = torch.sum(n_true * n_pred, dim=1)
    scalar_prod = torch.clip(scalar_prod, -1 + epsilon, 1 - epsilon)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))


class AngularDistanceLoss(nn.Module):
    """Loss module for angular reconstruction."""

    def __init__(self, use_unit_vectors: bool = False, epsilon: float = 1e-7):
        """
        Args:
            use_unit_vectors: If True, expects unit vectors instead of angles
            epsilon: Numerical stability epsilon for unit vector mode
        """
        super().__init__()
        self.use_unit_vectors = use_unit_vectors
        self.epsilon = epsilon

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate angular distance loss.

        Args:
            pred: Predictions, shape (N, 2) for angles or (N, 3) for unit vectors
            target: Targets, shape (N, 2) for angles or (N, 3) for unit vectors

        Returns:
            Mean angular distance in radians
        """
        if self.use_unit_vectors:
            return angular_dist_score_unit_vectors(target, pred, self.epsilon)
        else:
            az_pred, zen_pred = pred[:, 0], pred[:, 1]
            az_true, zen_true = target[:, 0], target[:, 1]
            return angular_dist_score(az_true, zen_true, az_pred, zen_pred)

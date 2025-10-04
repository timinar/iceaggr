"""
Unit tests for loss functions.
"""

import numpy as np
import pytest
import torch

from iceaggr.training.losses import (
    AngularDistanceLoss,
    angular_dist_score,
    angular_dist_score_unit_vectors,
    angles_to_unit_vector,
    unit_vector_to_angles,
)


class TestAngleConversions:
    """Test angle <-> unit vector conversions."""

    def test_angles_to_unit_vector(self):
        """Test conversion from angles to unit vectors."""
        # North pole (zenith=0)
        az = torch.tensor([0.0])
        zen = torch.tensor([0.0])
        vec = angles_to_unit_vector(az, zen)
        expected = torch.tensor([[0.0, 0.0, 1.0]])
        assert torch.allclose(vec, expected, atol=1e-6)

        # Equator, east (azimuth=0, zenith=π/2)
        az = torch.tensor([0.0])
        zen = torch.tensor([np.pi / 2])
        vec = angles_to_unit_vector(az, zen)
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(vec, expected, atol=1e-6)

        # Equator, north (azimuth=π/2, zenith=π/2)
        az = torch.tensor([np.pi / 2])
        zen = torch.tensor([np.pi / 2])
        vec = angles_to_unit_vector(az, zen)
        expected = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(vec, expected, atol=1e-6)

    def test_unit_vector_to_angles(self):
        """Test conversion from unit vectors to angles."""
        # North pole
        vec = torch.tensor([[0.0, 0.0, 1.0]])
        angles = unit_vector_to_angles(vec)
        # Azimuth can be anything when zenith=0, just check zenith
        assert torch.allclose(angles[:, 1], torch.tensor([0.0]), atol=1e-6)

        # Equator, east
        vec = torch.tensor([[1.0, 0.0, 0.0]])
        angles = unit_vector_to_angles(vec)
        expected = torch.tensor([[0.0, np.pi / 2]])
        assert torch.allclose(angles, expected, atol=1e-6)

        # Equator, north
        vec = torch.tensor([[0.0, 1.0, 0.0]])
        angles = unit_vector_to_angles(vec)
        expected = torch.tensor([[np.pi / 2, np.pi / 2]])
        assert torch.allclose(angles, expected, atol=1e-6)

    def test_round_trip_conversion(self):
        """Test that angle -> vector -> angle preserves values."""
        az = torch.tensor([0.5, 1.0, 1.5, 2.0])
        zen = torch.tensor([0.3, 0.8, 1.2, 1.5])

        vec = angles_to_unit_vector(az, zen)
        az_rec, zen_rec = unit_vector_to_angles(vec).T

        assert torch.allclose(az, az_rec, atol=1e-6)
        assert torch.allclose(zen, zen_rec, atol=1e-6)

    def test_unit_norm_check(self):
        """Test unit norm assertion."""
        # Non-unit vector
        vec = torch.tensor([[2.0, 0.0, 0.0]])
        with pytest.raises(AssertionError):
            unit_vector_to_angles(vec, check_unit_norm=True)

        # Unit vector should pass
        vec = torch.tensor([[1.0, 0.0, 0.0]])
        angles = unit_vector_to_angles(vec, check_unit_norm=True)
        assert angles.shape == (1, 2)


class TestAngularDistance:
    """Test angular distance calculations."""

    def test_zero_distance(self):
        """Test that identical angles have very small distance (numerical precision)."""
        az = torch.tensor([1.0, 2.0, 3.0])
        zen = torch.tensor([0.5, 1.0, 1.5])

        dist = angular_dist_score(az, zen, az, zen)
        # Should be very close to zero (within numerical precision)
        assert dist < 1e-3, f"Distance should be near zero, got {dist}"

    def test_opposite_directions(self):
        """Test distance for opposite directions (should be π)."""
        # North pole vs south pole
        az_true = torch.tensor([0.0])
        zen_true = torch.tensor([0.0])
        az_pred = torch.tensor([0.0])
        zen_pred = torch.tensor([np.pi])

        dist = angular_dist_score(az_true, zen_true, az_pred, zen_pred)
        assert torch.allclose(dist, torch.tensor(np.pi), atol=1e-6)

    def test_perpendicular_directions(self):
        """Test distance for perpendicular directions (should be π/2)."""
        # North pole (0, 0, 1) vs equator (1, 0, 0)
        az_true = torch.tensor([0.0])
        zen_true = torch.tensor([0.0])
        az_pred = torch.tensor([0.0])
        zen_pred = torch.tensor([np.pi / 2])

        dist = angular_dist_score(az_true, zen_true, az_pred, zen_pred)
        assert torch.allclose(dist, torch.tensor(np.pi / 2), atol=1e-6)

    def test_batch_distance(self):
        """Test distance calculation for batches."""
        batch_size = 10
        az_true = torch.rand(batch_size) * 2 * np.pi
        zen_true = torch.rand(batch_size) * np.pi
        az_pred = az_true + torch.randn(batch_size) * 0.1
        zen_pred = zen_true + torch.randn(batch_size) * 0.1

        dist = angular_dist_score(az_true, zen_true, az_pred, zen_pred)
        assert dist >= 0.0
        assert dist < np.pi  # Should be small for similar angles

    def test_numerical_stability(self):
        """Test that clipping prevents numerical issues."""
        # Very similar angles (potential for numerical instability)
        az_true = torch.tensor([1.0] * 100)
        zen_true = torch.tensor([0.5] * 100)
        az_pred = az_true + 1e-8
        zen_pred = zen_true + 1e-8

        dist = angular_dist_score(az_true, zen_true, az_pred, zen_pred)
        assert torch.isfinite(dist)


class TestUnitVectorDistance:
    """Test angular distance with unit vectors."""

    def test_zero_distance_vectors(self):
        """Test that identical vectors have zero distance."""
        vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        dist = angular_dist_score_unit_vectors(vec, vec)
        assert torch.allclose(dist, torch.tensor(0.0), atol=1e-6)

    def test_opposite_vectors(self):
        """Test distance for opposite vectors (should be π)."""
        vec_true = torch.tensor([[1.0, 0.0, 0.0]])
        vec_pred = torch.tensor([[-1.0, 0.0, 0.0]])

        dist = angular_dist_score_unit_vectors(vec_true, vec_pred)
        assert torch.allclose(dist, torch.tensor(np.pi), atol=1e-6)

    def test_consistency_with_angle_version(self):
        """Test that unit vector version matches angle version."""
        az_true = torch.tensor([0.5, 1.0, 1.5])
        zen_true = torch.tensor([0.3, 0.8, 1.2])
        az_pred = torch.tensor([0.6, 1.1, 1.4])
        zen_pred = torch.tensor([0.4, 0.7, 1.3])

        # Angle version
        dist_angles = angular_dist_score(az_true, zen_true, az_pred, zen_pred)

        # Vector version
        vec_true = angles_to_unit_vector(az_true, zen_true)
        vec_pred = angles_to_unit_vector(az_pred, zen_pred)
        dist_vectors = angular_dist_score_unit_vectors(vec_true, vec_pred)

        assert torch.allclose(dist_angles, dist_vectors, atol=1e-5)


class TestAngularDistanceLoss:
    """Test AngularDistanceLoss module."""

    def test_angle_mode(self):
        """Test loss module in angle mode."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)

        pred = torch.tensor([[0.5, 1.0], [1.0, 0.5]])
        target = torch.tensor([[0.6, 1.1], [0.9, 0.6]])

        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)
        assert loss >= 0.0

    def test_unit_vector_mode(self):
        """Test loss module in unit vector mode."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=True)

        pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
        # Normalize to unit vectors
        pred = pred / pred.norm(dim=1, keepdim=True)
        target = target / target.norm(dim=1, keepdim=True)

        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)
        assert loss >= 0.0

    def test_zero_loss_for_identical(self):
        """Test that identical predictions give zero loss."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)

        angles = torch.tensor([[0.5, 1.0], [1.0, 0.5]])
        loss = loss_fn(angles, angles)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_gradients_flow(self):
        """Test that gradients flow through loss."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)

        pred = torch.tensor([[0.5, 1.0]], requires_grad=True)
        target = torch.tensor([[0.6, 1.1]])

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

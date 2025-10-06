"""Unit tests for loss functions."""

import pytest
import torch
import math
from iceaggr.training.losses import (
    angular_dist_score,
    angular_dist_score_unit_vectors,
    angles_to_unit_vector,
    unit_vector_to_angles,
    AngularDistanceLoss,
)


class TestAngularDistanceScore:
    """Test angular distance computation."""

    def test_identical_directions(self):
        """Test that identical directions give near-zero distance."""
        az = torch.tensor([0.0, 1.0, 2.0])
        zen = torch.tensor([1.0, 1.5, 2.0])

        dist = angular_dist_score(az, zen, az, zen)
        # Near zero (abs() causes small numerical error)
        assert dist < 1e-3

    def test_opposite_directions(self):
        """Test that opposite directions give π distance."""
        # North pole vs South pole (zenith=0 vs zenith=π)
        az_true = torch.tensor([0.0])
        zen_true = torch.tensor([0.0])
        az_pred = torch.tensor([0.0])
        zen_pred = torch.tensor([math.pi])

        dist = angular_dist_score(az_true, zen_true, az_pred, zen_pred)
        assert torch.isclose(dist, torch.tensor(math.pi), atol=1e-5)

    def test_perpendicular_directions(self):
        """Test perpendicular directions give π/2 distance."""
        # Pointing up (zen=0) vs pointing horizontally (zen=π/2)
        az_true = torch.tensor([0.0])
        zen_true = torch.tensor([0.0])
        az_pred = torch.tensor([0.0])
        zen_pred = torch.tensor([math.pi / 2])

        dist = angular_dist_score(az_true, zen_true, az_pred, zen_pred)
        assert torch.isclose(dist, torch.tensor(math.pi / 2), atol=1e-5)

    def test_batch_computation(self):
        """Test that batch computation works."""
        batch_size = 100
        az_true = torch.rand(batch_size) * 2 * math.pi
        zen_true = torch.rand(batch_size) * math.pi
        az_pred = torch.rand(batch_size) * 2 * math.pi
        zen_pred = torch.rand(batch_size) * math.pi

        dist = angular_dist_score(az_true, zen_true, az_pred, zen_pred)

        # Should be a scalar
        assert dist.shape == ()
        # Should be in valid range [0, π]
        assert 0 <= dist <= math.pi


class TestAngleUnitVectorConversion:
    """Test conversion between angles and unit vectors."""

    def test_angles_to_unit_vector_north_pole(self):
        """Test conversion for north pole (zenith=0)."""
        az = torch.tensor([0.0])
        zen = torch.tensor([0.0])

        vec = angles_to_unit_vector(az, zen)
        expected = torch.tensor([[0.0, 0.0, 1.0]])

        assert torch.allclose(vec, expected, atol=1e-6)

    def test_angles_to_unit_vector_equator(self):
        """Test conversion for equator (zenith=π/2)."""
        # Point along x-axis
        az = torch.tensor([0.0])
        zen = torch.tensor([math.pi / 2])

        vec = angles_to_unit_vector(az, zen)
        expected = torch.tensor([[1.0, 0.0, 0.0]])

        assert torch.allclose(vec, expected, atol=1e-6)

    def test_roundtrip_conversion(self):
        """Test that angle → vector → angle is identity."""
        az = torch.tensor([0.5, 1.0, 2.0, 3.0])
        zen = torch.tensor([0.5, 1.0, 1.5, 2.0])

        vec = angles_to_unit_vector(az, zen)
        az_back, zen_back = unit_vector_to_angles(vec).T

        assert torch.allclose(az, az_back, atol=1e-5)
        assert torch.allclose(zen, zen_back, atol=1e-5)

    def test_unit_vectors_have_unit_norm(self):
        """Test that converted vectors have unit norm."""
        az = torch.rand(100) * 2 * math.pi
        zen = torch.rand(100) * math.pi

        vec = angles_to_unit_vector(az, zen)
        norms = torch.sqrt(torch.sum(vec ** 2, dim=1))

        assert torch.allclose(norms, torch.ones(100), atol=1e-5)


class TestAngularDistanceLoss:
    """Test AngularDistanceLoss module."""

    def test_loss_module_angles(self):
        """Test loss module with angles."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)

        pred = torch.tensor([[0.0, 1.0], [1.0, 1.5]])
        target = torch.tensor([[0.1, 1.1], [1.1, 1.6]])

        loss = loss_fn(pred, target)

        # Should be a scalar
        assert loss.shape == ()
        # Should be positive
        assert loss > 0

    def test_loss_module_unit_vectors(self):
        """Test loss module with unit vectors."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=True)

        # Create some random unit vectors
        vec1 = torch.randn(10, 3)
        vec1 = vec1 / torch.norm(vec1, dim=1, keepdim=True)

        vec2 = torch.randn(10, 3)
        vec2 = vec2 / torch.norm(vec2, dim=1, keepdim=True)

        loss = loss_fn(vec1, vec2)

        # Should be a scalar
        assert loss.shape == ()
        # Should be in valid range
        assert 0 <= loss <= math.pi

    def test_loss_gradients(self):
        """Test that loss produces gradients."""
        loss_fn = AngularDistanceLoss(use_unit_vectors=False)

        pred = torch.tensor([[0.0, 1.0], [1.0, 1.5]], requires_grad=True)
        target = torch.tensor([[0.1, 1.1], [1.1, 1.6]])

        loss = loss_fn(pred, target)
        loss.backward()

        # Should have gradients
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)


class TestAngularDistanceEquivalence:
    """Test equivalence between angle and unit vector formulations."""

    def test_angle_vs_unit_vector_equivalence(self):
        """Test that both formulations give same result."""
        # Generate random angles
        az_true = torch.rand(50) * 2 * math.pi
        zen_true = torch.rand(50) * math.pi
        az_pred = torch.rand(50) * 2 * math.pi
        zen_pred = torch.rand(50) * math.pi

        # Compute using angles
        dist_angles = angular_dist_score(az_true, zen_true, az_pred, zen_pred)

        # Convert to unit vectors and compute
        n_true = angles_to_unit_vector(az_true, zen_true)
        n_pred = angles_to_unit_vector(az_pred, zen_pred)
        dist_vectors = angular_dist_score_unit_vectors(n_true, n_pred)

        # Should be very close
        assert torch.isclose(dist_angles, dist_vectors, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

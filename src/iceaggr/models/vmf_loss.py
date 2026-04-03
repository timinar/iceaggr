"""
Von Mises-Fisher mixture loss and head for directional prediction on S².

Provides:
    - VMFMixtureLoss:  NLL of a K-component vMF mixture on S²
    - VMFMixtureHead:  Maps event embedding → K vMF mixture components
    - vmf_weighted_mean: κ-weighted mean direction for point estimates
    - angular_error_deg: Mean angular error in degrees

For K=1 this reduces to a single vMF NLL — well-calibrated probabilistic
directional loss with adaptive gradient weighting via κ.
For K>1 the mixture can represent multimodal posteriors (or, as observed
empirically, use low-κ components as a "safety valve" for noisy events).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _log_sinh_stable(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(sinh(x)) for x > 0."""
    return torch.where(
        x > 20.0,
        x - math.log(2.0),
        torch.log(torch.sinh(x) + 1e-8),
    )


# ---------------------------------------------------------------------------
# VMF mixture loss
# ---------------------------------------------------------------------------

class VMFMixtureLoss(nn.Module):
    """
    Negative log-likelihood of a mixture of K von Mises-Fisher distributions on S².

    p(x) = Σ_k π_k · C_3(κ_k) · exp(κ_k · μ_k^T x)
    C_3(κ) = κ / (4π sinh(κ))

    Args:
        kappa_min: Minimum κ (added after softplus)
        kappa_max: Maximum κ (clamped)
        kappa_reg: L2 regularization weight on κ
    """

    def __init__(
        self,
        kappa_min: float = 1.0,
        kappa_max: float = 500.0,
        kappa_reg: float = 1e-4,
    ):
        super().__init__()
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_reg = kappa_reg

    def forward(
        self,
        mu: torch.Tensor,
        raw_kappa: torch.Tensor,
        log_weights: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mu:          (B, K, 3) unit directions
            raw_kappa:   (B, K) raw concentration (before softplus)
            log_weights: (B, K) unnormalized log mixture weights
            target:      (B, 3) target unit vector

        Returns:
            Scalar NLL (lower is better)
        """
        # Always fp32 — sinh/exp overflow in fp16
        with torch.amp.autocast("cuda", enabled=False):
            return self._forward_fp32(
                mu.float(), raw_kappa.float(), log_weights.float(), target.float()
            )

    def _forward_fp32(
        self,
        mu: torch.Tensor,
        raw_kappa: torch.Tensor,
        log_weights: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        raw_kappa = torch.clamp(raw_kappa, min=-20.0, max=20.0)
        kappa = torch.clamp(
            F.softplus(raw_kappa) + self.kappa_min, max=self.kappa_max
        )

        log_pi = F.log_softmax(log_weights, dim=-1)  # (B, K)
        dot = torch.sum(mu * target.unsqueeze(1), dim=-1)  # (B, K)
        log_c3 = (
            torch.log(kappa + 1e-8)
            - math.log(4.0 * math.pi)
            - _log_sinh_stable(kappa)
        )
        log_component = log_pi + log_c3 + kappa * dot  # (B, K)
        log_likelihood = torch.logsumexp(log_component, dim=-1)  # (B,)
        nll = -log_likelihood.mean()
        if self.kappa_reg > 0:
            nll = nll + self.kappa_reg * (kappa**2).mean()
        return nll


# ---------------------------------------------------------------------------
# VMF mixture head
# ---------------------------------------------------------------------------

class VMFMixtureHead(nn.Module):
    """
    Maps an event embedding → K vMF mixture components.

    Per component: unit direction μ (3D), raw κ scalar, log mixture weight.

    Args:
        d_model:      Input embedding dimension
        hidden_dim:   Hidden layer dimension
        n_components: Number of vMF mixture components (K)
    """

    def __init__(self, d_model: int, hidden_dim: int, n_components: int = 1):
        super().__init__()
        self.n_components = n_components
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_components * 5),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embedding: torch.Tensor):
        """
        Args:
            embedding: (B, d_model) event embedding (e.g. from CLS token or readout)

        Returns:
            mu:          (B, K, 3) unit directions
            raw_kappa:   (B, K) raw concentration parameters
            log_weights: (B, K) unnormalized log mixture weights
        """
        B = embedding.shape[0]
        K = self.n_components
        raw = self.net(embedding).view(B, K, 5)
        mu = F.normalize(raw[:, :, :3], dim=-1)  # (B, K, 3)
        raw_kappa = raw[:, :, 3]  # (B, K)
        log_weights = raw[:, :, 4]  # (B, K)
        return mu, raw_kappa, log_weights


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def vmf_weighted_mean(
    mu: torch.Tensor,
    raw_kappa: torch.Tensor,
    log_weights: torch.Tensor,
    kappa_min: float = 1.0,
    kappa_max: float = 500.0,
) -> torch.Tensor:
    """
    κ-weighted mean direction from a vMF mixture → (B, 3) unit vector.

    Higher-κ components (more confident) contribute more to the mean.
    """
    mu = mu.float()
    raw_kappa = torch.clamp(raw_kappa.float(), -20.0, 20.0)
    log_weights = log_weights.float()
    kappa = torch.clamp(F.softplus(raw_kappa) + kappa_min, max=kappa_max)
    w = F.softmax(log_weights, dim=-1)
    mean_dir = (w * kappa).unsqueeze(-1) * mu
    return F.normalize(mean_dir.sum(dim=1), dim=-1)


def angular_error_deg(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean angular error in degrees between two sets of unit vectors."""
    dot = (pred * target).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.rad2deg(torch.acos(dot)).mean()

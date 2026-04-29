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
    """Numerically stable log(sinh(x)) for x > 0, with finite gradient.

    Uses the identity log(sinh(x)) = x - log(2) + log1p(-exp(-2x)).
    Single branch (no torch.where → no masked-NaN-backward gotcha), and
    exp(-2x) underflows cleanly to 0 for large x, leaving the dominant
    x - log(2) term. For our regime (x ≥ kappa_min ≥ 1) the log1p argument
    stays in (-1, 0), so the derivative

        d/dx log(sinh(x)) = 1 + 2 exp(-2x) / (1 - exp(-2x)) = coth(x)

    evaluates finite for all x ∈ [1, ∞). The legacy where-based form
    silently returned NaN gradients for x > ~89 (fp32 sinh overflow in the
    unused branch).
    """
    return x - math.log(2.0) + torch.log1p(-torch.exp(-2.0 * x))


def _kappa_from_raw(
    raw_kappa: torch.Tensor,
    kappa_min: float,
    kappa_max: float,
    param: str = 'softplus',
    kappa_temperature: float = 1.0,
) -> torch.Tensor:
    """Map raw κ scalar to κ ∈ [κ_min, κ_max].

    param='softplus' (default):
        κ = clamp(softplus(raw) + κ_min, max=κ_max)
        Smooth, asymmetric. κ_max is a non-binding safety net for the
        gradient-on-μ amplitude. This is what the unclamped K=1/K=3 runs use.

    param='sigmoid':
        κ = κ_min + (κ_max - κ_min) · σ(raw / T)
        PR #6's bounded form. Symmetric saturation at both ends; κ_max is a
        modeling assumption (range cap), not a safety net. At zero-init bias
        and T=1, κ_init ≈ (κ_min + κ_max) / 2 — overconfident.
    """
    if param == 'sigmoid':
        t = max(float(kappa_temperature), 1e-6)
        span = float(kappa_max) - float(kappa_min)
        return float(kappa_min) + span * torch.sigmoid(raw_kappa / t)
    # default: softplus + safety-net clamp
    return torch.clamp(F.softplus(raw_kappa) + kappa_min, max=kappa_max)


# ---------------------------------------------------------------------------
# VMF mixture loss
# ---------------------------------------------------------------------------

class VMFMixtureLoss(nn.Module):
    """
    Negative log-likelihood of a mixture of K von Mises-Fisher distributions on S².

    p(x) = Σ_k π_k · C_3(κ_k) · exp(κ_k · μ_k^T x)
    C_3(κ) = κ / (4π sinh(κ))

    κ is parameterized as clamp(softplus(raw_κ) + κ_min, max=κ_max). The
    upper clamp is a *safety net* set well above any physically meaningful
    concentration (default 10000 ↔ σ ≈ 0.8° on S²; our best events imply
    MLE-optimal κ ~ 1600). It bounds the per-step gradient on μ (which
    scales with κ) without artificially capping training-time κ — the
    binding constraint in the legacy code was clamp(raw_κ, -20, 20), which
    capped κ at ~21 and is what we removed.

    Args:
        kappa_min: Minimum κ (softplus offset, κ floor)
        kappa_max: Safety-net ceiling on κ (non-binding for any reasonable run)
        kappa_reg: L2 regularization weight on κ (annealable via the buffer)
        kappa_param: 'softplus' (default) or 'sigmoid' (PR #6 form)
        kappa_temperature: Sigmoid steepness; only used when kappa_param='sigmoid'
    """

    def __init__(
        self,
        kappa_min: float = 1.0,
        kappa_max: float = 10000.0,
        kappa_reg: float = 1e-4,
        kappa_param: str = 'softplus',
        kappa_temperature: float = 1.0,
    ):
        super().__init__()
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_param = kappa_param
        self.kappa_temperature = kappa_temperature
        # Registered buffer so torch.compile sees in-place updates (used by
        # train_flat.py's anneal schedule). In-place ops via .fill_() keep the
        # compiled graph valid.
        self.register_buffer('kappa_reg', torch.tensor(float(kappa_reg)))

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
        kappa = _kappa_from_raw(
            raw_kappa, self.kappa_min, self.kappa_max,
            param=self.kappa_param, kappa_temperature=self.kappa_temperature,
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
        # Always apply (kappa_reg == 0 is a no-op); no Python branch so the
        # compiled graph stays stable when kappa_reg is annealed.
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
    kappa_max: float = 10000.0,
    kappa_param: str = 'softplus',
    kappa_temperature: float = 1.0,
) -> torch.Tensor:
    """
    κ-weighted mean direction from a vMF mixture → (B, 3) unit vector.

    Higher-κ components (more confident) contribute more to the mean.
    Uses the same κ map as VMFMixtureLoss (selectable via kappa_param).
    """
    mu = mu.float()
    raw_kappa = raw_kappa.float()
    log_weights = log_weights.float()
    kappa = _kappa_from_raw(
        raw_kappa, kappa_min, kappa_max,
        param=kappa_param, kappa_temperature=kappa_temperature,
    )
    w = F.softmax(log_weights, dim=-1)
    mean_dir = (w * kappa).unsqueeze(-1) * mu
    return F.normalize(mean_dir.sum(dim=1), dim=-1)


def angular_error_deg(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean angular error in degrees between two sets of unit vectors."""
    dot = (pred * target).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.rad2deg(torch.acos(dot)).mean()

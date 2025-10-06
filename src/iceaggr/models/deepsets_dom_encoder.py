"""
DeepSets-based DOM encoder for hierarchical IceCube transformer.

This module implements permutation-invariant pulse aggregation within each DOM using
the DeepSets architecture with relative temporal/charge encodings.

Key features:
- Handles variable-length pulse sequences (1 to 1000s per DOM)
- Relative encodings capture temporal structure (rescattering vs direct hits)
- Multi-head pooling preserves different information types
- Pure tensor operations for hardware efficiency
"""

import torch
import torch.nn as nn
from typing import Tuple

from iceaggr.utils import get_logger

logger = get_logger(__name__)

#TODO consider installing torch_scatter for potentially faster scatter ops
# Native PyTorch scatter operations (no torch_scatter dependency needed)
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """Scatter mean: compute mean of src values for each unique index."""
    out = torch.zeros(dim_size, src.shape[1], dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
    count.scatter_add_(dim, index.unsqueeze(1), torch.ones_like(src[:, :1]))
    return out / (count + 1e-8)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> tuple:
    """Scatter max: compute max of src values for each unique index."""
    out = torch.full((dim_size, src.shape[1]), float('-inf'), dtype=src.dtype, device=src.device)
    out.scatter_reduce_(dim, index.unsqueeze(1).expand_as(src), src, reduce='amax', include_self=False)
    return out, None


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """Scatter add: sum src values for each unique index."""
    if src.dim() == 1:
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, index, src)
    else:
        out = torch.zeros(dim_size, src.shape[1], dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
    return out


class RelativeEncoder(nn.Module):
    """
    Computes relative encodings for pulses within each DOM.

    These features capture temporal and charge relationships that are lost
    during pooling but are critical for distinguishing direct Cherenkov hits
    from rescattered photons.

    Features:
        Temporal (3):
            - Δt_first: time - first_pulse_time (direct vs scattered)
            - Δt_median: time - median_pulse_time (position in sequence)
            - t_normalized: (time - mean) / (std + eps) (relative timing)

        Charge (3):
            - charge_fraction: charge / total_charge (relative importance)
            - charge_ratio: charge / max_charge (brightness ratio)
            - log_charge_rank: log(charge_rank / N) (ordering information)

    Args:
        eps: Small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        times: torch.Tensor,
        charges: torch.Tensor,
        dom_idx: torch.Tensor,
        num_doms: int
    ) -> torch.Tensor:
        """
        Compute relative encodings for pulses.

        Args:
            times: (n_pulses,) pulse times
            charges: (n_pulses,) pulse charges
            dom_idx: (n_pulses,) which DOM each pulse belongs to
            num_doms: Total number of DOMs in batch

        Returns:
            rel_features: (n_pulses, 6) relative encodings
        """
        # Temporal features
        # First pulse time per DOM
        first_times = scatter_min_1d(times, dom_idx, dim=0, dim_size=num_doms)[0]
        first_times_expanded = first_times[dom_idx]
        delta_t_first = times - first_times_expanded

        # Median pulse time per DOM (approximate with mean for efficiency)
        mean_times = scatter_mean_1d(times, dom_idx, dim=0, dim_size=num_doms)
        mean_times_expanded = mean_times[dom_idx]
        delta_t_median = times - mean_times_expanded

        # Normalized time (standardized per DOM)
        time_std = scatter_std_1d(times, dom_idx, dim=0, dim_size=num_doms, mean=mean_times)
        time_std_expanded = time_std[dom_idx]
        t_normalized = (times - mean_times_expanded) / (time_std_expanded + self.eps)

        # Charge features
        # Total charge per DOM
        total_charges_1d = torch.zeros(num_doms, dtype=charges.dtype, device=charges.device)
        total_charges_1d.scatter_add_(0, dom_idx, charges)
        total_charges_expanded = total_charges_1d[dom_idx]
        charge_fraction = charges / (total_charges_expanded + self.eps)

        # Max charge per DOM
        max_charges_1d = torch.full((num_doms,), float('-inf'), dtype=charges.dtype, device=charges.device)
        max_charges_1d.scatter_reduce_(0, dom_idx, charges, reduce='amax', include_self=False)
        max_charges_expanded = max_charges_1d[dom_idx]
        charge_ratio = charges / (max_charges_expanded + self.eps)

        # Charge rank (lower is brighter)
        # This is approximate - sorts within each DOM
        charge_rank = compute_charge_rank(charges, dom_idx, num_doms)
        dom_sizes_1d = torch.zeros(num_doms, dtype=torch.long, device=dom_idx.device)
        dom_sizes_1d.scatter_add_(0, dom_idx, torch.ones_like(dom_idx))
        dom_sizes_expanded = dom_sizes_1d[dom_idx].float()
        log_charge_rank = torch.log((charge_rank + 1) / (dom_sizes_expanded + self.eps))

        # Stack all features
        rel_features = torch.stack([
            delta_t_first,
            delta_t_median,
            t_normalized,
            charge_fraction,
            charge_ratio,
            log_charge_rank
        ], dim=1)  # (n_pulses, 6)

        return rel_features


def scatter_min_1d(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter min operation for 1D tensors."""
    out = torch.full((dim_size,), float('inf'), dtype=src.dtype, device=src.device)
    arg_out = torch.zeros(dim_size, dtype=torch.long, device=src.device)
    out.scatter_reduce_(dim, index, src, reduce='amin', include_self=False)
    return out, arg_out


def scatter_std_1d(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int, mean: torch.Tensor) -> torch.Tensor:
    """Compute standard deviation per group for 1D tensors."""
    mean_expanded = mean[index]
    sq_diff = (src - mean_expanded) ** 2

    # Sum squared differences per group
    var = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    var.scatter_add_(dim, index, sq_diff)

    # Divide by count per group
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.scatter_add_(dim, index, torch.ones_like(src))

    var = var / (count + 1e-8)
    return torch.sqrt(var + 1e-8)


def scatter_mean_1d(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """Scatter mean for 1D tensors."""
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, index, src)
    count.scatter_add_(dim, index, torch.ones_like(src))
    return out / (count + 1e-8)


def compute_charge_rank(charges: torch.Tensor, dom_idx: torch.Tensor, num_doms: int) -> torch.Tensor:
    """
    Compute charge rank within each DOM (0 = brightest).

    This is an approximate implementation that's efficient but not exact.
    Uses argsort within each DOM.
    """
    # Create a sortable key: (dom_idx, -charge) so we can sort by DOM then charge descending
    # We'll use a simpler approach: just count how many charges in the same DOM are larger

    ranks = torch.zeros_like(charges, dtype=torch.long)

    # For each unique DOM, compute ranks
    for dom_id in range(num_doms):
        mask = dom_idx == dom_id
        if mask.sum() == 0:
            continue

        dom_charges = charges[mask]
        # Rank by descending charge (largest charge gets rank 0)
        sorted_indices = torch.argsort(dom_charges, descending=True)
        rank_values = torch.empty_like(sorted_indices)
        rank_values[sorted_indices] = torch.arange(len(sorted_indices), device=charges.device)

        ranks[mask] = rank_values

    return ranks


class DeepSetsDOMEncoder(nn.Module):
    """
    DeepSets encoder for DOM-level pulse aggregation.

    Architecture:
        pulses (+ relative encodings) → MLP_encode → pooling → MLP_decode → DOM embedding

    Multi-head pooling:
        - Mean: Average signal representation
        - Max: Strongest feature per dimension
        - Charge-weighted: Physics-informed weighting

    Args:
        d_pulse: Dimension of input pulse features (default: 4 = time, charge, sensor_id, auxiliary)
        d_relative: Dimension of relative encodings (default: 6)
        d_latent: Hidden dimension for encoded pulses
        d_output: Output DOM embedding dimension
        hidden_dim: Hidden layer dimension in MLPs
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_pulse: int = 4,
        d_relative: int = 6,
        d_latent: int = 128,
        d_output: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_pulse = d_pulse
        self.d_relative = d_relative
        self.d_latent = d_latent
        self.d_output = d_output

        # Relative encoding module
        self.relative_encoder = RelativeEncoder()

        # Encoding MLP: pulse features + relative encodings → latent space
        self.mlp_encode = nn.Sequential(
            nn.Linear(d_pulse + d_relative, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent),
            nn.LayerNorm(d_latent),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Decoding MLP: pooled features → output embedding
        # Input: 3 * d_latent (mean + max + charge-weighted)
        self.mlp_decode = nn.Sequential(
            nn.Linear(3 * d_latent, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_output),
            nn.LayerNorm(d_output),
        )

        logger.info(f"DeepSetsDOMEncoder: {d_pulse}+{d_relative} → {d_latent} → {d_output}")
        logger.info(f"  Pooling: 3-head (mean + max + charge-weighted)")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        pulse_features: torch.Tensor,
        pulse_to_dom_idx: torch.Tensor,
        num_doms: int
    ) -> torch.Tensor:
        """
        Encode pulses to DOM embeddings using DeepSets.

        Args:
            pulse_features: (n_pulses, d_pulse) pulse features [time, charge, sensor_id, auxiliary]
            pulse_to_dom_idx: (n_pulses,) which DOM each pulse belongs to
            num_doms: Total number of DOMs in batch

        Returns:
            dom_embeddings: (num_doms, d_output) DOM-level embeddings
        """
        n_pulses = pulse_features.shape[0]

        # Extract time and charge for relative encodings
        times = pulse_features[:, 0]  # (n_pulses,)
        charges = pulse_features[:, 1]  # (n_pulses,)

        # Compute relative encodings
        rel_features = self.relative_encoder(
            times, charges, pulse_to_dom_idx, num_doms
        )  # (n_pulses, d_relative)

        # Concatenate pulse features with relative encodings
        pulse_with_rel = torch.cat([pulse_features, rel_features], dim=1)  # (n_pulses, d_pulse + d_relative)

        # Encode pulses to latent space
        encoded = self.mlp_encode(pulse_with_rel)  # (n_pulses, d_latent)

        # Multi-head pooling
        # 1. Mean pooling
        mean_pool = scatter_mean(encoded, pulse_to_dom_idx, dim=0, dim_size=num_doms)  # (num_doms, d_latent)

        # 2. Max pooling
        max_pool = scatter_max(encoded, pulse_to_dom_idx, dim=0, dim_size=num_doms)[0]  # (num_doms, d_latent)

        # 3. Charge-weighted pooling
        charges_expanded = charges.unsqueeze(1)  # (n_pulses, 1)
        weighted_encoded = encoded * charges_expanded  # (n_pulses, d_latent)
        weighted_sum = scatter_add(weighted_encoded, pulse_to_dom_idx, dim=0, dim_size=num_doms)  # (num_doms, d_latent)
        charge_sum = scatter_add(charges, pulse_to_dom_idx, dim=0, dim_size=num_doms).unsqueeze(1)  # (num_doms, 1)
        charge_pool = weighted_sum / (charge_sum + 1e-8)  # (num_doms, d_latent)

        # Concatenate pooling heads
        pooled = torch.cat([mean_pool, max_pool, charge_pool], dim=1)  # (num_doms, 3 * d_latent)

        # Decode to output embeddings
        dom_embeddings = self.mlp_decode(pooled)  # (num_doms, d_output)

        return dom_embeddings

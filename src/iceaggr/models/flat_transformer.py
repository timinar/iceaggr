"""
Simplified flat transformer for IceCube direction prediction.

Instead of a hierarchical T1→T2 architecture, this model:
1. Concatenates the first K pulse features per DOM into a flat vector
2. Prepends DOM geometry (x, y, z) and pulse count
3. Feeds these fixed-size DOM tokens into a single transformer
4. Predicts direction from CLS token

With K=84, input_dim = 4 + 3*84 = 256 (matches d_model naturally).

This covers the vast majority of DOMs: 50-70% have ≤10 pulses,
and 99%+ of DOMs have <84 pulses per the data analysis.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from .event_transformer import TransformerBlock
from .directional_head import DirectionalHead


def build_flat_dom_vectors(
    pulse_features: torch.Tensor,
    dom_positions: torch.Tensor,
    pulse_to_dom_idx: torch.Tensor,
    pulse_idx_in_dom: torch.Tensor,
    dom_pulse_counts: torch.Tensor,
    total_doms: int,
    max_pulses_per_dom: int = 84,
) -> torch.Tensor:
    """
    Build flat feature vectors per DOM by concatenating first K pulse features.

    Each DOM gets a vector: [x, y, z, log_n_pulses, t1, q1, a1, ..., tK, qK, aK]
    Pulses beyond K are dropped. DOMs with fewer than K pulses are zero-padded.

    Normalization (absolute, same for all events):
        - time: (t - 1e4) / 3e4
        - charge: log10(charge) / 3.0
        - auxiliary: aux - 0.5
        - n_pulses: log1p(n) / 3.0 - 1.0
        - positions: already ~[-1, 1]

    Args:
        pulse_features: (total_pulses, 4) - [time, charge, sensor_id, auxiliary]
        dom_positions: (total_doms, 3) - DOM x, y, z coordinates
        pulse_to_dom_idx: (total_pulses,) - DOM index for each pulse
        pulse_idx_in_dom: (total_pulses,) - pulse index within its DOM (0-based, time-sorted)
        dom_pulse_counts: (total_doms,) - number of pulses per DOM
        total_doms: number of DOMs in the batch
        max_pulses_per_dom: K, max pulses to keep per DOM (default: 84)

    Returns:
        dom_vectors: (total_doms, 4 + 3*K) flat feature vectors
    """
    device = pulse_features.device
    K = max_pulses_per_dom

    # Keep only first K pulses per DOM (pulses are already time-sorted)
    keep_mask = pulse_idx_in_dom < K

    time_raw = pulse_features[keep_mask, 0]
    charge_raw = pulse_features[keep_mask, 1]
    aux_raw = pulse_features[keep_mask, 3]
    dom_idx = pulse_to_dom_idx[keep_mask]
    p_idx = pulse_idx_in_dom[keep_mask]

    # Normalize pulse features (absolute normalization)
    time_norm = (time_raw - 1e4) / 3e4
    charge_norm = torch.log10(charge_raw.clamp(min=1e-6)) / 3.0
    aux_norm = aux_raw - 0.5

    # Scatter into (total_doms, K, 3) — zeros for missing pulses
    pulse_tensor = torch.zeros(total_doms, K, 3, device=device, dtype=pulse_features.dtype)
    pulse_tensor[dom_idx, p_idx, 0] = time_norm
    pulse_tensor[dom_idx, p_idx, 1] = charge_norm
    pulse_tensor[dom_idx, p_idx, 2] = aux_norm

    # Flatten pulse features: (total_doms, K*3)
    pulse_flat = pulse_tensor.reshape(total_doms, K * 3)

    # DOM-level features
    n_pulses_norm = (torch.log1p(dom_pulse_counts.float()) / 3.0 - 1.0).unsqueeze(1)  # (D, 1)

    # Concatenate: [x, y, z, n_pulses, pulse_features_flat]
    dom_vectors = torch.cat([
        dom_positions,          # (D, 3)
        n_pulses_norm,          # (D, 1)
        pulse_flat,             # (D, K*3)
    ], dim=1)  # (D, 4 + K*3)

    return dom_vectors


def pad_to_event_batch(
    dom_vectors: torch.Tensor,
    dom_to_event_idx: torch.Tensor,
    event_dom_counts: torch.Tensor,
    batch_size: int,
    max_doms: int = 128,
    dom_min_time: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad DOM vectors to fixed (batch_size, max_doms, input_dim) with attention mask.

    Events with more than max_doms DOMs are subsampled (earliest DOMs kept).
    Events with fewer DOMs are zero-padded.

    Args:
        dom_vectors: (total_doms, input_dim) flat DOM feature vectors
        dom_to_event_idx: (total_doms,) event index for each DOM
        event_dom_counts: (batch_size,) DOMs per event
        batch_size: number of events
        max_doms: maximum DOMs per event (default: 128)
        dom_min_time: (total_doms,) first pulse time per DOM (for subsampling)

    Returns:
        Tuple of:
            - padded: (batch_size, max_doms, input_dim)
            - mask: (batch_size, max_doms) boolean, True = valid DOM
    """
    device = dom_vectors.device
    input_dim = dom_vectors.shape[1]
    total_doms = dom_vectors.shape[0]

    # Compute within-event DOM indices
    dom_starts = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
    dom_starts[1:] = event_dom_counts.cumsum(0)
    dom_idx_in_event = torch.arange(total_doms, device=device) - dom_starts[dom_to_event_idx]

    # Handle events with too many DOMs: keep earliest by min_time
    needs_subsample = event_dom_counts > max_doms

    if needs_subsample.any():
        if dom_min_time is not None:
            priority = -dom_min_time.to(device)  # negate so topk picks earliest
        else:
            priority = torch.rand(total_doms, device=device)

        keep_mask = torch.ones(total_doms, dtype=torch.bool, device=device)

        for event_idx in needs_subsample.nonzero(as_tuple=True)[0]:
            start = dom_starts[event_idx]
            end = dom_starts[event_idx + 1]
            event_priorities = priority[start:end]
            _, top_indices = event_priorities.topk(max_doms, largest=True)
            keep_mask[start:end] = False
            keep_mask[start + top_indices] = True

        kept = keep_mask.nonzero(as_tuple=True)[0]
        dom_vectors = dom_vectors[kept]
        dom_to_event_idx = dom_to_event_idx[kept]

        # Recompute within-event indices
        clamped_counts = event_dom_counts.clamp(max=max_doms)
        kept_starts = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        kept_starts[1:] = clamped_counts.cumsum(0)
        dom_idx_in_event = torch.arange(dom_vectors.shape[0], device=device) - kept_starts[dom_to_event_idx]

    # Scatter into padded tensors
    valid = dom_idx_in_event < max_doms
    ev_idx = dom_to_event_idx[valid]
    dom_idx = dom_idx_in_event[valid]

    padded = torch.zeros(batch_size, max_doms, input_dim, device=device, dtype=dom_vectors.dtype)
    mask = torch.zeros(batch_size, max_doms, dtype=torch.bool, device=device)

    padded[ev_idx, dom_idx] = dom_vectors[valid]
    mask[ev_idx, dom_idx] = True

    return padded, mask


class FlatTransformerModel(nn.Module):
    """
    Simplified transformer for IceCube direction prediction.

    Each DOM is represented as a flat vector of concatenated pulse features.
    A single transformer processes all DOMs in an event to predict direction.

    Architecture:
        DOM vectors (B, max_doms, input_dim)
        → Linear projection → (B, max_doms, d_model)
        → Prepend CLS token → (B, max_doms+1, d_model)
        → N × TransformerBlock (with attention masking)
        → CLS token → DirectionalHead → (B, 3) unit vectors

    Args:
        config: Dictionary with:
            - max_pulses_per_dom: K, pulses per DOM (default: 84)
            - d_model: transformer hidden dim (default: 256)
            - max_doms: max DOMs per event (default: 128)
            - num_heads: attention heads (default: 8)
            - num_layers: transformer layers (default: 4)
            - hidden_dim: FFN hidden dim (default: 512)
            - head_hidden_dim: direction head hidden dim (default: 128)
            - dropout: dropout rate (default: 0.1)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if config is None:
            config = {}

        self.max_pulses_per_dom = config.get('max_pulses_per_dom', 84)
        self.d_model = config.get('d_model', 256)
        self.max_doms = config.get('max_doms', 128)
        dropout = config.get('dropout', 0.1)

        # Input dimension: 3 (xyz) + 1 (n_pulses) + 3*K (pulse features)
        self.input_dim = 4 + 3 * self.max_pulses_per_dom

        # Project flat vector to model dimension
        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        # CLS token for event-level aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Transformer blocks
        num_layers = config.get('num_layers', 4)
        num_heads = config.get('num_heads', 8)
        hidden_dim = config.get('hidden_dim', 512)

        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)

        # Direction prediction head
        self.direction_head = DirectionalHead(
            embed_dim=self.d_model,
            hidden_dim=config.get('head_hidden_dim', 128),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        dom_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            dom_vectors: (batch_size, max_doms, input_dim) flat DOM features
            padding_mask: (batch_size, max_doms) True = valid DOM

        Returns:
            Predicted directions (batch_size, 3) as unit vectors
        """
        B = dom_vectors.shape[0]
        device = dom_vectors.device

        # Project to model dimension
        x = self.input_proj(dom_vectors)  # (B, max_doms, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, max_doms+1, d_model)

        # Extend mask for CLS (always valid)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Transformer
        for block in self.blocks:
            x = block(x, full_mask)

        x = self.final_norm(x)

        # CLS token → direction
        return self.direction_head(x[:, 0, :])

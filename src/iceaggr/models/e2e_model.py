"""
End-to-end hierarchical transformer model (T1 + T2).

Combines DOM-level and event-level transformers for neutrino direction reconstruction.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .dom_transformer import DOMTransformer
from .event_transformer import EventTransformer, EventAccumulator


class HierarchicalTransformer(nn.Module):
    """
    End-to-end hierarchical transformer for neutrino direction reconstruction.

    Architecture:
        1. T1 (DOMTransformer): Pulses → DOM embeddings
        2. T2 (EventTransformer): DOM embeddings + geometry → Direction prediction
    """

    def __init__(
        self,
        # T1 config
        d_model: int = 128,
        t1_n_heads: int = 8,
        t1_n_layers: int = 4,
        t1_max_seq_len: int = 512,
        t1_max_batch_size: int = 64,
        # T2 config
        t2_n_heads: int = 8,
        t2_n_layers: int = 4,
        t2_max_doms: int = 2048,
        # Other
        dropout: float = 0.1,
        sensor_geometry_path: str = None,
    ):
        """
        Args:
            d_model: Embedding dimension (shared by T1 and T2)
            t1_n_heads: Number of attention heads in T1
            t1_n_layers: Number of transformer layers in T1
            t1_max_seq_len: Maximum sequence length for T1
            t1_max_batch_size: Maximum batch size for T1 (prevents OOM)
            t2_n_heads: Number of attention heads in T2
            t2_n_layers: Number of transformer layers in T2
            t2_max_doms: Maximum DOMs per event for T2
            dropout: Dropout rate
            sensor_geometry_path: Path to sensor geometry CSV
        """
        super().__init__()

        self.d_model = d_model
        self.t1_max_batch_size = t1_max_batch_size  # Store for forward pass

        # T1: DOM-level transformer
        self.t1 = DOMTransformer(
            d_model=d_model,
            n_heads=t1_n_heads,
            n_layers=t1_n_layers,
            max_seq_len=t1_max_seq_len,
            dropout=dropout,
        )

        # T2: Event-level transformer
        self.t2 = EventTransformer(
            d_model=d_model,
            n_heads=t2_n_heads,
            n_layers=t2_n_layers,
            max_doms=t2_max_doms,
            dropout=dropout,
            geometry_path=sensor_geometry_path,
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through T1 and T2.

        Args:
            batch: Output from collate_dom_packing containing:
                - packed_sequences: (bsz, max_seq_len, 4)
                - dom_boundaries: (bsz, max_seq_len)
                - dom_mask: (bsz, max_seq_len)
                - metadata: dict with event/DOM mappings

        Returns:
            predictions: (batch_size, 2) - azimuth, zenith angles
        """
        # T1: Process pulses → DOM embeddings
        dom_embeddings, metadata = self.t1(batch, max_batch_size=self.t1_max_batch_size)

        # T2: Process DOMs → Direction prediction
        predictions = self.t2(
            dom_embeddings=dom_embeddings,
            dom_ids=metadata['sensor_ids'],
            dom_to_event_idx=metadata['dom_to_event_idx'],
            batch_size=len(metadata['event_ids']),
        )

        return predictions

    def forward_with_accumulation(
        self,
        t1_batches: list[Dict[str, Any]],
        accumulator: EventAccumulator,
    ) -> torch.Tensor:
        """
        Forward pass with event accumulation for extreme events.

        Use this when events are split across multiple T1 batches.

        Args:
            t1_batches: List of T1 batches (from DataLoader)
            accumulator: EventAccumulator instance

        Returns:
            predictions: (total_events, 2) - azimuth, zenith angles
        """
        # Process all T1 batches
        for batch in t1_batches:
            dom_embeddings, metadata = self.t1(batch)
            accumulator.add_batch(dom_embeddings, metadata)

        # Get complete events and process with T2
        all_predictions = []
        for t2_batch in accumulator.get_complete_events(batch_size=32):
            predictions = self.t2(**t2_batch)
            all_predictions.append(predictions)

        return torch.cat(all_predictions, dim=0)

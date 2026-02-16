"""Model components for IceCube flat transformer direction prediction."""

from .losses import (
    angular_distance_loss,
    angular_dist_score_unit_vectors,
    angles_to_unit_vector,
    unit_vector_to_angles,
    AngularDistanceLoss,
)
from .event_transformer import EventTransformer, TransformerBlock
from .directional_head import DirectionalHead
from .flat_transformer import (
    FlatTransformerModel,
    build_flat_dom_vectors,
    pad_to_event_batch,
)
from .flat_transformer_v2 import FlatTransformerV2

__all__ = [
    # Losses
    "angular_distance_loss",
    "angular_dist_score_unit_vectors",
    "angles_to_unit_vector",
    "unit_vector_to_angles",
    "AngularDistanceLoss",
    # Event processing
    "EventTransformer",
    "TransformerBlock",
    # Direction prediction
    "DirectionalHead",
    # Full model (flat/simplified)
    "FlatTransformerModel",
    "build_flat_dom_vectors",
    "pad_to_event_batch",
    # Full model (flat v2 / nanochat-style)
    "FlatTransformerV2",
]

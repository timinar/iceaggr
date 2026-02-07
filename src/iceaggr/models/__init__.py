"""Model components for IceCube hierarchical direction prediction."""

from .losses import (
    angular_distance_loss,
    angular_dist_score_unit_vectors,
    angles_to_unit_vector,
    unit_vector_to_angles,
    AngularDistanceLoss,
)
from .pulse_embedder import PulseEmbedder, build_pulse_features
from .dom_encoder import DOMPooling, DOMTransformerEncoder
from .event_transformer import EventTransformer, TransformerBlock
from .directional_head import DirectionalHead
from .hierarchical_model import HierarchicalDOMModel, count_parameters
from .flat_transformer import (
    FlatTransformerModel,
    build_flat_dom_vectors,
    pad_to_event_batch,
)

__all__ = [
    # Losses
    "angular_distance_loss",
    "angular_dist_score_unit_vectors",
    "angles_to_unit_vector",
    "unit_vector_to_angles",
    "AngularDistanceLoss",
    # Pulse embedding
    "PulseEmbedder",
    "build_pulse_features",
    # DOM encoding
    "DOMPooling",
    "DOMTransformerEncoder",
    # Event processing
    "EventTransformer",
    "TransformerBlock",
    # Direction prediction
    "DirectionalHead",
    # Full model (hierarchical)
    "HierarchicalDOMModel",
    "count_parameters",
    # Full model (flat/simplified)
    "FlatTransformerModel",
    "build_flat_dom_vectors",
    "pad_to_event_batch",
]

"""Data loading utilities for IceCube events."""

from .dataset import (
    IceCubeDataset,
    IceCubeSubsampledDataset,
    get_dataloader,
    get_subsampled_dataloader,
)
from .samplers import BatchAwareSampler, BucketBatchSampler
from .collators import (
    collate_variable_length,
    collate_with_dom_grouping,
    collate_with_dom_grouping_legacy,
    collate_padded_subsampled,
    make_collate_flat,
)
from .collators_npe import (
    FEATURE_COLS_NPE,
    INPUT_DIM_NPE,
    compute_dom_features_npe,
    normalize_dom_features_npe,
    make_collate_npe15,
)
from .geometry import GeometryLoader

__all__ = [
    # Datasets
    "IceCubeDataset",
    "IceCubeSubsampledDataset",
    # Dataloaders
    "get_dataloader",
    "get_subsampled_dataloader",
    # Samplers
    "BatchAwareSampler",
    "BucketBatchSampler",
    # Collators
    "collate_variable_length",
    "collate_with_dom_grouping",
    "collate_with_dom_grouping_legacy",
    "collate_padded_subsampled",
    "make_collate_flat",
    # NPE-15 summary-statistics tokenization
    "FEATURE_COLS_NPE",
    "INPUT_DIM_NPE",
    "compute_dom_features_npe",
    "normalize_dom_features_npe",
    "make_collate_npe15",
    # Geometry
    "GeometryLoader",
]

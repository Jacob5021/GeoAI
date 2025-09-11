"""
Flood Detection & Prediction Submodule
"""

from .app import flood_detector, flood_predictor
from .flood_utils import (
    compute_flood_risk_index,
    classify_fri,
    resample_raster,
    read_raster_as_array,
)

__all__ = [
    "flood_detector",
    "flood_predictor",
    "main",
    "compute_flood_risk_index",
    "classify_fri",
    "resample_raster",
    "read_raster_as_array",
]

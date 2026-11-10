"""Reference implementation of MOSAIC county-month forecasting."""

from .model import EvidenceGatedHurdleForecaster, build_feature_channel_indices

__all__ = ["EvidenceGatedHurdleForecaster", "build_feature_channel_indices"]

__version__ = "1.0.0"

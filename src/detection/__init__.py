"""Feature-only distribution-shift detection for streaming windows."""

from .config import PCAPSIConfig
from .pca_psi import PCAPSIDriftDetector
from .types import DriftSignal

__all__ = ["DriftSignal", "PCAPSIConfig", "PCAPSIDriftDetector"]

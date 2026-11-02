"""Detection output suitable for logging and runtime adaptation decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftSignal:
    """A thresholded distribution statistic, not an accuracy or confidence estimate."""

    detected: bool
    score: float
    threshold: float
    component_scores: tuple[float, ...]
    sample_count: int
    window_id: int | None = None
    detector: str = "pca_psi"

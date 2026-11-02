"""Fixed-reference population stability index (PSI) histograms."""

from dataclasses import dataclass

import numpy as np


def _finite_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("PSI values must be a non-empty, finite one-dimensional array")
    return vector


@dataclass(frozen=True)
class PSIHistogram:
    """Store reference proportions and reuse the same bin edges for every window."""

    edges: np.ndarray
    proportions: np.ndarray
    epsilon: float

    @classmethod
    def from_reference(
        cls, reference: np.ndarray, *, bins: int = 10, epsilon: float = 1e-6
    ) -> "PSIHistogram":
        reference = _finite_vector(reference)
        if (
            isinstance(bins, bool)
            or not isinstance(bins, (int, np.integer))
            or bins < 2
        ):
            raise ValueError("bins must be an integer of at least two")
        if not np.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("epsilon must be finite and strictly between zero and one")
        quantiles = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
        if len(quantiles) == 1:
            # A single unbounded bin would conceal every change from a constant reference.
            value = quantiles[0]
            upper = np.nextafter(value, np.inf)
            if not np.isfinite(upper):
                raise ValueError(
                    "The constant reference is too large to form finite bin boundaries"
                )
            edges = np.array([-np.inf, value, upper, np.inf])
        elif len(quantiles) == 2:
            # Retain separation when tied quantiles leave only two distinct values.
            midpoint = quantiles[0] / 2.0 + quantiles[1] / 2.0
            edges = np.array([-np.inf, midpoint, np.inf])
        else:
            edges = quantiles.copy()
            edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(reference, bins=edges)
        proportions = counts.astype(np.float64) / reference.size
        edges.setflags(write=False)
        proportions.setflags(write=False)
        return cls(edges=edges, proportions=proportions, epsilon=float(epsilon))

    def score(self, current: np.ndarray) -> float:
        current = _finite_vector(current)
        counts, _ = np.histogram(current, bins=self.edges)
        current_ratio = np.clip(counts / current.size, self.epsilon, None)
        reference_ratio = np.clip(self.proportions, self.epsilon, None)
        return float(
            np.sum(
                (current_ratio - reference_ratio)
                * np.log(current_ratio / reference_ratio)
            )
        )

"""Explicit PCA and histogram settings for reproducible shift monitoring."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class PCAPSIConfig:
    """The default PSI threshold is a starting point, not a calibrated guarantee."""

    n_components: int | float | None = 0.95
    threshold: float = 0.2
    bins: int = 10
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        components = self.n_components
        if components is not None:
            if isinstance(components, bool):
                raise ValueError(
                    "n_components must be a count, a variance fraction, or None"
                )
            if isinstance(components, Integral):
                if components < 1:
                    raise ValueError("The PCA component count must be positive")
                object.__setattr__(self, "n_components", int(components))
            elif (
                isinstance(components, Real)
                and np.isfinite(components)
                and 0 < components < 1
            ):
                object.__setattr__(self, "n_components", float(components))
            else:
                raise ValueError(
                    "A PCA variance fraction must be strictly between zero and one"
                )
        if (
            isinstance(self.bins, bool)
            or not isinstance(self.bins, Integral)
            or self.bins < 2
        ):
            raise ValueError("bins must be an integer of at least two")
        object.__setattr__(self, "bins", int(self.bins))
        for name in ("threshold", "epsilon"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be strictly between zero and one")

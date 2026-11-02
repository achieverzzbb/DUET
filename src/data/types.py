"""Data-layer structures."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Supported tasks: binary and multiclass classification.
ProblemType = Literal["binary", "multiclass"]


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset configuration."""

    dataset_name: str
    path: Path
    label_column: str
    problem_type: ProblemType
    ignore_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamConfig:
    """Stream settings for one experiment."""

    bootstrap_size: int = 500
    bootstrap_val_ratio: float = 0.2
    window_size: int = 500
    seed: int = 2026

    # Configured change point; 5 means the shifted stage starts at W5.
    shift_start_window: int | None = None

    # More clusters create finer feature-space regions in the original
    # cluster-window reordering algorithm.
    shift_n_clusters: int = 5
    shift_cluster_order: str = "distance_from_w0"
    shift_block_windows: int = 3
    # ``domain_aligned_propensity`` builds a persistent source/target
    # covariate shift using feature-only propensity sampling.
    shift_mode: str = "cluster_window_reordering"
    shift_propensity_feature: str = "pca_pc2"
    shift_propensity_strength: float = 1.0
    shift_propensity_direction: str = "low_to_high"

    def shift_summary(self) -> str:
        """Return the effective shift configuration for experiment logs."""
        if self.shift_start_window is None:
            return "disabled"
        if self.shift_mode == "domain_aligned_propensity":
            return (
                "mode=domain_aligned_propensity, "
                f"start=W{self.shift_start_window}, "
                f"feature={self.shift_propensity_feature}, "
                f"strength={self.shift_propensity_strength:g}, "
                f"direction={self.shift_propensity_direction}"
            )
        return (
            "mode=cluster_window_reordering, "
            f"start=W{self.shift_start_window}, "
            f"clusters={self.shift_n_clusters}, "
            f"order={self.shift_cluster_order}, "
            f"block_windows={self.shift_block_windows}"
        )

@dataclass(frozen=True)
class BootstrapWindow:
    """Labeled W0 window used for initialization."""

    window_id: int

    train_features: pd.DataFrame
    train_labels: np.ndarray
    train_row_ids: np.ndarray

    val_features: pd.DataFrame
    val_labels: np.ndarray
    val_row_ids: np.ndarray

    # Complete W0 class set and canonical probability-column order.
    classes: np.ndarray


@dataclass(frozen=True)
class StreamWindow:
    """Feature-only online window supplied for inference."""

    window_id: int
    features: pd.DataFrame
    row_ids: np.ndarray


@dataclass(frozen=True)
class EvalTarget:
    """Window ground-truth labels for evaluation only."""

    window_id: int
    labels: np.ndarray
    row_ids: np.ndarray

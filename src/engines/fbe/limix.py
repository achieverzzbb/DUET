"""LimiX teacher model."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inference.predictor import LimiXPredictor
from src.data.types import BootstrapWindow


class LimiXTeacher:
    """ Infer subsequent windows using the fixed W0 context. """

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        device: str = "cpu",
        seed: int = 2026,
    ) -> None:
        # LimiX predictor.
        self.predictor = LimiXPredictor(
            device=torch.device(device),
            model_path=str(model_path),
            inference_config=str(config_path),
            seed=seed,
        )
        self.support_features: pd.DataFrame | None = None
        self.support_labels: np.ndarray | None = None
        self.classes: np.ndarray | None = None

    def initialize(self, bootstrap: BootstrapWindow) -> None:
        """Initialize the LimiX context from W0 training data."""
        self.support_features = bootstrap.train_features
        self.support_labels = bootstrap.train_labels
        self.classes = bootstrap.classes

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """ Predict the current window and return probabilities in W0 class order. """
        if self.support_features is None or self.support_labels is None:
            raise RuntimeError("LimiX has not been initialized with W0")

        probabilities = self.predictor.predict(
            x_train=self.support_features.to_numpy(),
            y_train=self.support_labels,
            x_test=features.to_numpy(),
            task_type="Classification",
        )
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

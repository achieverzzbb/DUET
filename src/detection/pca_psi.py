"""PCA projection fitted on bootstrap features, followed by per-component PSI."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PCAPSIConfig
from .psi import PSIHistogram
from .types import DriftSignal


class PCAPSIDriftDetector:
    """Compare each window with a fixed bootstrap distribution, without labels.

    Preprocessing follows the existing stream builder's numeric imputation,
    scaling, and categorical one-hot encoding recipe. Unlike stream generation,
    no transform is fitted or updated on incoming windows.
    """

    def __init__(self, config: PCAPSIConfig | None = None) -> None:
        self.config = PCAPSIConfig() if config is None else config
        self.reset()

    @property
    def is_fitted(self) -> bool:
        return self._pca is not None

    @property
    def feature_columns(self) -> tuple:
        return self._feature_columns

    @property
    def component_count(self) -> int:
        return len(self._histograms)

    @property
    def explained_variance_ratio(self) -> tuple[float, ...]:
        if self._pca is None:
            raise RuntimeError("Fit the detector on bootstrap features first")
        return tuple(float(value) for value in self._pca.explained_variance_ratio_)

    def reset(self) -> None:
        """Discard the reference explicitly; adaptation never resets it automatically."""
        self._feature_columns: tuple = ()
        self._numeric_columns: tuple = ()
        self._categorical_columns: tuple = ()
        self._preprocessor: ColumnTransformer | None = None
        self._pca: PCA | None = None
        self._histograms: tuple[PSIHistogram, ...] = ()

    def fit(self, reference_features: pd.DataFrame) -> PCAPSIDriftDetector:
        """Build the reference from W0 features only, not from the full evaluation stream."""
        if self.is_fitted:
            raise RuntimeError("Reset the detector before fitting a new reference")
        self._validate_frame(reference_features)
        if len(reference_features) < 2:
            raise ValueError("PCA requires at least two bootstrap samples")
        numeric_columns = []
        categorical_columns = []
        for column, dtype in reference_features.dtypes.items():
            if pd.api.types.is_complex_dtype(dtype):
                raise ValueError("Complex-valued features are not supported")
            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
                dtype
            ):
                numeric_columns.append(column)
            elif (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(dtype)
            ):
                categorical_columns.append(column)
            else:
                raise ValueError(
                    f"Encode unsupported feature dtype before detection: {dtype}"
                )

        prepared = self._prepare_features(
            reference_features, tuple(numeric_columns), tuple(categorical_columns)
        )
        transformers = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median", keep_empty_features=True
                                ),
                            ),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    [
                        column in numeric_columns
                        for column in reference_features.columns
                    ],
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False, dtype=np.float64
                    ),
                    [
                        column in categorical_columns
                        for column in reference_features.columns
                    ],
                )
            )
        preprocessor = ColumnTransformer(
            transformers, remainder="drop", sparse_threshold=0.0
        )
        encoded = np.asarray(preprocessor.fit_transform(prepared), dtype=np.float64)
        if not np.isfinite(encoded).all():
            raise ValueError("Reference preprocessing produced non-finite values")
        if not np.any(np.var(encoded, axis=0) > 0):
            raise ValueError(
                "The PCA reference must contain at least one varying encoded feature"
            )
        components = self.config.n_components
        if isinstance(components, int) and components > min(encoded.shape):
            raise ValueError(
                "n_components must not exceed the reference sample or encoded feature count"
            )
        pca = PCA(n_components=components, svd_solver="full")
        pca.fit(encoded)
        # Use the online transform path for the reference as well to avoid bin-edge roundoff.
        projected = pca.transform(encoded)
        histograms = tuple(
            PSIHistogram.from_reference(
                projected[:, index], bins=self.config.bins, epsilon=self.config.epsilon
            )
            for index in range(projected.shape[1])
        )

        # Publish fitted state only after every reference component is valid.
        self._feature_columns = tuple(reference_features.columns)
        self._numeric_columns = tuple(numeric_columns)
        self._categorical_columns = tuple(categorical_columns)
        self._preprocessor = preprocessor
        self._pca = pca
        self._histograms = histograms
        return self

    def observe(
        self, features: pd.DataFrame, *, window_id: int | None = None
    ) -> DriftSignal:
        """Return the maximum component PSI and its threshold decision without refitting."""
        if self._pca is None or self._preprocessor is None:
            raise RuntimeError("Fit the detector on bootstrap features first")
        self._validate_frame(features)
        if tuple(features.columns) != self._feature_columns:
            raise ValueError(
                "Feature columns and their order must match the bootstrap reference"
            )
        if window_id is not None:
            if (
                isinstance(window_id, bool)
                or not isinstance(window_id, Integral)
                or window_id < 0
            ):
                raise ValueError("window_id must be a non-negative integer or None")
            window_id = int(window_id)
        prepared = self._prepare_features(
            features, self._numeric_columns, self._categorical_columns
        )
        encoded = np.asarray(self._preprocessor.transform(prepared), dtype=np.float64)
        projected = self._pca.transform(encoded)
        component_scores = tuple(
            histogram.score(projected[:, index])
            for index, histogram in enumerate(self._histograms)
        )
        score = max(component_scores)
        return DriftSignal(
            detected=score >= self.config.threshold,
            score=score,
            threshold=self.config.threshold,
            component_scores=component_scores,
            sample_count=len(features),
            window_id=window_id,
        )

    @staticmethod
    def _validate_frame(features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Detector features must be a pandas DataFrame")
        if features.empty or not features.columns.is_unique:
            raise ValueError(
                "Detector features must have rows and distinct, non-empty columns"
            )

    @staticmethod
    def _prepare_features(
        features: pd.DataFrame, numeric_columns: tuple, categorical_columns: tuple
    ) -> pd.DataFrame:
        prepared = features.copy()
        for column in numeric_columns:
            if pd.api.types.is_complex_dtype(features[column].dtype):
                raise ValueError("Complex-valued features are not supported")
            try:
                values = pd.to_numeric(features[column], errors="raise").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Numeric feature contains non-numeric values: {column}"
                ) from error
            if np.isinf(values).any():
                raise ValueError(
                    "Infinite feature values must be cleaned before detection"
                )
            prepared[column] = values
        for column in categorical_columns:
            values = features[column].astype("string")
            # Prefix real values so the missing marker cannot collide with a category.
            prepared[column] = np.where(
                values.isna(), "missing:", "value:" + values.fillna("")
            )
        prepared.columns = pd.RangeIndex(features.shape[1])
        return prepared

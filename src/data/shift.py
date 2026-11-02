"""Reorder tabular rows into a reproducible clustered distribution shift."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


PROPENSITY_FEATURES = {"pca_pc1", "pca_pc2", "pca_pc3", "pca_pc4", "pca_pc5", "pca_pc6", "robust_radial"}


def build_domain_partition(
    data: pd.DataFrame,
    label_column: str,
    source_size: int,
    seed: int,
    propensity_feature: str = "pca_pc2",
    propensity_strength: float = 1.0,
    propensity_direction: str = "low_to_high",
    ignore_columns: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Return feature-only source and target row positions.

    The partition is a covariate shift: labels are never inspected when
    assigning rows to domains. ``low_to_high`` puts lower propensity scores in
    the source domain and higher scores in the target domain.
    """
    _validate_domain_arguments(
        data=data,
        label_column=label_column,
        source_size=source_size,
        propensity_feature=propensity_feature,
        propensity_strength=propensity_strength,
        propensity_direction=propensity_direction,
        ignore_columns=ignore_columns,
    )
    feature_columns = [
        column
        for column in data.columns
        if column not in {label_column, *ignore_columns}
    ]
    features = data.loc[:, feature_columns].copy()
    scores = _propensity_scores(features, propensity_feature)
    standardized = (scores - np.median(scores)) / (
        np.std(scores) + np.finfo(np.float64).eps
    )
    direction = 1 if propensity_direction == "low_to_high" else -1
    log_weights = -direction * propensity_strength * np.clip(
        standardized,
        -8.0,
        8.0,
    )
    rng = np.random.default_rng(seed)
    # Gumbel-top-k samples without replacement while retaining a smooth,
    # reproducible propensity rather than an abrupt score threshold.
    uniform = np.clip(rng.uniform(size=len(scores)), 1e-12, 1.0 - 1e-12)
    gumbels = -np.log(-np.log(uniform))
    source_positions = np.argpartition(
        log_weights + gumbels,
        -source_size,
    )[-source_size:]
    source_mask = np.zeros(len(data), dtype=bool)
    source_mask[source_positions] = True
    target_positions = np.flatnonzero(~source_mask)
    order_rng = np.random.default_rng(seed)
    return (
        np.asarray(order_rng.permutation(source_positions), dtype=np.int64),
        np.asarray(order_rng.permutation(target_positions), dtype=np.int64),
    )


def apply_cluster_shift(
    data: pd.DataFrame,
    label_column: str,
    window_size: int,
    shift_start_window: int | None,
    seed: int,
    n_clusters: int = 5,
    ignore_columns: tuple[str, ...] = (),
    reference_features: pd.DataFrame | None = None,
    cluster_order: str = "random",
    block_windows: int = 1,
) -> pd.DataFrame:
    """Reorder rows so post-shift windows are concentrated in feature clusters.

    Windows before ``shift_start_window`` remain randomly ordered. Starting at
    the shift window, every complete window is drawn from one feature-space
    cluster whenever possible. Only row order changes; features, labels, and
    original row ids remain untouched.
    """
    _validate_arguments(
        data=data,
        label_column=label_column,
        window_size=window_size,
        shift_start_window=shift_start_window,
        n_clusters=n_clusters,
        ignore_columns=ignore_columns,
        reference_features=reference_features,
        cluster_order=cluster_order,
        block_windows=block_windows,
    )

    rng = np.random.default_rng(seed)
    all_positions = np.arange(len(data))

    if shift_start_window is None:
        return data.iloc[rng.permutation(all_positions)].copy()

    stable_size = (shift_start_window - 1) * window_size
    shuffled_positions = rng.permutation(all_positions)
    stable_positions = shuffled_positions[:stable_size]
    shifted_positions = shuffled_positions[stable_size:]

    shifted_data = data.iloc[shifted_positions]
    shifted_features = shifted_data.drop(
        columns=[label_column, *ignore_columns],
    )
    encoded_features, encoded_reference = _encode_features(
        shifted_features,
        reference_features,
    )

    effective_clusters = min(n_clusters, len(shifted_data))
    if effective_clusters < 2:
        raise ValueError(
            "The shifted stage needs at least two rows to form distinct regions"
        )

    clusterer = MiniBatchKMeans(
        n_clusters=effective_clusters,
        random_state=seed,
        batch_size=min(1024, len(shifted_data)),
        n_init=10,
    )
    cluster_labels = clusterer.fit_predict(encoded_features)

    ordered_cluster_ids = _order_cluster_ids(
        mode=cluster_order,
        cluster_labels=cluster_labels,
        encoded_features=encoded_features,
        encoded_reference=encoded_reference,
        cluster_count=effective_clusters,
        rng=rng,
    )
    cluster_pools: dict[int, list[int]] = {}
    for cluster_id in ordered_cluster_ids:
        local_positions = np.flatnonzero(cluster_labels == cluster_id)
        cluster_pools[int(cluster_id)] = list(rng.permutation(local_positions))

    full_window_clusters = sum(
        len(pool) >= window_size for pool in cluster_pools.values()
    )
    if full_window_clusters < 2:
        raise ValueError(
            "Clustering cannot form complete windows from two regions; reduce "
            "window_size or n_clusters, or increase shifted-stage rows"
        )

    ordered_local_positions = _cycle_cluster_windows(
        cluster_pools=cluster_pools,
        cluster_order=ordered_cluster_ids,
        window_size=window_size,
        block_windows=block_windows,
        rng=rng,
    )
    ordered_shifted_positions = shifted_positions[ordered_local_positions]
    ordered_positions = np.concatenate(
        [stable_positions, ordered_shifted_positions]
    )
    return data.iloc[ordered_positions].copy()


def _encode_features(
    features: pd.DataFrame,
    reference_features: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode shifted and reference features in the same feature space."""
    if features.shape[1] == 0:
        raise ValueError("At least one feature is required to build a shift")

    categorical_columns = features.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    numerical_columns = [
        column for column in features.columns if column not in categorical_columns
    ]

    transformers = []
    if numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numerical", numerical_pipeline, numerical_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    encoded = np.asarray(transformer.fit_transform(features), dtype=np.float64)
    if reference_features is None:
        return encoded, None
    encoded_reference = np.asarray(
        transformer.transform(reference_features),
        dtype=np.float64,
    )
    return encoded, encoded_reference


def _order_cluster_ids(
    mode: str,
    cluster_labels: np.ndarray,
    encoded_features: np.ndarray,
    encoded_reference: np.ndarray | None,
    cluster_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Order clusters randomly or by descending distance from W0 mean."""
    if mode == "random":
        return rng.permutation(cluster_count)

    if encoded_reference is None:
        raise ValueError("distance_from_w0 ordering requires reference_features")

    reference_mean = encoded_reference.mean(axis=0)
    distances = np.empty(cluster_count, dtype=np.float64)
    for cluster_id in range(cluster_count):
        cluster_mean = encoded_features[cluster_labels == cluster_id].mean(axis=0)
        distances[cluster_id] = np.linalg.norm(cluster_mean - reference_mean)
    return np.argsort(-distances, kind="stable")


def _cycle_cluster_windows(
    cluster_pools: dict[int, list[int]],
    cluster_order: np.ndarray,
    window_size: int,
    block_windows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cycle through clusters, emitting persistent single-cluster blocks."""
    ordered: list[int] = []

    while True:
        emitted = False
        for raw_cluster_id in cluster_order:
            cluster_id = int(raw_cluster_id)
            pool = cluster_pools[cluster_id]
            window_count = min(block_windows, len(pool) // window_size)
            if window_count > 0:
                emitted_size = window_count * window_size
                ordered.extend(pool[:emitted_size])
                del pool[:emitted_size]
                emitted = True
        if not emitted:
            break

    leftovers = [position for pool in cluster_pools.values() for position in pool]
    if leftovers:
        ordered.extend(rng.permutation(leftovers).tolist())

    return np.asarray(ordered, dtype=np.int64)


def _propensity_scores(
    features: pd.DataFrame,
    propensity_feature: str,
) -> np.ndarray:
    """Compute robust, feature-only scores for propensity sampling."""
    if features.shape[1] == 0:
        raise ValueError("At least one feature is required to build a shift")
    if features.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).shape[1]:
        raise ValueError(
            "domain_aligned_propensity currently requires numerical features"
        )

    values = features.to_numpy(dtype=np.float64, copy=True)
    # MiniBooNE uses -999 as an all-feature missing-record sentinel. Treat it
    # as missing while building the domain score instead of making it a tail.
    values[values == -999.0] = np.nan
    values = SimpleImputer(strategy="median").fit_transform(values)
    values = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(values)
    values = np.clip(values, -10.0, 10.0)

    if propensity_feature == "robust_radial":
        return np.sqrt(np.mean(np.square(values), axis=1))

    component_index = int(propensity_feature.removeprefix("pca_pc")) - 1
    component_count = component_index + 1
    if component_count > min(values.shape):
        raise ValueError(
            f"{propensity_feature} requires at least {component_count} "
            "samples and numerical features"
        )
    return PCA(
        n_components=component_count,
        random_state=0,
    ).fit_transform(values)[:, component_index]


def _validate_domain_arguments(
    data: pd.DataFrame,
    label_column: str,
    source_size: int,
    propensity_feature: str,
    propensity_strength: float,
    propensity_direction: str,
    ignore_columns: tuple[str, ...],
) -> None:
    if data.empty:
        raise ValueError("Cannot build a stream from an empty dataset")
    if label_column not in data.columns:
        raise ValueError(f"Label column does not exist: {label_column!r}")
    if not 0 < source_size < len(data):
        raise ValueError("source_size must leave non-empty source and target")
    if propensity_feature not in PROPENSITY_FEATURES:
        raise ValueError(
            f"Unknown propensity feature: {propensity_feature!r}"
        )
    if not np.isfinite(propensity_strength) or propensity_strength <= 0:
        raise ValueError("propensity_strength must be finite and positive")
    if propensity_direction not in {"low_to_high", "high_to_low"}:
        raise ValueError(
            "propensity_direction must be 'low_to_high' or 'high_to_low'"
        )
    missing_ignored = set(ignore_columns) - set(data.columns)
    if missing_ignored:
        raise ValueError(
            f"Ignored columns do not exist: {sorted(missing_ignored)}"
        )


def _validate_arguments(
    data: pd.DataFrame,
    label_column: str,
    window_size: int,
    shift_start_window: int | None,
    n_clusters: int,
    ignore_columns: tuple[str, ...],
    reference_features: pd.DataFrame | None,
    cluster_order: str,
    block_windows: int,
) -> None:
    if data.empty:
        raise ValueError("Cannot build a stream from an empty dataset")
    if label_column not in data.columns:
        raise ValueError(f"Label column does not exist: {label_column!r}")
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if cluster_order not in {"random", "distance_from_w0"}:
        raise ValueError(
            "cluster_order must be 'random' or 'distance_from_w0'"
        )
    if block_windows < 1:
        raise ValueError("block_windows must be at least 1")

    missing_ignored = set(ignore_columns) - set(data.columns)
    if missing_ignored:
        raise ValueError(f"Ignored columns do not exist: {sorted(missing_ignored)}")

    if shift_start_window is None:
        return
    if shift_start_window < 1:
        raise ValueError("shift_start_window must start at W1 or later")

    feature_columns = [
        column
        for column in data.columns
        if column not in {label_column, *ignore_columns}
    ]
    if cluster_order == "distance_from_w0":
        if reference_features is None or reference_features.empty:
            raise ValueError(
                "distance_from_w0 ordering requires non-empty reference_features"
            )
        if list(reference_features.columns) != feature_columns:
            raise ValueError(
                "reference_features columns must match the shifted features"
            )

    stable_size = (shift_start_window - 1) * window_size
    if stable_size >= len(data):
        raise ValueError(
            f"W{shift_start_window} exceeds the stream range: pre-shift stage "
            f"requires {stable_size} rows, but only {len(data)} are available"
        )

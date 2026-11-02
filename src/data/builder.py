""" Build W0 and subsequent streaming windows from a complete static dataset. """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .shift import apply_cluster_shift, build_domain_partition
from .stream import DataStream
from .types import (
    BootstrapWindow,
    DatasetSpec,
    EvalTarget,
    StreamConfig,
    StreamWindow,
)


class StreamBuilder:
    """ Build a reproducible stream from dataset and experiment settings. """

    def __init__(
        self,
        dataset_spec: DatasetSpec,
        stream_config: StreamConfig,
    ) -> None:
        self.dataset_spec = dataset_spec
        self.stream_config = stream_config

    def build(self) -> DataStream:
        """ Build labeled W0, feature-only online windows, and separate evaluation labels. """
        data = self._load_data()
        self._validate_config(data)

        if (
            self.stream_config.shift_start_window is not None
            and self.stream_config.shift_mode == "domain_aligned_propensity"
        ):
            bootstrap, ordered_data = self._build_domain_aligned_stream(data)
        else:
            bootstrap, remaining_data = self._build_bootstrap(data)
            bootstrap_features = pd.concat(
                [bootstrap.train_features, bootstrap.val_features],
                axis=0,
            )
            ordered_data = apply_cluster_shift(
                data=remaining_data,
                label_column=self.dataset_spec.label_column,
                ignore_columns=self.dataset_spec.ignore_columns,
                window_size=self.stream_config.window_size,
                shift_start_window=self.stream_config.shift_start_window,
                seed=self.stream_config.seed,
                n_clusters=self.stream_config.shift_n_clusters,
                reference_features=bootstrap_features,
                cluster_order=self.stream_config.shift_cluster_order,
                block_windows=self.stream_config.shift_block_windows,
            )
        stream_windows, eval_targets = self._build_stream_windows(ordered_data)

        stream = DataStream(
            bootstrap_window=bootstrap,
            stream_windows=stream_windows,
            eval_target=eval_targets,
        )
        self._validate_row_partition(data, bootstrap, stream_windows)
        return stream

    def _build_domain_aligned_stream(
        self,
        data: pd.DataFrame,
    ) -> tuple[BootstrapWindow, pd.DataFrame]:
        """Build W0/stable windows from source and shifted windows from target."""
        config = self.stream_config
        if config.shift_start_window is None:
            raise RuntimeError("A domain-aligned shift requires a change point")

        stable_size = (config.shift_start_window - 1) * config.window_size
        source_size = config.bootstrap_size + stable_size
        source_positions, target_positions = build_domain_partition(
            data=data,
            label_column=self.dataset_spec.label_column,
            ignore_columns=self.dataset_spec.ignore_columns,
            source_size=source_size,
            seed=config.seed,
            propensity_feature=config.shift_propensity_feature,
            propensity_strength=config.shift_propensity_strength,
            propensity_direction=config.shift_propensity_direction,
        )
        source_data = data.iloc[source_positions]
        target_data = data.iloc[target_positions]
        bootstrap, stable_data = self._build_bootstrap(source_data)
        if len(stable_data) != stable_size:
            raise RuntimeError("Source domain does not match the stable stage size")
        return bootstrap, pd.concat([stable_data, target_data], axis=0)

    def _load_data(self) -> pd.DataFrame:
        """ Read the CSV and validate dataset settings and the class space. """
        if not self.dataset_spec.path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_spec.path}"
            )

        data = pd.read_csv(self.dataset_spec.path)

        if self.dataset_spec.label_column not in data.columns:
            raise ValueError(f"Label column not found: {self.dataset_spec.label_column}")

        missing_ignored = set(self.dataset_spec.ignore_columns) - set(data.columns)
        if missing_ignored:
            raise ValueError(f"Ignored columns not found: {sorted(missing_ignored)}")

        if self.dataset_spec.label_column in self.dataset_spec.ignore_columns:
            raise ValueError("The label column must not also be ignored")

        if data.columns.duplicated().any():
            duplicated = data.columns[data.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate dataset columns: {duplicated}")

        if data[self.dataset_spec.label_column].isna().any():
            raise ValueError("The label column contains missing values")

        class_count = data[self.dataset_spec.label_column].nunique()
        if self.dataset_spec.problem_type == "binary" and class_count != 2:
            raise ValueError(
                f"A binary dataset must contain two classes; found {class_count} classes"
            )
        if self.dataset_spec.problem_type == "multiclass" and class_count < 3:
            raise ValueError(
                f"A multiclass dataset must contain at least three classes; found {class_count} classes"
            )

        if not self._feature_columns(data):
            raise ValueError("The dataset contains no feature columns for training")

        result = data.copy()
        # Assign each sample a unique row_id.
        result.index = pd.RangeIndex(len(result), name="row_id")
        return result

    def _build_bootstrap(
        self,
        data: pd.DataFrame,
    ) -> tuple[BootstrapWindow, pd.DataFrame]:
        """ Construct stratified W0 training and validation splits. """

        # Extract sample IDs and labels.
        indices = data.index.to_numpy()
        labels = data[self.dataset_spec.label_column].to_numpy()

        w0_ids, remaining_ids = train_test_split(
            indices,
            train_size=self.stream_config.bootstrap_size,
            stratify=labels,
            random_state=self.stream_config.seed,
        )

        # Separate W0 from the remaining data.
        w0_data = data.loc[w0_ids]
        remaining_data = data.loc[remaining_ids]

        train_ids, val_ids = train_test_split(
            w0_ids,
            test_size=self.stream_config.bootstrap_val_ratio,
            stratify=w0_data[self.dataset_spec.label_column],
            random_state=self.stream_config.seed,
        )

        feature_columns = self._feature_columns(data)
        classes = np.sort(w0_data[self.dataset_spec.label_column].unique())

        bootstrap = BootstrapWindow(
            window_id=0,
            train_features=data.loc[train_ids, feature_columns].copy(),
            train_labels=data.loc[
                train_ids, self.dataset_spec.label_column
            ].to_numpy(copy=True),
            train_row_ids=np.asarray(train_ids).copy(),
            val_features=data.loc[val_ids, feature_columns].copy(),
            val_labels=data.loc[
                val_ids, self.dataset_spec.label_column
            ].to_numpy(copy=True),
            val_row_ids=np.asarray(val_ids).copy(),
            classes=classes,
        )

        self._validate_bootstrap(bootstrap)

        return bootstrap, remaining_data

    def _feature_columns(
        self,
        data: pd.DataFrame,
    ) -> list[str]:
        excluded = {
            self.dataset_spec.label_column,
            *self.dataset_spec.ignore_columns,
        }   
        return [
            column
            for column in data.columns
            if column not in excluded
        ]

    def _build_stream_windows(
        self,
        ordered_data: pd.DataFrame,
    ) -> tuple[list[StreamWindow], dict[int, EvalTarget]]:
        """ Split W1...WT and store features separately from ground-truth labels. """
        feature_columns = self._feature_columns(ordered_data)
        stream_windows: list[StreamWindow] = []
        eval_targets: dict[int, EvalTarget] = {}

        for window_id, start in enumerate(
            range(0, len(ordered_data), self.stream_config.window_size),
            start=1,
        ):
            batch = ordered_data.iloc[
                start : start + self.stream_config.window_size
            ]
            row_ids = batch.index.to_numpy(copy=True)

            stream_windows.append(
                StreamWindow(
                    window_id=window_id,
                    features=batch.loc[:, feature_columns].copy(),
                    row_ids=row_ids,
                )
            )
            eval_targets[window_id] = EvalTarget(
                window_id=window_id,
                labels=batch[self.dataset_spec.label_column].to_numpy(copy=True),
                row_ids=row_ids.copy(),
            )

        return stream_windows, eval_targets

    def _validate_config(self, data: pd.DataFrame) -> None:
        """Check that W0 can contain separate training and validation data."""
        config = self.stream_config
        class_count = data[self.dataset_spec.label_column].nunique()

        if config.bootstrap_size <= 0:
            raise ValueError("bootstrap_size must be positive")
        if config.bootstrap_size >= len(data):
            raise ValueError("bootstrap_size must be smaller than the dataset size")
        if config.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0 < config.bootstrap_val_ratio < 1:
            raise ValueError("bootstrap_val_ratio must be strictly between zero and one")

        val_size = int(np.ceil(config.bootstrap_size * config.bootstrap_val_ratio))
        train_size = config.bootstrap_size - val_size
        if train_size < class_count or val_size < class_count:
            raise ValueError(
                "Both W0 splits must contain at least one sample from each class"
            )

        class_counts = data[self.dataset_spec.label_column].value_counts()
        if (class_counts < 2).any():
            rare_classes = class_counts[class_counts < 2].index.tolist()
            raise ValueError(f"Insufficient samples for stratified splitting in classes: {rare_classes}")

        if config.shift_start_window is not None:
            if config.shift_start_window < 1:
                raise ValueError("shift_start_window must be at least one")
            source_size = config.bootstrap_size + (
                config.shift_start_window - 1
            ) * config.window_size
            if source_size >= len(data):
                raise ValueError(
                    f"W{config.shift_start_window} exceeds the stream range: "
                    f"source requires {source_size} rows, dataset has {len(data)}"
                )
        if config.shift_mode not in {
            "cluster_window_reordering",
            "domain_aligned_propensity",
        }:
            raise ValueError(
                "shift_mode must be 'cluster_window_reordering' or "
                "'domain_aligned_propensity'"
            )

    def _validate_bootstrap(self, bootstrap: BootstrapWindow) -> None:
        """Ensure both W0 splits cover the complete class space."""
        expected = set(bootstrap.classes.tolist())
        train_classes = set(np.unique(bootstrap.train_labels).tolist())
        val_classes = set(np.unique(bootstrap.val_labels).tolist())

        if train_classes != expected:
            raise ValueError("The W0 training split does not cover all classes")
        if val_classes != expected:
            raise ValueError("The W0 validation split does not cover all classes")

    @staticmethod
    def _validate_row_partition(
        data: pd.DataFrame,
        bootstrap: BootstrapWindow,
        stream_windows: list[StreamWindow],
    ) -> None:
        """Ensure each sample appears exactly once to prevent leakage across windows."""
        partitions = [
            bootstrap.train_row_ids,
            bootstrap.val_row_ids,
            *(window.row_ids for window in stream_windows),
        ]
        all_row_ids = np.concatenate(partitions)

        if len(all_row_ids) != len(data):
            raise ValueError("The stream contains missing or extra samples")
        if len(np.unique(all_row_ids)) != len(all_row_ids):
            raise ValueError("W0 and online windows contain overlapping samples")
        if set(all_row_ids.tolist()) != set(data.index.tolist()):
            raise ValueError("The stream sample set differs from the original dataset")

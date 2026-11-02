"""Explicit runtime configuration shared by experiment entry points.

The normal command-line entry points may still construct this configuration
from ``.env`` for backwards compatibility.  Dedicated experiments can build it
directly, which keeps their independent variables visible in source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.types import StreamConfig
from src.detection import PCAPSIConfig
from src.engines.tbe.autogluon import AUTOGLUON_MODEL_KEYS


@dataclass(frozen=True)
class AutoGluonRuntimeConfig:
    """Training/search settings that must be explicit in controlled runs."""

    model_keys: tuple[str, ...] = AUTOGLUON_MODEL_KEYS
    fit_strategy: str = "parallel"
    cpus_per_model: int = 4
    presets: str = "medium_quality"
    num_bag_folds: int = 0
    num_bag_sets: int = 1
    num_stack_levels: int = 0
    use_bag_holdout: bool = False
    dynamic_stacking: bool = False

    def validate(self, training_num_cpus: int) -> None:
        if not self.model_keys:
            raise ValueError("At least one AutoGluon model is required")
        if self.fit_strategy not in {"parallel", "sequential"}:
            raise ValueError("fit_strategy must be parallel or sequential")
        if self.cpus_per_model <= 0:
            raise ValueError("cpus_per_model must be positive")
        if not self.presets.strip():
            raise ValueError("presets must not be empty")
        if self.num_bag_folds == 1 or self.num_bag_folds < 0:
            raise ValueError("num_bag_folds must be 0 or at least 2")
        if self.num_bag_sets <= 0:
            raise ValueError("num_bag_sets must be positive")
        if self.num_stack_levels < 0:
            raise ValueError("num_stack_levels must be non-negative")
        if self.use_bag_holdout and self.num_bag_folds == 0:
            raise ValueError(
                "use_bag_holdout requires num_bag_folds to be enabled"
            )
        required = len(self.model_keys) * self.cpus_per_model
        if self.fit_strategy == "parallel" and training_num_cpus < required:
            raise ValueError(
                "Parallel search requires at least "
                f"{required} training CPUs, got {training_num_cpus}"
            )


@dataclass(frozen=True)
class ExperimentRuntimeConfig:
    """Model and stream settings for one reproducible experiment run.

    ``autogluon`` is used by standalone baselines.  DUET has a separate
    student configuration because its tree engine is on the online critical
    path and should avoid expensive bagging ensembles.
    """

    dataset_name: str
    stream: StreamConfig
    train_time_limit: int
    window_interval_seconds: float = 1.0
    accuracy_tolerance: float = 0.01
    distill_shift_weight: float = 0.6
    distill_window_count: int = 2
    max_w0_accuracy_drop: float = 0.02
    # None preserves the original one-shard-per-GPU behavior. Dedicated
    # scalability experiments may set a fixed upper bound so shard boundaries
    # remain independent of GPU count.
    limix_max_shard_size: int | None = None
    limix_warmup_sample_count: int = 2_048
    autogluon: AutoGluonRuntimeConfig = field(
        default_factory=AutoGluonRuntimeConfig
    )
    run_student_comparison: bool = False
    # Optional override for the DUET student.  Keeping this after the legacy
    # fields preserves positional construction for older callers.
    duet_autogluon: AutoGluonRuntimeConfig | None = None
    drift_detector: PCAPSIConfig = field(default_factory=PCAPSIConfig)

    def student_autogluon_config(self) -> AutoGluonRuntimeConfig:
        """Return DUET's student config, falling back for old callers."""

        return self.duet_autogluon or self.autogluon

    def validate(self, training_num_cpus: int) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")
        if self.train_time_limit <= 0:
            raise ValueError("train_time_limit must be positive")
        if self.window_interval_seconds < 0:
            raise ValueError("window_interval_seconds must be non-negative")
        if not 0 <= self.accuracy_tolerance <= 1:
            raise ValueError("accuracy_tolerance must be between 0 and 1")
        if not 0 < self.distill_shift_weight < 1:
            raise ValueError("distill_shift_weight must be between 0 and 1")
        if self.distill_window_count <= 0:
            raise ValueError("distill_window_count must be positive")
        if not 0 <= self.max_w0_accuracy_drop <= 1:
            raise ValueError("max_w0_accuracy_drop must be between 0 and 1")
        if (
            self.limix_max_shard_size is not None
            and self.limix_max_shard_size <= 0
        ):
            raise ValueError("limix_max_shard_size must be positive")
        if self.limix_warmup_sample_count <= 0:
            raise ValueError("limix_warmup_sample_count must be positive")
        self.autogluon.validate(training_num_cpus)
        self.student_autogluon_config().validate(training_num_cpus)

"""AutoGluon tree-model student. """

import os
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from src.data.types import BootstrapWindow, ProblemType
from src.system.cpu_resources import get_cpu_resource_config


AUTOGLUON_FIT_STRATEGY = os.getenv(
    "DUET_AUTOGLUON_FIT_STRATEGY",
    "parallel",
).strip().lower()
if AUTOGLUON_FIT_STRATEGY not in {"sequential", "parallel"}:
    raise ValueError(
        "DUET_AUTOGLUON_FIT_STRATEGY must be sequential or parallel"
    )

AUTOGLUON_MODEL_KEYS = ("GBM", "CAT", "XGB", "RF", "XT")
AUTOGLUON_CPUS_PER_MODEL = int(
    os.getenv("DUET_AUTOGLUON_CPUS_PER_MODEL", "4")
)
if AUTOGLUON_CPUS_PER_MODEL <= 0:
    raise ValueError("DUET_AUTOGLUON_CPUS_PER_MODEL must be positive")

AUTOGLUON_REQUIRED_PARALLEL_CPUS = (
    len(AUTOGLUON_MODEL_KEYS) * AUTOGLUON_CPUS_PER_MODEL
)


def tree_model_hyperparameters(
    model_keys: tuple[str, ...] = AUTOGLUON_MODEL_KEYS,
    cpus_per_model: int = AUTOGLUON_CPUS_PER_MODEL,
) -> dict[str, dict]:
    """Assign an equal CPU budget to each specified tree-model candidate."""
    return {
        model_key: {
            "ag_args_fit": {
                "num_cpus": cpus_per_model,
                "num_gpus": 0,
            },
        }
        for model_key in model_keys
    }


def bagging_fit_arguments(
    *,
    num_bag_folds: int,
    num_bag_sets: int,
    num_stack_levels: int,
    use_bag_holdout: bool,
    dynamic_stacking: bool,
) -> dict[str, int | bool]:
    """Return explicit bagging options without changing legacy non-bagged fits."""
    if num_bag_folds == 0:
        return {}
    return {
        "num_bag_folds": num_bag_folds,
        "num_bag_sets": num_bag_sets,
        "num_stack_levels": num_stack_levels,
        "use_bag_holdout": use_bag_holdout,
        "dynamic_stacking": dynamic_stacking,
    }


def shutdown_ray() -> None:
    """Release the AutoGluon Ray session after training."""
    try:
        import ray
    except ImportError:
        return

    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        # Ray cleanup failures must not affect saved models.
        return


class AutoGluonStudent:
    """ Train on W0 and serve subsequent windows on CPUs. """

    def __init__(
        self,
        model_path: Path,
        problem_type: ProblemType,
        time_limit: int = 300,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        model_keys: tuple[str, ...] | None = None,
        fit_strategy: str | None = None,
        cpus_per_model: int | None = None,
        presets: str = "medium_quality",
        num_bag_folds: int = 0,
        num_bag_sets: int = 1,
        num_stack_levels: int = 0,
        use_bag_holdout: bool = False,
        dynamic_stacking: bool = False,
    ) -> None:
        self.model_path = model_path
        self.problem_type = problem_type
        self.time_limit = time_limit
        self.model_keys = (
            AUTOGLUON_MODEL_KEYS
            if model_keys is None
            else tuple(model_keys)
        )
        if not self.model_keys:
            raise ValueError("At least one AutoGluon model must be configured")
        unsupported_models = set(self.model_keys) - set(AUTOGLUON_MODEL_KEYS)
        if unsupported_models:
            raise ValueError(
                "Unsupported AutoGluon model keys: "
                f"{sorted(unsupported_models)}"
            )
        self.fit_strategy = (
            AUTOGLUON_FIT_STRATEGY
            if fit_strategy is None
            else fit_strategy.strip().lower()
        )
        if self.fit_strategy not in {"sequential", "parallel"}:
            raise ValueError("fit_strategy must be sequential or parallel")
        self.cpus_per_model = (
            AUTOGLUON_CPUS_PER_MODEL
            if cpus_per_model is None
            else cpus_per_model
        )
        if self.cpus_per_model <= 0:
            raise ValueError("cpus_per_model must be positive")
        self.presets = presets.strip()
        self.num_bag_folds = num_bag_folds
        self.num_bag_sets = num_bag_sets
        self.num_stack_levels = num_stack_levels
        self.use_bag_holdout = use_bag_holdout
        self.dynamic_stacking = dynamic_stacking
        if not self.presets:
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
        self.num_cpus = (
            get_cpu_resource_config().training_num_cpus
            if num_cpus is None
            else num_cpus
        )
        # Keep LimiX resident on GPUs while AutoGluon searches on CPUs.
        self.num_gpus = 0 if num_gpus is None else num_gpus
        required_parallel_cpus = (
            len(self.model_keys) * self.cpus_per_model
        )
        if (
            self.fit_strategy == "parallel"
            and self.num_cpus < required_parallel_cpus
        ):
            raise ValueError(
                "Parallel AutoGluon search requires at least "
                f"{required_parallel_cpus} CPUs "
                f"({len(self.model_keys)} models x "
                f"{self.cpus_per_model} CPUs), but only "
                f"{self.num_cpus} were configured"
            )
        self.predictor: TabularPredictor | None = None
        self.model_name: str | None = None
        self.classes: np.ndarray | None = None

    def fit(self, bootstrap: BootstrapWindow) -> None:
        """Fit on W0 training data and select the best tree using W0 validation data. """
        label_column = "__target__"

        train_data = bootstrap.train_features.copy()
        train_data[label_column] = bootstrap.train_labels

        val_data = bootstrap.val_features.copy()
        val_data[label_column] = bootstrap.val_labels

        # Train the configured tree-model candidates.
        predictor = TabularPredictor(
            label=label_column,
            problem_type=self.problem_type,
            eval_metric="accuracy",
            path=str(self.model_path),
            verbosity=0,
        ).fit(
            train_data=train_data,
            tuning_data=val_data,
            presets=self.presets,
            time_limit=self.time_limit,
            hyperparameters=tree_model_hyperparameters(
                self.model_keys,
                self.cpus_per_model,
            ),
            fit_weighted_ensemble=False,
            num_gpus=self.num_gpus,
            num_cpus=self.num_cpus,
            fit_strategy=self.fit_strategy,
            **bagging_fit_arguments(
                num_bag_folds=self.num_bag_folds,
                num_bag_sets=self.num_bag_sets,
                num_stack_levels=self.num_stack_levels,
                use_bag_holdout=self.use_bag_holdout,
                dynamic_stacking=self.dynamic_stacking,
            ),
        )

        # Use saved deployable model metadata consistently across inference workers.
        predictor.save(silent=True)
        predictor = TabularPredictor.load(path=str(self.model_path))

        # Store the selected model.
        self.predictor = predictor
        self.model_name = predictor.model_best
        self.classes = bootstrap.classes.copy()

    def load_trained(
        self,
        model_name: str,
        classes: np.ndarray,
    ) -> None:
        """Load a model trained by an isolated background process."""
        predictor = TabularPredictor.load(path=str(self.model_path))
        if model_name not in predictor.model_names(can_infer=True):
            raise ValueError(
                f"AutoGluon model {model_name!r} is absent from {self.model_path}"
            )
        self.predictor = predictor
        self.model_name = model_name
        self.classes = np.asarray(classes).copy()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """ Return window predictions as probabilities in W0 class order. """
        if self.predictor is None or self.model_name is None or self.classes is None:
            raise RuntimeError("AutoGluon has not been trained on W0")

        probabilities = self.predictor.predict_proba(
            features,
            model=self.model_name,
        )
        aligned = [
            probabilities[column].to_numpy()
            for class_label in self.classes
            for column in probabilities.columns
            if column == class_label
        ]
        if len(aligned) != len(self.classes):
            raise ValueError("AutoGluon probability columns do not match W0 classes")
        return np.column_stack(aligned).astype(np.float64)

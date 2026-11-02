"""Distill LimiX predictions into an AutoGluon tree model."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.data.types import BootstrapWindow, ProblemType
from src.engines.tbe.autogluon import (
    AUTOGLUON_CPUS_PER_MODEL,
    AUTOGLUON_FIT_STRATEGY,
    AUTOGLUON_MODEL_KEYS,
    AutoGluonStudent,
    bagging_fit_arguments,
    tree_model_hyperparameters,
)
from src.system.cpu_resources import get_cpu_resource_config


LABEL_COLUMN = "__distill_label__"
WEIGHT_COLUMN = "__teacher_probability__"
AUTOGLUON_NUM_GPUS = 0


def _training_options(
    num_cpus: int | None,
    model_keys: tuple[str, ...] | None,
    fit_strategy: str | None,
    cpus_per_model: int | None,
) -> tuple[int, tuple[str, ...], str, int]:
    """Resolve optional settings without freezing CPU resources at import."""
    return (
        get_cpu_resource_config().training_num_cpus
        if num_cpus is None
        else num_cpus,
        AUTOGLUON_MODEL_KEYS if model_keys is None else tuple(model_keys),
        AUTOGLUON_FIT_STRATEGY if fit_strategy is None else fit_strategy,
        AUTOGLUON_CPUS_PER_MODEL
        if cpus_per_model is None
        else cpus_per_model,
    )

def _normalize_probabilities(
    features: pd.DataFrame,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.array(
        teacher_probabilities,
        dtype=np.float64,
        copy=True,
    )
    classes = np.asarray(classes)

    if probabilities.shape != (len(features), len(classes)):
        raise ValueError("Teacher probabilities must have shape (samples, classes)")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Teacher probabilities must be finite and non-negative")
    if (probabilities.sum(axis=1) <= 0).any():
        raise ValueError("Each teacher probability row must have a positive sum")
    if LABEL_COLUMN in features or WEIGHT_COLUMN in features:
        raise ValueError("Feature names conflict with reserved distillation columns")

    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, classes


def build_hard_label_data(
    features: pd.DataFrame,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    """ Convert LimiX probabilities to argmax pseudo-labels. 
    Args:
        features (pd.DataFrame): Features
        teacher_probabilities (np.ndarray): Probabilities produced by the teacher.
        classes (np.ndarray): Class labels
    """
    probabilities, classes = _normalize_probabilities(
        features,
        teacher_probabilities,
        classes,
    )
    data = features.reset_index(drop=True).copy()
    data[LABEL_COLUMN] = classes[np.argmax(probabilities, axis=1)]
    return data


def build_soft_label_data(
    features: pd.DataFrame,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """ Expand LimiX probabilities into weighted AutoGluon training rows. """
    probabilities, classes = _normalize_probabilities(
        features,
        teacher_probabilities,
        classes,
    )
    if sample_weights is None:
        sample_weights = np.ones(len(features), dtype=np.float64)
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if sample_weights.shape != (len(features),):
            raise ValueError("Sample weights must have shape (samples,)")
        if not np.isfinite(sample_weights).all() or (sample_weights <= 0).any():
            raise ValueError("Sample weights must be finite and positive")

    expanded = []
    for class_index, class_label in enumerate(classes):
        class_data = features.reset_index(drop=True).copy()
        class_data[LABEL_COLUMN] = class_label
        class_data[WEIGHT_COLUMN] = np.clip(
            probabilities[:, class_index] * sample_weights,
            1e-8,
            None,
        )
        expanded.append(class_data)

    return pd.concat(expanded, ignore_index=True)


def _weighted_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    return log_loss(y_true, y_pred, sample_weight=sample_weight)


SOFT_LOG_LOSS = make_scorer(
    "soft_log_loss",
    _weighted_log_loss,
    optimum=0,
    greater_is_better=False,
    needs_proba=True,
)


def soft_cross_entropy(
    teacher_probabilities: np.ndarray,
    student_probabilities: np.ndarray,
) -> float:
    """Compute soft cross-entropy between student and teacher probabilities."""
    teacher = np.array(teacher_probabilities, dtype=np.float64, copy=True)
    student = np.array(student_probabilities, dtype=np.float64, copy=True)
    if teacher.shape != student.shape:
        raise ValueError("Teacher and student probability shapes do not match")

    teacher /= teacher.sum(axis=1, keepdims=True)
    student = np.clip(student, 1e-15, None)
    student /= student.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(teacher * np.log(student), axis=1)))


def labels_to_probabilities(
    labels: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Convert W0 true labels to one-hot probability targets."""
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    probabilities = (labels[:, None] == classes[None, :]).astype(np.float64)
    if (probabilities.sum(axis=1) != 1).any():
        raise ValueError("W0 labels contain classes outside the canonical class set")
    return probabilities


def distill_hard_student(
    features: pd.DataFrame,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
    model_path: Path,
    problem_type: ProblemType,
    time_limit: int = 300,
    validation_ratio: float = 0.2,
    seed: int = 2026,
    num_cpus: int | None = None,
    model_keys: tuple[str, ...] | None = None,
    fit_strategy: str | None = None,
    cpus_per_model: int | None = None,
    presets: str = "medium_quality",
    num_bag_folds: int = 0,
    num_bag_sets: int = 1,
    num_stack_levels: int = 0,
    use_bag_holdout: bool = False,
    dynamic_stacking: bool = False,
) -> AutoGluonStudent:
    """ Train the student using LimiX argmax pseudo-labels. """
    data = build_hard_label_data(features, teacher_probabilities, classes)
    pseudo_labels = data[LABEL_COLUMN]
    class_counts = pseudo_labels.value_counts()

    if len(class_counts) != len(classes) or (class_counts < 2).any():
        raise ValueError("Hard pseudo-labels do not cover all classes; collect more windows")

    train_data, val_data = train_test_split(
        data,
        test_size=validation_ratio,
        stratify=pseudo_labels,
        random_state=seed,
    )
    # AutoGluon retrains and selects models using accuracy.
    num_cpus, model_keys, fit_strategy, cpus_per_model = _training_options(
        num_cpus, model_keys, fit_strategy, cpus_per_model
    )
    predictor = TabularPredictor(
        label=LABEL_COLUMN,
        problem_type=problem_type,
        eval_metric="accuracy",
        path=str(model_path),
        verbosity=0,
    ).fit(
        train_data=train_data,
        tuning_data=val_data,
        presets=presets,
        time_limit=time_limit,
        hyperparameters=tree_model_hyperparameters(
            model_keys, cpus_per_model
        ),
        fit_weighted_ensemble=False,
        num_cpus=num_cpus,
        num_gpus=AUTOGLUON_NUM_GPUS,
        fit_strategy=fit_strategy,
        **bagging_fit_arguments(
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            num_stack_levels=num_stack_levels,
            use_bag_holdout=use_bag_holdout,
            dynamic_stacking=dynamic_stacking,
        ),
    )

    predictor.save(silent=True)
    predictor = TabularPredictor.load(path=str(model_path))

    # Attach the trained predictor to the student.
    student = AutoGluonStudent(
        model_path,
        problem_type,
        time_limit,
        num_cpus=num_cpus,
        model_keys=model_keys,
        fit_strategy=fit_strategy,
        cpus_per_model=cpus_per_model,
        presets=presets,
        num_bag_folds=num_bag_folds,
        num_bag_sets=num_bag_sets,
        num_stack_levels=num_stack_levels,
        use_bag_holdout=use_bag_holdout,
        dynamic_stacking=dynamic_stacking,
    )
    student.predictor = predictor
    student.model_name = predictor.model_best
    student.classes = np.asarray(classes).copy()
    return student


def distill_soft_student(
    features: pd.DataFrame,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
    model_path: Path,
    problem_type: ProblemType,
    time_limit: int = 300,
    validation_ratio: float = 0.2,
    seed: int = 2026,
    num_cpus: int | None = None,
    model_keys: tuple[str, ...] | None = None,
    fit_strategy: str | None = None,
    cpus_per_model: int | None = None,
    presets: str = "medium_quality",
    num_bag_folds: int = 0,
    num_bag_sets: int = 1,
    num_stack_levels: int = 0,
    use_bag_holdout: bool = False,
    dynamic_stacking: bool = False,
) -> AutoGluonStudent:
    """ Train the student using the complete LimiX probability distribution. """
    probabilities, classes = _normalize_probabilities(
        features,
        teacher_probabilities,
        classes,
    )
    train_indices, val_indices = train_test_split(
        np.arange(len(features)),
        test_size=validation_ratio,
        random_state=seed,
    )
    train_data = build_soft_label_data(
        features.iloc[train_indices],
        probabilities[train_indices],
        classes,
    )
    val_data = build_soft_label_data(
        features.iloc[val_indices],
        probabilities[val_indices],
        classes,
    )

    num_cpus, model_keys, fit_strategy, cpus_per_model = _training_options(
        num_cpus, model_keys, fit_strategy, cpus_per_model
    )
    predictor = TabularPredictor(
        label=LABEL_COLUMN,
        problem_type=problem_type,
        eval_metric=SOFT_LOG_LOSS,
        sample_weight=WEIGHT_COLUMN,
        weight_evaluation=True,
        path=str(model_path),
        verbosity=0,
    ).fit(
        train_data=train_data,
        tuning_data=val_data,
        presets=presets,
        time_limit=time_limit,
        hyperparameters=tree_model_hyperparameters(
            model_keys, cpus_per_model
        ),
        fit_weighted_ensemble=False,
        num_cpus=num_cpus,
        num_gpus=AUTOGLUON_NUM_GPUS,
        fit_strategy=fit_strategy,
        calibrate=False,
        **bagging_fit_arguments(
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            num_stack_levels=num_stack_levels,
            use_bag_holdout=use_bag_holdout,
            dynamic_stacking=dynamic_stacking,
        ),
    )

    predictor.save(silent=True)
    predictor = TabularPredictor.load(path=str(model_path))

    student = AutoGluonStudent(
        model_path,
        problem_type,
        time_limit,
        num_cpus=num_cpus,
        model_keys=model_keys,
        fit_strategy=fit_strategy,
        cpus_per_model=cpus_per_model,
        presets=presets,
        num_bag_folds=num_bag_folds,
        num_bag_sets=num_bag_sets,
        num_stack_levels=num_stack_levels,
        use_bag_holdout=use_bag_holdout,
        dynamic_stacking=dynamic_stacking,
    )
    student.predictor = predictor
    student.classes = classes.copy()

    validation_features = features.iloc[val_indices]
    model_scores = {}
    for model_name in predictor.model_names(can_infer=True):
        student.model_name = model_name
        model_scores[model_name] = soft_cross_entropy(
            probabilities[val_indices],
            student.predict_proba(validation_features),
        )

    student.model_name = min(model_scores, key=model_scores.get)
    return student


def distill_replay_student(
    bootstrap: BootstrapWindow,
    shift_features: pd.DataFrame,
    teacher_probabilities: np.ndarray,
    model_path: Path,
    problem_type: ProblemType,
    shift_weight: float = 0.5,
    time_limit: int = 300,
    validation_ratio: float = 0.2,
    seed: int = 2026,
    num_cpus: int | None = None,
    model_keys: tuple[str, ...] | None = None,
    fit_strategy: str | None = None,
    cpus_per_model: int | None = None,
    presets: str = "medium_quality",
    num_bag_folds: int = 0,
    num_bag_sets: int = 1,
    num_stack_levels: int = 0,
    use_bag_holdout: bool = False,
    dynamic_stacking: bool = False,
) -> AutoGluonStudent:
    """Train on W0 true labels and LimiX probabilities from a shift window."""
    if not 0 < shift_weight < 1:
        raise ValueError("shift_weight must be strictly between zero and one")

    shift_probabilities, classes = _normalize_probabilities(
        shift_features,
        teacher_probabilities,
        bootstrap.classes,
    )
    shift_train_indices, shift_val_indices = train_test_split(
        np.arange(len(shift_features)),
        test_size=validation_ratio,
        random_state=seed,
    )

    replay_train_probabilities = labels_to_probabilities(
        bootstrap.train_labels,
        classes,
    )
    replay_val_probabilities = labels_to_probabilities(
        bootstrap.val_labels,
        classes,
    )

    replay_weight = 1.0 - shift_weight
    train_total = len(bootstrap.train_features) + len(shift_train_indices)
    val_total = len(bootstrap.val_features) + len(shift_val_indices)

    replay_train_weights = np.full(
        len(bootstrap.train_features),
        replay_weight * train_total / len(bootstrap.train_features),
    )
    shift_train_weights = np.full(
        len(shift_train_indices),
        shift_weight * train_total / len(shift_train_indices),
    )
    replay_val_weights = np.full(
        len(bootstrap.val_features),
        replay_weight * val_total / len(bootstrap.val_features),
    )
    shift_val_weights = np.full(
        len(shift_val_indices),
        shift_weight * val_total / len(shift_val_indices),
    )

    train_data = pd.concat(
        [
            build_soft_label_data(
                bootstrap.train_features,
                replay_train_probabilities,
                classes,
                replay_train_weights,
            ),
            build_soft_label_data(
                shift_features.iloc[shift_train_indices],
                shift_probabilities[shift_train_indices],
                classes,
                shift_train_weights,
            ),
        ],
        ignore_index=True,
    )
    val_data = pd.concat(
        [
            build_soft_label_data(
                bootstrap.val_features,
                replay_val_probabilities,
                classes,
                replay_val_weights,
            ),
            build_soft_label_data(
                shift_features.iloc[shift_val_indices],
                shift_probabilities[shift_val_indices],
                classes,
                shift_val_weights,
            ),
        ],
        ignore_index=True,
    )

    num_cpus, model_keys, fit_strategy, cpus_per_model = _training_options(
        num_cpus, model_keys, fit_strategy, cpus_per_model
    )
    predictor = TabularPredictor(
        label=LABEL_COLUMN,
        problem_type=problem_type,
        eval_metric=SOFT_LOG_LOSS,
        sample_weight=WEIGHT_COLUMN,
        weight_evaluation=True,
        path=str(model_path),
        verbosity=0,
    ).fit(
        train_data=train_data,
        tuning_data=val_data,
        presets=presets,
        time_limit=time_limit,
        hyperparameters=tree_model_hyperparameters(
            model_keys, cpus_per_model
        ),
        fit_weighted_ensemble=False,
        num_cpus=num_cpus,
        num_gpus=AUTOGLUON_NUM_GPUS,
        fit_strategy=fit_strategy,
        calibrate=False,
        **bagging_fit_arguments(
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            num_stack_levels=num_stack_levels,
            use_bag_holdout=use_bag_holdout,
            dynamic_stacking=dynamic_stacking,
        ),
    )

    predictor.save(silent=True)
    predictor = TabularPredictor.load(path=str(model_path))

    student = AutoGluonStudent(
        model_path,
        problem_type,
        time_limit,
        num_cpus=num_cpus,
        model_keys=model_keys,
        fit_strategy=fit_strategy,
        cpus_per_model=cpus_per_model,
        presets=presets,
        num_bag_folds=num_bag_folds,
        num_bag_sets=num_bag_sets,
        num_stack_levels=num_stack_levels,
        use_bag_holdout=use_bag_holdout,
        dynamic_stacking=dynamic_stacking,
    )
    student.predictor = predictor
    student.classes = classes.copy()

    model_scores = {}
    for model_name in predictor.model_names(can_infer=True):
        student.model_name = model_name
        replay_loss = soft_cross_entropy(
            replay_val_probabilities,
            student.predict_proba(bootstrap.val_features),
        )
        shift_loss = soft_cross_entropy(
            shift_probabilities[shift_val_indices],
            student.predict_proba(shift_features.iloc[shift_val_indices]),
        )
        model_scores[model_name] = (
            replay_weight * replay_loss
            + shift_weight * shift_loss
        )

    student.model_name = min(model_scores, key=model_scores.get)
    return student

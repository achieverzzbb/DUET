"""Persistent CPU data-parallel inference pool for AutoGluon models."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_worker_predictor = None
_worker_model_name: str | None = None
_worker_classes: np.ndarray | None = None
_worker_id: int | None = None

_pending_predictor = None
_pending_model_name: str | None = None
_pending_classes: np.ndarray | None = None


def _load_predictor(model_path: str, model_name: str):
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(path=model_path)
    if model_name not in predictor.model_names():
        raise ValueError(
            f"Model {model_name!r} is not present in {model_path}"
        )
    predictor.persist(models=[model_name], with_ancestors=True)
    return predictor


def _initialize_worker(
    worker_id: int,
    cpu_ids: tuple[int, ...],
    thread_count: int,
    model_path: str | None,
    model_name: str | None,
    classes: np.ndarray | None,
) -> None:
    """Create a long-lived worker and optionally load its first model."""

    global _worker_predictor
    global _worker_model_name
    global _worker_classes
    global _worker_id

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))

    # Import once during worker startup so later reloads avoid import startup.
    from autogluon.tabular import TabularPredictor

    del TabularPredictor
    _worker_id = worker_id
    if model_path is not None:
        if model_name is None or classes is None:
            raise ValueError("Initial model metadata is incomplete")
        _worker_predictor = _load_predictor(model_path, model_name)
        _worker_model_name = model_name
        _worker_classes = np.asarray(classes)


def _worker_status() -> tuple[int, bool]:
    if _worker_id is None:
        raise RuntimeError("AutoGluon worker is not initialized")
    return _worker_id, _worker_predictor is not None


def _prepare_reload_worker(
    model_path: str,
    model_name: str,
    classes: np.ndarray,
    warmup_features: pd.DataFrame | None,
) -> int:
    """Load and warm a candidate without replacing the active model."""

    global _pending_predictor
    global _pending_model_name
    global _pending_classes

    if _worker_id is None:
        raise RuntimeError("AutoGluon worker is not initialized")

    predictor = _load_predictor(model_path, model_name)
    pending_classes = np.asarray(classes)
    if warmup_features is not None and len(warmup_features) > 0:
        probabilities = predictor.predict_proba(
            warmup_features,
            model=model_name,
        )
        missing = set(pending_classes) - set(probabilities.columns)
        if missing:
            raise ValueError(
                f"Candidate probability columns are missing: {sorted(missing)}"
            )

    _pending_predictor = predictor
    _pending_model_name = model_name
    _pending_classes = pending_classes
    return _worker_id


def _activate_reload_worker() -> int:
    global _worker_predictor
    global _worker_model_name
    global _worker_classes
    global _pending_predictor
    global _pending_model_name
    global _pending_classes

    if _worker_id is None or _pending_predictor is None:
        raise RuntimeError("No prepared AutoGluon model is available")

    _worker_predictor = _pending_predictor
    _worker_model_name = _pending_model_name
    _worker_classes = _pending_classes
    _pending_predictor = None
    _pending_model_name = None
    _pending_classes = None
    return _worker_id


def _discard_reload_worker() -> int:
    global _pending_predictor
    global _pending_model_name
    global _pending_classes

    if _worker_id is None:
        raise RuntimeError("AutoGluon worker is not initialized")
    _pending_predictor = None
    _pending_model_name = None
    _pending_classes = None
    return _worker_id


def _predict_worker(
    start: int,
    features: pd.DataFrame,
) -> tuple[int, np.ndarray]:
    if (
        _worker_predictor is None
        or _worker_model_name is None
        or _worker_classes is None
    ):
        raise RuntimeError("AutoGluon worker has no active model")

    probabilities = _worker_predictor.predict_proba(
        features,
        model=_worker_model_name,
    )
    missing = set(_worker_classes) - set(probabilities.columns)
    if missing:
        raise ValueError(
            f"AutoGluon probability columns are missing: {sorted(missing)}"
        )
    aligned = [
        probabilities[class_label].to_numpy()
        for class_label in _worker_classes
    ]
    return start, np.column_stack(aligned).astype(np.float64)


class AutoGluonPool:
    """Manage persistent, CPU-pinned AutoGluon inference workers."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_name: str | None = None,
        classes: np.ndarray | None = None,
        cpu_sets: Iterable[Iterable[int]] = ((0,),),
        threads_per_worker: int = 1,
    ) -> None:
        self.model_path = None if model_path is None else Path(model_path)
        self.model_name = model_name
        self.classes = None if classes is None else np.asarray(classes)
        self.cpu_sets = tuple(tuple(cpu_ids) for cpu_ids in cpu_sets)
        self.threads_per_worker = threads_per_worker
        self.executors: list[ProcessPoolExecutor] = []
        self.initialized = False
        self.model_ready = model_path is not None

        if not self.cpu_sets or any(not cpu_ids for cpu_ids in self.cpu_sets):
            raise ValueError("At least one non-empty CPU worker set is required")
        if threads_per_worker <= 0:
            raise ValueError("threads_per_worker must be positive")
        if self.model_ready and (model_name is None or classes is None):
            raise ValueError("Initial model metadata is incomplete")

    def initialize(self) -> None:
        """Start workers once for the lifetime of the stream."""

        if self.initialized:
            raise RuntimeError("AutoGluon inference pool is already initialized")

        context = mp.get_context("spawn")
        try:
            for worker_id, cpu_ids in enumerate(self.cpu_sets):
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(
                        worker_id,
                        cpu_ids,
                        self.threads_per_worker,
                        None if self.model_path is None else str(self.model_path),
                        self.model_name,
                        self.classes,
                    ),
                )
                self.executors.append(executor)

            status_tasks = [
                executor.submit(_worker_status)
                for executor in self.executors
            ]
            statuses = [task.result() for task in status_tasks]
        except Exception:
            self.close()
            raise

        self.initialized = True
        worker_ids = [worker_id for worker_id, _ in statuses]
        if any(ready != self.model_ready for _, ready in statuses):
            self.close()
            raise RuntimeError("AutoGluon workers reported inconsistent state")
        print(
            "AutoGluon persistent CPU inference pool initialized, "
            f"workers={worker_ids}, model_ready={self.model_ready}"
        )

    def reload_model(
        self,
        model_path: str | Path,
        model_name: str,
        classes: np.ndarray,
        warmup_features: pd.DataFrame | None = None,
    ) -> None:
        """Prepare on every worker, then atomically activate a new model."""

        if not self.initialized:
            raise RuntimeError("AutoGluon inference pool is not initialized")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"AutoGluon model path does not exist: {path}")
        class_array = np.asarray(classes)
        if class_array.ndim != 1 or len(class_array) == 0:
            raise ValueError("classes must be a non-empty one-dimensional array")

        try:
            prepare_tasks = [
                executor.submit(
                    _prepare_reload_worker,
                    str(path),
                    model_name,
                    class_array,
                    warmup_features,
                )
                for executor in self.executors
            ]
            prepared_ids = [task.result() for task in prepare_tasks]
        except Exception:
            discard_tasks = [
                executor.submit(_discard_reload_worker)
                for executor in self.executors
            ]
            for task in discard_tasks:
                try:
                    task.result()
                except Exception:
                    pass
            raise

        activation_tasks = [
            executor.submit(_activate_reload_worker)
            for executor in self.executors
        ]
        activated_ids = [task.result() for task in activation_tasks]
        if sorted(prepared_ids) != sorted(activated_ids):
            raise RuntimeError("AutoGluon workers activated inconsistent models")

        self.model_path = path
        self.model_name = model_name
        self.classes = class_array.copy()
        self.model_ready = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Split a batch across workers and restore original row order."""

        if not self.initialized or not self.model_ready or self.classes is None:
            raise RuntimeError("AutoGluon inference pool has no active model")

        sample_count = len(features)
        class_count = len(self.classes)
        if sample_count == 0:
            return np.empty((0, class_count), dtype=np.float64)

        worker_count = min(len(self.executors), sample_count)
        boundaries = np.linspace(
            0,
            sample_count,
            worker_count + 1,
            dtype=np.int64,
        )
        tasks = [
            executor.submit(
                _predict_worker,
                int(start),
                features.iloc[int(start) : int(stop)].copy(),
            )
            for executor, start, stop in zip(
                self.executors,
                boundaries[:-1],
                boundaries[1:],
            )
        ]

        probabilities = np.empty(
            (sample_count, class_count),
            dtype=np.float64,
        )
        for task, expected_start, expected_stop in zip(
            tasks,
            boundaries[:-1],
            boundaries[1:],
        ):
            start, shard_probabilities = task.result()
            expected_shape = (
                int(expected_stop - expected_start),
                class_count,
            )
            if (
                start != int(expected_start)
                or shard_probabilities.shape != expected_shape
            ):
                raise ValueError("AutoGluon shard probability shape is invalid")
            probabilities[start : start + len(shard_probabilities)] = (
                shard_probabilities
            )
        return probabilities

    def close(self) -> None:
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self.executors.clear()
        self.initialized = False
        self.model_ready = False

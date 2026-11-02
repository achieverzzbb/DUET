"""LimiX multi-GPU data-parallel inference."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable
import os

import numpy as np
import pandas as pd
import torch

from src.data.types import BootstrapWindow
from src.engines.fbe.limix import LimiXTeacher

_worker_model: LimiXTeacher | None = None
_worker_gpu_id: int | None = None

# Large enough to exercise realistic preprocessing, CUDA allocation and model
# kernels without paying the cost of a complete formal stream window.
DEFAULT_WARMUP_SAMPLE_COUNT = 2_048
BALANCED_POOL_WARMUP_REPETITIONS = 4

def _initialize_worker(
    gpu_id: int,
    model_path: str,
    config_path: str,
    bootstrap: BootstrapWindow,
    seed: int,
    cpu_ids: tuple[int, ...],
    warmup_sample_count: int,
) -> None:
    """ Initialize one LimiX model on the assigned GPU. """

    global _worker_model, _worker_gpu_id

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this worker process")

    torch.cuda.set_device(gpu_id)
    torch.set_num_threads(1)
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))

    _worker_gpu_id = gpu_id
    _worker_model = LimiXTeacher(
        model_path = Path(model_path),
        config_path = Path(config_path),
        device = f"cuda:{gpu_id}",
        seed = seed
    )

    # Initialize the model using the bootstrap context.
    _worker_model.initialize(bootstrap)

    # Warm up.
    warmup_source = bootstrap.val_features
    if len(warmup_source) == 0:
        warmup_source = bootstrap.train_features
    if len(warmup_source) == 0:
        raise ValueError("LimiX warmup requires at least one bootstrap row")
    warmup_positions = (
        np.arange(warmup_sample_count, dtype=np.int64)
        % len(warmup_source)
    )
    warmup_features = (
        warmup_source.iloc[warmup_positions].reset_index(drop=True).copy()
    )
    _worker_model.predict_proba(warmup_features)
    torch.cuda.synchronize(gpu_id)


def _worker_status() -> int:
    """Return the GPU ID assigned to this worker."""

    if _worker_model is None or _worker_gpu_id is None:
        raise RuntimeError("The LimiX worker has not been initialized")

    return _worker_gpu_id


def _predict_worker(
    start: int,
    features: pd.DataFrame
) -> tuple[int, np.ndarray]:
    """ Predict one data shard on the current GPU. 
    start: Starting position of the shard in the current batch.
    """
    if _worker_model is None or _worker_gpu_id is None:
        raise RuntimeError("The LimiX worker has not been initialized")

    probabilities = _worker_model.predict_proba(features)
    torch.cuda.synchronize(_worker_gpu_id)

    return start, probabilities


class LimiXPool:
    """ Manage LimiX replicas across GPUs. """
    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        gpu_ids: Iterable[int] = range(8),
        seed: int = 2026,
        cpu_ids: Iterable[int] | None = None,
        max_shard_size: int | None = None,
        warmup_sample_count: int = DEFAULT_WARMUP_SAMPLE_COUNT,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.gpu_ids = tuple(gpu_ids)
        self.seed = seed
        self.cpu_ids = tuple(cpu_ids or ())
        self.max_shard_size = max_shard_size
        self.warmup_sample_count = warmup_sample_count

        self.executors: list[ProcessPoolExecutor] = []
        self.classes: np.ndarray | None = None
        self.initialized = False

        if not self.gpu_ids:
            raise ValueError("At least one GPU must be specified")

        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("GPU IDs must be distinct")

        if max_shard_size is not None and max_shard_size <= 0:
            raise ValueError("max_shard_size must be positive")
        if warmup_sample_count <= 0:
            raise ValueError("warmup_sample_count must be positive")

    @property
    def effective_warmup_sample_count(self) -> int:
        """Rows sent to each GPU while warming the real prediction path."""
        if self.max_shard_size is None:
            return self.warmup_sample_count
        return max(self.warmup_sample_count, self.max_shard_size)

    def initialize(self, bootstrap: BootstrapWindow) -> None:
        """ Load each GPU model with the same bootstrap context. """
        if self.initialized:
            raise RuntimeError("The LimiX pool is already initialized")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this environment")

        visible_gpu_count = torch.cuda.device_count()
        invalid_gpu_ids = [
            gpu_id
            for gpu_id in self.gpu_ids
            if gpu_id < 0 or gpu_id >= visible_gpu_count
        ]
        if invalid_gpu_ids:
            raise ValueError(
                f"Unavailable GPU IDs: {invalid_gpu_ids}, "
                f"Visible GPU count: {visible_gpu_count}"
            )

        self.classes = np.asarray(bootstrap.classes)
        context = mp.get_context("spawn")
        worker_cpu_ids = [
            tuple(self.cpu_ids[index::len(self.gpu_ids)])
            for index in range(len(self.gpu_ids))
        ]

        try:
            for gpu_id in self.gpu_ids:
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(
                        gpu_id,
                        str(self.model_path),
                        str(self.config_path),
                        bootstrap,
                        self.seed,
                        worker_cpu_ids[len(self.executors)],
                        self.effective_warmup_sample_count,
                    ),
                )
                self.executors.append(executor)

            ready_tasks = [
                executor.submit(_worker_status)
                for executor in self.executors
            ]
            ready_gpu_ids = [
                task.result()
                for task in ready_tasks
            ]
        except Exception:
            self.close()
            raise

        self.initialized = True

        # The initializer warms CUDA/model kernels inside each worker, while
        # this pass also warms DataFrame slicing, IPC, result collection, and
        # fixed-microbatch scheduling before online timing begins.
        try:
            self._warmup_balanced_pool(bootstrap)
        except Exception:
            self.close()
            raise

        print(
            "LimiX multi-GPU pool initialized, "
            f"GPU={ready_gpu_ids}"
        )

        print(
            "LimiX sharding: "
            f"max_shard_size={self.max_shard_size or 'gpu_balanced'}, "
            "effective_warmup_samples_per_gpu="
            f"{self.effective_warmup_sample_count}"
        )

    def _warmup_balanced_pool(self, bootstrap: BootstrapWindow) -> None:
        """Warm the complete balanced prediction path outside measurements."""
        source = bootstrap.val_features
        if len(source) == 0:
            source = bootstrap.train_features
        if len(source) == 0:
            raise ValueError("LimiX warmup requires at least one bootstrap row")

        samples_per_gpu = self.effective_warmup_sample_count
        sample_count = samples_per_gpu * len(self.executors)
        positions = np.arange(sample_count, dtype=np.int64) % len(source)
        features = source.iloc[positions].reset_index(drop=True).copy()

        for _ in range(BALANCED_POOL_WARMUP_REPETITIONS):
            self.predict_proba(features)

        print(
            "LimiX end-to-end warmup completed, "
            f"rounds={BALANCED_POOL_WARMUP_REPETITIONS}, "
            f"samples_per_gpu={samples_per_gpu}"
        )

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """ Split input and merge prediction probabilities from all GPUs. """

        if not self.initialized or self.classes is None:
            raise RuntimeError("The LimiX pool has not been initialized")

        sample_count = len(features)
        class_count = len(self.classes)

        if sample_count == 0:
            return np.empty(
                (0, class_count),
                dtype=np.float64,
            )

        if self.max_shard_size is not None:
            return self._predict_fixed_microbatches(
                features,
                class_count,
            )

        active_worker_count = min(
            len(self.executors),
            sample_count,
        )

        boundaries = np.linspace(
            0,
            sample_count,
            active_worker_count + 1,
            dtype=np.int64,
        )

        tasks = []
        for executor, start, stop in zip(
            self.executors,
            boundaries[:-1],
            boundaries[1:],
        ):
            shard = features.iloc[int(start) : int(stop)].copy()

            # Predict the shards.
            tasks.append(
                executor.submit(
                    _predict_worker,
                    int(start),
                    shard,
                )
            )

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
            shard_probabilities = np.asarray(
                shard_probabilities,
                dtype=np.float64,
            )

            expected_shape = (
                int(expected_stop - expected_start),
                class_count,
            )
            if (
                start != int(expected_start)
                or shard_probabilities.shape != expected_shape
            ):
                raise ValueError(
                    "LimiX shard probability shape mismatch, "
                    f"expected {expected_shape}, "
                    f"got {shard_probabilities.shape}"
                )
            # Merge predicted probabilities.
            stop = start + len(shard_probabilities)
            probabilities[start:stop] = shard_probabilities

        return probabilities

    def _predict_fixed_microbatches(
        self,
        features: pd.DataFrame,
        class_count: int,
    ) -> np.ndarray:
        """Run GPU-independent microbatches in bounded parallel rounds."""
        if self.max_shard_size is None:
            raise RuntimeError("Fixed microbatching is not configured")

        sample_count = len(features)
        shards = [
            (start, min(sample_count, start + self.max_shard_size))
            for start in range(0, sample_count, self.max_shard_size)
        ]
        active_worker_count = min(len(self.executors), len(shards))
        probabilities = np.empty(
            (sample_count, class_count),
            dtype=np.float64,
        )

        # A round submits at most one shard to each GPU. Waiting between
        # rounds bounds DataFrame copies and process-queue memory usage.
        for round_start in range(0, len(shards), active_worker_count):
            round_shards = shards[
                round_start : round_start + active_worker_count
            ]
            tasks = []
            for worker_index, (start, stop) in enumerate(round_shards):
                shard = features.iloc[start:stop].copy()
                tasks.append(
                    self.executors[worker_index].submit(
                        _predict_worker,
                        start,
                        shard,
                    )
                )

            for task, (expected_start, expected_stop) in zip(
                tasks,
                round_shards,
            ):
                start, shard_probabilities = task.result()
                shard_probabilities = np.asarray(
                    shard_probabilities,
                    dtype=np.float64,
                )
                expected_shape = (
                    expected_stop - expected_start,
                    class_count,
                )
                if (
                    start != expected_start
                    or shard_probabilities.shape != expected_shape
                ):
                    raise ValueError(
                        "LimiX microbatch output mismatch: "
                        f"expected start={expected_start}, "
                        f"shape={expected_shape}; got start={start}, "
                        f"shape={shard_probabilities.shape}"
                )
                probabilities[start:expected_stop] = shard_probabilities

        return probabilities

    def close(self) -> None:
        """Close all LimiX worker processes."""

        for executor in self.executors:
            executor.shutdown(
                wait=True,
                cancel_futures=True,
            )

        self.executors.clear()
        self.initialized = False

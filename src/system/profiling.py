"""Measure offline accuracy and timing profiles for scheduling."""

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OfflineModelProfile:
    """Offline estimates for one model in a given data state."""

    accuracy: float
    startup_seconds: float
    throughput: float


@dataclass(frozen=True)
class SchedulingProfile:
    """Scheduling profiles for one dataset."""

    dataset_name: str
    calibration_seed: int
    accuracy_quantile: float
    limix: OfflineModelProfile
    stable_student: OfflineModelProfile
    shifted_student: OfflineModelProfile
    future_student: OfflineModelProfile
    distillation_seconds: float
    mixed_overhead_seconds: float = 0.0


def conservative_accuracy(
    window_accuracies: list[float],
    quantile: float = 0.1,
) -> float:
    """Use a lower quantile of window accuracies as a conservative estimate."""

    if not window_accuracies:
        raise ValueError("At least one window accuracy is required")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")

    values = np.asarray(window_accuracies, dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Window accuracies must be finite and between zero and one")
    return float(np.quantile(values, quantile))


def measure_window_accuracies(
    predict_proba: Callable[[pd.DataFrame], np.ndarray],
    windows: list,
    labels_by_window: dict[int, np.ndarray],
    classes: np.ndarray,
    progress_label: str | None = None,
) -> list[float]:
    """Measure per-window accuracy on offline calibration windows."""

    accuracies = []
    total = len(windows)
    for index, window in enumerate(windows, start=1):
        if progress_label is not None:
            print(
                f"  {progress_label}: W{window.window_id} "
                f"({index}/{total}, samples={len(window.features)})",
                flush=True,
            )
        probabilities = np.asarray(
            predict_proba(window.features),
            dtype=np.float64,
        )
        labels = np.asarray(labels_by_window[window.window_id])
        if probabilities.shape != (len(labels), len(classes)):
            raise ValueError("Model probability shape does not match the calibration window")
        predictions = classes[probabilities.argmax(axis=1)]
        accuracies.append(float(np.mean(predictions == labels)))
        if progress_label is not None:
            print(
                f"  {progress_label}: W{window.window_id} complete, "
                f"accuracy={accuracies[-1]:.4f}",
                flush=True,
            )
    return accuracies


def measure_latency_profile(
    predict_proba: Callable[[pd.DataFrame], np.ndarray],
    features: pd.DataFrame,
    batch_sizes: tuple[int, ...],
    repetitions: int = 3,
    synchronize: Callable[[], None] | None = None,
    progress_label: str | None = None,
) -> tuple[float, float]:
    """Fit inference time T(n)=startup+n/throughput."""

    sizes = sorted(set(batch_sizes))
    if not sizes or sizes[0] <= 0 or sizes[-1] > len(features):
        raise ValueError("batch_sizes must be within the available sample range")
    if len(sizes) < 2:
        raise ValueError("At least two distinct batch sizes are required")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    predict_proba(features.iloc[: sizes[0]])
    if synchronize is not None:
        synchronize()

    median_seconds = []
    for size in sizes:
        measurements = []
        batch = features.iloc[:size]
        for repetition in range(1, repetitions + 1):
            if progress_label is not None:
                print(
                    f"  {progress_label}: samples={size}, "
                    f"repeat={repetition}/{repetitions}",
                    flush=True,
                )
            if synchronize is not None:
                synchronize()
            start = perf_counter()
            predict_proba(batch)
            if synchronize is not None:
                synchronize()
            measurements.append(perf_counter() - start)
        median_seconds.append(float(np.median(measurements)))

    design = np.column_stack(
        [np.ones(len(sizes), dtype=np.float64), np.asarray(sizes)]
    )
    startup_seconds, seconds_per_sample = np.linalg.lstsq(
        design,
        np.asarray(median_seconds),
        rcond=None,
    )[0]

    startup_seconds = max(0.0, float(startup_seconds))
    if seconds_per_sample <= 0:
        largest_time = median_seconds[-1]
        seconds_per_sample = max(
            (largest_time - startup_seconds) / sizes[-1],
            largest_time / sizes[-1],
            1e-12,
        )

    return startup_seconds, 1.0 / float(seconds_per_sample)


def measure_mixed_overhead(
    limix_predict: Callable[[pd.DataFrame], np.ndarray],
    student_predict: Callable[[pd.DataFrame], np.ndarray],
    features: pd.DataFrame,
    batch_size: int,
    limix_profile: OfflineModelProfile,
    student_profile: OfflineModelProfile,
    limix_ratios: tuple[float, ...] = (0.5, 0.75, 0.9),
    repetitions: int = 3,
) -> float:
    """Measure dual-engine parallel overhead beyond the independent latency models."""
    if batch_size <= 1 or batch_size > len(features):
        raise ValueError("batch_size must be between two and the available sample count")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not limix_ratios or any(
        ratio <= 0 or ratio >= 1 for ratio in limix_ratios
    ):
        raise ValueError("All limix_ratios must be strictly between zero and one")

    batch = features.iloc[:batch_size]
    overheads = []

    def run_once(ratio: float, executor: ThreadPoolExecutor) -> float:
        limix_count = min(
            batch_size - 1,
            max(1, round(batch_size * ratio)),
        )
        start = perf_counter()
        student_task = executor.submit(
            student_predict,
            batch.iloc[limix_count:].copy(),
        )
        limix_predict(batch.iloc[:limix_count].copy())
        student_task.result()
        elapsed = perf_counter() - start

        expected = max(
            limix_profile.startup_seconds
            + limix_count / limix_profile.throughput,
            student_profile.startup_seconds
            + (batch_size - limix_count) / student_profile.throughput,
        )
        return max(0.0, elapsed - expected)

    with ThreadPoolExecutor(max_workers=1) as executor:
        run_once(limix_ratios[0], executor)
        for ratio in limix_ratios:
            for _ in range(repetitions):
                overheads.append(run_once(ratio, executor))

    return float(np.median(overheads))


def save_profile(profile: SchedulingProfile, path: Path) -> None:
    """Save offline scheduling profiles."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(profile), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_profile(
    path: Path,
    dataset_name: str | None = None,
) -> SchedulingProfile:
    """Load and validate offline scheduling profiles."""

    if not path.exists():
        raise FileNotFoundError(
            f"Scheduling profile not found: {path}; run src/profile_models.py first"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    profile = SchedulingProfile(
        dataset_name=data["dataset_name"],
        calibration_seed=int(data["calibration_seed"]),
        accuracy_quantile=float(data["accuracy_quantile"]),
        limix=OfflineModelProfile(**data["limix"]),
        stable_student=OfflineModelProfile(**data["stable_student"]),
        shifted_student=OfflineModelProfile(**data["shifted_student"]),
        future_student=OfflineModelProfile(**data["future_student"]),
        distillation_seconds=float(data["distillation_seconds"]),
        mixed_overhead_seconds=float(
            data.get("mixed_overhead_seconds", 0.0)
        ),
    )
    _validate_profile(profile)

    if dataset_name is not None and profile.dataset_name != dataset_name:
        raise ValueError(
            f"Scheduling profile belongs to {profile.dataset_name}; current dataset is {dataset_name}"
        )
    return profile


def _validate_profile(profile: SchedulingProfile) -> None:
    if not profile.dataset_name:
        raise ValueError("dataset_name must not be empty")
    if not 0 <= profile.accuracy_quantile <= 1:
        raise ValueError("accuracy_quantile must be between zero and one")
    if (
        not np.isfinite(profile.distillation_seconds)
        or profile.distillation_seconds <= 0
    ):
        raise ValueError("distillation_seconds must be finite and positive")
    if (
        not np.isfinite(profile.mixed_overhead_seconds)
        or profile.mixed_overhead_seconds < 0
    ):
        raise ValueError("mixed_overhead_seconds must be finite and non-negative")
    profiles = (
        profile.limix,
        profile.stable_student,
        profile.shifted_student,
        profile.future_student,
    )
    for model in profiles:
        if not 0 <= model.accuracy <= 1:
            raise ValueError("Model accuracy must be between zero and one")
        if not np.isfinite(model.startup_seconds) or model.startup_seconds < 0:
            raise ValueError("Model startup time must be finite and non-negative")
        if not np.isfinite(model.throughput) or model.throughput <= 0:
            raise ValueError("Model throughput must be finite and positive")

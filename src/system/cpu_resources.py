"""CPU affinity and priority controls for DUET runtime components."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import threading
from typing import Iterator


@dataclass(frozen=True)
class CPUResourceConfig:
    inference_cpu_sets: tuple[tuple[int, ...], ...]
    training_cpu_ids: tuple[int, ...]
    limix_cpu_ids: tuple[int, ...]
    inference_threads_per_worker: int
    training_num_cpus: int
    background_nice_increment: int


_background_state = threading.local()
_CPU_UNIVERSE_ENV = "_DUET_CPU_UNIVERSE"


def parse_cpu_ids(expression: str) -> tuple[int, ...]:
    """Parse a CPU list such as ``0-3,8,12-15``."""

    cpu_ids: list[int] = []
    for part in expression.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2:
                raise ValueError(f"Invalid CPU range: {part}")
            start, end = (int(value) for value in bounds)
            if start < 0 or end < start:
                raise ValueError(f"Invalid CPU range: {part}")
            cpu_ids.extend(range(start, end + 1))
        else:
            cpu_id = int(part)
            if cpu_id < 0:
                raise ValueError(f"Invalid CPU id: {cpu_id}")
            cpu_ids.append(cpu_id)

    if len(cpu_ids) != len(set(cpu_ids)):
        raise ValueError(f"Duplicate CPU ids in: {expression}")
    return tuple(cpu_ids)


def parse_cpu_sets(expression: str) -> tuple[tuple[int, ...], ...]:
    """Parse semicolon-separated worker CPU sets."""

    cpu_sets = tuple(
        parse_cpu_ids(part)
        for part in expression.split(";")
        if part.strip()
    )
    if not cpu_sets or any(not cpu_set for cpu_set in cpu_sets):
        raise ValueError("At least one non-empty inference CPU set is required")
    return cpu_sets


def _available_cpu_ids() -> tuple[int, ...]:
    inherited_cpu_ids = os.getenv(_CPU_UNIVERSE_ENV)
    if inherited_cpu_ids:
        return parse_cpu_ids(inherited_cpu_ids)
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.cpu_count() or 1))


def _linux_core_groups(
    available_cpu_ids: tuple[int, ...],
) -> dict[int, list[tuple[int, ...]]]:
    """Return SMT sibling groups keyed by physical package."""

    topology_root = Path("/sys/devices/system/cpu")
    grouped: dict[tuple[int, int], list[int]] = {}
    try:
        for cpu_id in available_cpu_ids:
            topology = topology_root / f"cpu{cpu_id}" / "topology"
            package_id = int(
                (topology / "physical_package_id").read_text().strip()
            )
            core_id = int((topology / "core_id").read_text().strip())
            grouped.setdefault((package_id, core_id), []).append(cpu_id)
    except (OSError, ValueError):
        return {}

    packages: dict[int, list[tuple[int, ...]]] = {}
    for (package_id, core_id), sibling_ids in sorted(grouped.items()):
        del core_id
        packages.setdefault(package_id, []).append(
            tuple(sorted(sibling_ids))
        )
    return packages


def _flatten(groups: list[tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(cpu_id for group in groups for cpu_id in group)


def _default_partition(
    available_cpu_ids: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]:
    packages = _linux_core_groups(available_cpu_ids)
    if packages and all(len(groups) >= 3 for groups in packages.values()):
        inference_sets: list[tuple[int, ...]] = []
        training_groups: list[tuple[int, ...]] = []
        limix_groups: list[tuple[int, ...]] = []

        for groups in packages.values():
            core_count = len(groups)
            inference_count = max(1, core_count // 3)
            limix_count = max(1, core_count // 6)
            if inference_count + limix_count >= core_count:
                inference_count = 1
                limix_count = 1

            inference_sets.append(_flatten(groups[:inference_count]))
            training_groups.extend(
                groups[inference_count : core_count - limix_count]
            )
            limix_groups.extend(groups[core_count - limix_count :])

        return (
            tuple(inference_sets),
            _flatten(training_groups),
            _flatten(limix_groups),
        )

    if len(available_cpu_ids) < 3:
        raise RuntimeError("DUET CPU isolation requires at least three CPUs")

    inference_count = max(1, len(available_cpu_ids) // 3)
    limix_count = max(1, len(available_cpu_ids) // 6)
    if inference_count + limix_count >= len(available_cpu_ids):
        inference_count = 1
        limix_count = 1

    inference_ids = available_cpu_ids[:inference_count]
    training_ids = available_cpu_ids[
        inference_count : len(available_cpu_ids) - limix_count
    ]
    limix_ids = available_cpu_ids[len(available_cpu_ids) - limix_count :]
    midpoint = max(1, len(inference_ids) // 2)
    inference_sets = (
        (inference_ids[:midpoint], inference_ids[midpoint:])
        if inference_ids[midpoint:]
        else (inference_ids,)
    )
    return inference_sets, training_ids, limix_ids


@lru_cache(maxsize=1)
def get_cpu_resource_config() -> CPUResourceConfig:
    """Resolve disjoint CPU sets, with optional environment overrides."""

    available = _available_cpu_ids()
    # Spawned workers inherit the caller's current affinity. Preserve the
    # original CPU universe so imports inside a pinned child validate against
    # the CPUs available before DUET divided them into component-specific sets.
    os.environ.setdefault(
        _CPU_UNIVERSE_ENV,
        ",".join(map(str, available)),
    )
    inference_sets, training_ids, limix_ids = _default_partition(available)

    inference_value = os.getenv("DUET_AUTOGLUON_INFERENCE_CPU_SETS")
    training_value = os.getenv("DUET_AUTOGLUON_TRAIN_CPU_IDS")
    limix_value = os.getenv("DUET_LIMIX_CPU_IDS")
    if inference_value:
        inference_sets = parse_cpu_sets(inference_value)
    if training_value:
        training_ids = parse_cpu_ids(training_value)
    if limix_value:
        limix_ids = parse_cpu_ids(limix_value)

    inference_ids = {
        cpu_id for cpu_set in inference_sets for cpu_id in cpu_set
    }
    inference_id_count = sum(len(cpu_set) for cpu_set in inference_sets)
    if inference_id_count != len(inference_ids):
        raise ValueError("Inference worker CPU sets must be disjoint")
    training_set = set(training_ids)
    limix_set = set(limix_ids)
    available_set = set(available)
    configured = inference_ids | training_set | limix_set
    unavailable = configured - available_set
    if unavailable:
        raise ValueError(
            f"Configured CPU ids are unavailable: {sorted(unavailable)}"
        )
    if not training_ids or not limix_ids:
        raise ValueError("Training and LimiX CPU sets must not be empty")
    if (
        inference_ids & training_set
        or inference_ids & limix_set
        or training_set & limix_set
    ):
        raise ValueError(
            "Inference, training, and LimiX CPU sets must be disjoint"
        )

    max_inference_threads = min(len(cpu_set) for cpu_set in inference_sets)
    inference_threads = min(
        int(
            os.getenv(
                "DUET_CPU_THREADS_PER_WORKER",
                str(max_inference_threads),
            )
        ),
        max_inference_threads,
    )
    if inference_threads <= 0:
        raise ValueError("DUET_CPU_THREADS_PER_WORKER must be positive")

    training_num_cpus = int(
        os.getenv(
            "DUET_AUTOGLUON_TRAIN_CPUS",
            str(min(20, len(training_ids))),
        )
    )
    if not 0 < training_num_cpus <= len(training_ids):
        raise ValueError(
            "DUET_AUTOGLUON_TRAIN_CPUS must fit inside the training CPU set"
        )

    nice_increment = int(os.getenv("DUET_BACKGROUND_TRAIN_NICE", "10"))
    if not 0 <= nice_increment <= 19:
        raise ValueError("DUET_BACKGROUND_TRAIN_NICE must be between 0 and 19")

    return CPUResourceConfig(
        inference_cpu_sets=inference_sets,
        training_cpu_ids=training_ids,
        limix_cpu_ids=limix_ids,
        inference_threads_per_worker=inference_threads,
        training_num_cpus=training_num_cpus,
        background_nice_increment=nice_increment,
    )


def pin_current_process(cpu_ids: tuple[int, ...]) -> None:
    """Pin the calling process or thread when the platform supports it."""

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))


@contextmanager
def cpu_affinity_scope(cpu_ids: tuple[int, ...]) -> Iterator[None]:
    """Temporarily constrain the calling thread and its child processes."""

    previous = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        previous = os.sched_getaffinity(0)
        os.sched_setaffinity(0, set(cpu_ids))
    try:
        yield
    finally:
        if previous is not None:
            os.sched_setaffinity(0, previous)


def configure_background_training(config: CPUResourceConfig) -> None:
    """Apply affinity and best-effort lower priority to a training thread."""

    pin_current_process(config.training_cpu_ids)
    if (
        not getattr(_background_state, "nice_configured", False)
        and config.background_nice_increment > 0
        and hasattr(os, "nice")
    ):
        try:
            os.nice(config.background_nice_increment)
        except OSError:
            pass
        _background_state.nice_configured = True


def format_cpu_sets(cpu_sets: tuple[tuple[int, ...], ...]) -> str:
    return ";".join(",".join(map(str, cpu_set)) for cpu_set in cpu_sets)

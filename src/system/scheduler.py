""" Scheduler """

import math
from dataclasses import dataclass
from typing import Literal

from src.system.resources import ResourceStatus

Route = Literal["limix", "autogluon"]
TrainingType = Literal["initial", "distillation"]


@dataclass(frozen=True)
class ModelEstimate:
    """Estimated model state used for scheduling."""

    accuracy: float
    throughput: float
    ready_seconds: float = 0.0
    startup_seconds: float = 0.0


@dataclass(frozen=True)
class AllocationDecision:
    """Batch allocation between LimiX and the tree model."""

    limix_count: int
    student_count: int
    limix_ratio: float
    expected_accuracy: float
    expected_seconds: float


def solve_allocation(
    batch_size: int,
    limix: ModelEstimate,
    student: ModelEstimate,
    min_accuracy: float,
    mixed_overhead_seconds: float = 0.0,
) -> AllocationDecision:
    """Minimize estimated batch completion time under the estimated accuracy constraint."""

    _validate_allocation_inputs(batch_size, limix, student, min_accuracy)
    if (
        not math.isfinite(mixed_overhead_seconds)
        or mixed_overhead_seconds < 0
    ):
        raise ValueError("mixed_overhead_seconds must be finite and non-negative")
    accuracy_gap = limix.accuracy - student.accuracy
    lower = 0
    upper = batch_size

    if accuracy_gap > 0:
        required = batch_size * (
            min_accuracy - student.accuracy
        ) / accuracy_gap
        lower = max(lower, math.ceil(required - 1e-12))
    elif accuracy_gap < 0:
        allowed = batch_size * (
            min_accuracy - student.accuracy
        ) / accuracy_gap
        upper = min(upper, math.floor(allowed + 1e-12))
    elif student.accuracy < min_accuracy:
        raise ValueError("Neither model can satisfy the estimated accuracy constraint")

    lower = max(0, lower)
    upper = min(batch_size, upper)
    if lower > upper:
        raise ValueError("No allocation satisfies the estimated accuracy constraint")

    time_optimal_count = (
        student.ready_seconds
        + student.startup_seconds
        - limix.ready_seconds
        - limix.startup_seconds
        + batch_size / student.throughput
    ) / (
        1.0 / limix.throughput
        + 1.0 / student.throughput
    )
    projected_count = min(max(time_optimal_count, lower), upper)

    candidate_counts = {lower, upper}
    candidate_counts.add(math.floor(projected_count))
    candidate_counts.add(math.ceil(projected_count))

    decisions = [
        _build_allocation_decision(
            batch_size=batch_size,
            limix_count=count,
            limix=limix,
            student=student,
            mixed_overhead_seconds=mixed_overhead_seconds,
        )
        for count in candidate_counts
        if lower <= count <= upper
    ]
    return min(
        decisions,
        key=lambda decision: (
            decision.expected_seconds,
            -decision.expected_accuracy,
        ),
    )


def _build_allocation_decision(
    batch_size: int,
    limix_count: int,
    limix: ModelEstimate,
    student: ModelEstimate,
    mixed_overhead_seconds: float = 0.0,
) -> AllocationDecision:
    student_count = batch_size - limix_count
    limix_seconds = _model_seconds(limix, limix_count)
    student_seconds = _model_seconds(student, student_count)
    expected_accuracy = (
        limix.accuracy * limix_count
        + student.accuracy * student_count
    ) / batch_size
    overhead = (
        mixed_overhead_seconds
        if limix_count > 0 and student_count > 0
        else 0.0
    )
    return AllocationDecision(
        limix_count=limix_count,
        student_count=student_count,
        limix_ratio=limix_count / batch_size,
        expected_accuracy=expected_accuracy,
        expected_seconds=max(limix_seconds, student_seconds) + overhead,
    )


def _model_seconds(model: ModelEstimate, sample_count: int) -> float:
    if sample_count == 0:
        return 0.0
    return (
        model.ready_seconds
        + model.startup_seconds
        + sample_count / model.throughput
    )


def _validate_allocation_inputs(
    batch_size: int,
    limix: ModelEstimate,
    student: ModelEstimate,
    min_accuracy: float,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be between zero and one")

    for name, estimate in (("limix", limix), ("student", student)):
        if not 0 <= estimate.accuracy <= 1:
            raise ValueError(f"{name} accuracy must be between zero and one")
        if not math.isfinite(estimate.throughput) or estimate.throughput <= 0:
            raise ValueError(f"{name} throughput must be finite and positive")
        if not math.isfinite(estimate.ready_seconds) or estimate.ready_seconds < 0:
            raise ValueError(f"{name} ready_seconds must be finite and non-negative")
        if not math.isfinite(estimate.startup_seconds) or estimate.startup_seconds < 0:
            raise ValueError(f"{name} startup_seconds must be finite and non-negative")

@dataclass
class SchedulerPolicy:
    gpu_idle_threshold: float = 20.0        # GPU utilization threshold for using LimiX.
    min_gpu_memory_mb: float = 2000.0       # GPU memory threshold for using LimiX.
    use_limix_when_gpu_idle: bool = True

@dataclass(frozen=True)
class SystemState:
    """ System state """
    student_ready: bool         # Whether the student is ready.
    training_running: bool      # Whether training is running.
    drift_detected: bool        # Whether a shift has been detected.
    resources: ResourceStatus   # Resource status.


@dataclass(frozen=True)
class ScheduleDecision:
    route: Route
    start_training: bool
    training_type: TrainingType | None
    reason: str


class Scheduler:
    def __init__(
        self,
        policy: SchedulerPolicy = SchedulerPolicy(),
    ) -> None:
        self.policy = policy

    def decide(self, state: SystemState) -> ScheduleDecision:
        """ Choose a scheduling policy from the combined system state. """
        resources = state.resources

        # Whether the GPU can run LimiX.
        gpu_feasible = (
            resources.gpu_available 
            and resources.gpu_memory_free_mb is not None
            and resources.gpu_memory_free_mb >= self.policy.min_gpu_memory_mb
        )

        # Whether the GPU is idle.
        gpu_idle = (
            gpu_feasible
            and resources.gpu_utilization is not None
            and resources.gpu_utilization < self.policy.gpu_idle_threshold
        )

        # The tree model is unavailable during cold start.
        if not state.student_ready:
            if not gpu_feasible:
                # Raise an error if no model is available.
                raise RuntimeError(
                    "The student is not ready and LimiX cannot run"
                )

            return ScheduleDecision(
                route="limix",
                start_training = not state.training_running,
                training_type=(
                    None
                    if state.training_running
                    else "initial"
                ),
                reason="cold_start",
            )

        # Distribution shift.
        if state.drift_detected:
            if gpu_feasible:
                # Use LimiX when the GPU is available.
                return ScheduleDecision(
                    route="limix",
                    start_training=not state.training_running,
                    training_type=(
                        None
                        if state.training_running
                        else "distillation"
                    ),
                    reason="distribution_shift",
                )

            # Fall back to the old tree model on shifted data when LimiX cannot run.
            return ScheduleDecision(
                route="autogluon",
                start_training=False,
                training_type=None,
                reason="distribution_shift_gpu_unavailable",
            )

        if (
            self.policy.use_limix_when_gpu_idle
            and gpu_idle
        ):
            return ScheduleDecision(
                route="limix",
                start_training=False,
                training_type=None,
                reason="stable_gpu_idle",
            )

        # Otherwise, use AutoGluon by default. 
        return ScheduleDecision(
            route="autogluon",
            start_training=False,
            training_type=None,
            reason="stable_student_route",
        )

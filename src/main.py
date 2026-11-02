""" Main workflow """

import time
import sys
import os
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

import torch
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import get_dataset_spec
from src.data.types import BootstrapWindow, ProblemType, StreamConfig
from src.data.builder import StreamBuilder
from src.data.stream import DataStream
from src.detection import PCAPSIConfig, PCAPSIDriftDetector
from src.experiment_runtime import (
    AutoGluonRuntimeConfig,
    ExperimentRuntimeConfig,
)
from src.engines.tbe.autogluon import (
    AUTOGLUON_CPUS_PER_MODEL,
    AUTOGLUON_FIT_STRATEGY,
    AUTOGLUON_MODEL_KEYS,
    AutoGluonStudent,
    shutdown_ray,
)
from src.engines.tbe.autogluon_parallel import AutoGluonPool
from src.engines.tbe.distillation import distill_replay_student
from src.system.resources import get_resource_status
from src.system.cpu_resources import (
    CPUResourceConfig,
    configure_background_training,
    cpu_affinity_scope,
    format_cpu_sets,
    get_cpu_resource_config,
    pin_current_process,
)
from src.system.profiling import load_profile
from src.system.scheduler import (
    AllocationDecision,
    ModelEstimate,
    Scheduler,
    SchedulerPolicy,
    SystemState,
    solve_allocation,
)
from src.evaluation.metrics import evaluate_window
from src.engines.fbe.limix_parallel import LimiXPool


@dataclass(frozen=True)
class BackgroundTrainingResult:
    """Serializable reference to a student trained in another process."""

    model_path: str
    model_name: str
    classes: np.ndarray
    training_seconds: float
    interval_start: float
    interval_end: float


def _configured_student(
    model_path: Path,
    problem_type: ProblemType,
    time_limit: int,
    num_cpus: int,
    model_config: AutoGluonRuntimeConfig,
) -> AutoGluonStudent:
    return AutoGluonStudent(
        model_path=model_path,
        problem_type=problem_type,
        time_limit=time_limit,
        num_cpus=num_cpus,
        model_keys=model_config.model_keys,
        fit_strategy=model_config.fit_strategy,
        cpus_per_model=model_config.cpus_per_model,
        presets=model_config.presets,
        num_bag_folds=model_config.num_bag_folds,
        num_bag_sets=model_config.num_bag_sets,
        num_stack_levels=model_config.num_stack_levels,
        use_bag_holdout=model_config.use_bag_holdout,
        dynamic_stacking=model_config.dynamic_stacking,
    )


def _prepare_background_training_worker(cpu_config: CPUResourceConfig) -> int:
    """Pin and pre-start the worker before any online window is timed."""
    configure_background_training(cpu_config)
    return os.getpid()


def _training_result(
    model: AutoGluonStudent,
    start: float,
    end: float,
) -> BackgroundTrainingResult:
    if model.model_name is None or model.classes is None:
        raise RuntimeError("AutoGluon background training produced no model")
    return BackgroundTrainingResult(
        model_path=str(model.model_path),
        model_name=model.model_name,
        classes=model.classes.copy(),
        training_seconds=end - start,
        interval_start=start,
        interval_end=end,
    )


def _train_initial_student_in_process(
    bootstrap: BootstrapWindow,
    model_path: Path,
    problem_type: ProblemType,
    time_limit: int,
    model_config: AutoGluonRuntimeConfig,
    cpu_config: CPUResourceConfig,
) -> BackgroundTrainingResult:
    configure_background_training(cpu_config)
    start = time.perf_counter()
    try:
        model = _configured_student(
            model_path,
            problem_type,
            time_limit,
            cpu_config.training_num_cpus,
            model_config,
        )
        model.fit(bootstrap)
        end = time.perf_counter()
        return _training_result(model, start, end)
    finally:
        shutdown_ray()


def _distill_student_in_process(
    bootstrap: BootstrapWindow,
    shift_features: pd.DataFrame,
    teacher_probabilities: np.ndarray,
    model_path: Path,
    problem_type: ProblemType,
    shift_weight: float,
    time_limit: int,
    seed: int,
    model_config: AutoGluonRuntimeConfig,
    cpu_config: CPUResourceConfig,
) -> BackgroundTrainingResult:
    configure_background_training(cpu_config)
    start = time.perf_counter()
    try:
        model = distill_replay_student(
            bootstrap=bootstrap,
            shift_features=shift_features,
            teacher_probabilities=teacher_probabilities,
            model_path=model_path,
            problem_type=problem_type,
            shift_weight=shift_weight,
            time_limit=time_limit,
            seed=seed,
            num_cpus=cpu_config.training_num_cpus,
            model_keys=model_config.model_keys,
            fit_strategy=model_config.fit_strategy,
            cpus_per_model=model_config.cpus_per_model,
            presets=model_config.presets,
            num_bag_folds=model_config.num_bag_folds,
            num_bag_sets=model_config.num_bag_sets,
            num_stack_levels=model_config.num_stack_levels,
            use_bag_holdout=model_config.use_bag_holdout,
            dynamic_stacking=model_config.dynamic_stacking,
        )
        end = time.perf_counter()
        return _training_result(model, start, end)
    finally:
        shutdown_ray()


def main(
    runtime_config: ExperimentRuntimeConfig | None = None,
    cpu_config: CPUResourceConfig | None = None,
    limix_gpu_ids: tuple[int, ...] | None = None,
    data_stream: DataStream | None = None,
    profile_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run the DUET streaming inference experiment.

    Explicit arguments support controlled experiments; calling without arguments preserves the original .env behavior.
    """

    if runtime_config is None:
        stream_config = StreamConfig(
            bootstrap_size=int(os.getenv("DUET_BOOTSTRAP_SIZE", "500")),
            bootstrap_val_ratio=0.2,
            window_size=int(os.getenv("DUET_WINDOW_SIZE", "2000")),
            seed=2026,
            shift_start_window=int(
                os.getenv("DUET_SHIFT_START_WINDOW", "15")
            ),
            shift_n_clusters=int(os.getenv("DUET_SHIFT_N_CLUSTERS", "8")),
            shift_cluster_order=os.getenv(
                "DUET_SHIFT_CLUSTER_ORDER", "distance_from_w0"
            ),
            shift_block_windows=int(
                os.getenv("DUET_SHIFT_BLOCK_WINDOWS", "3")
            ),
            shift_mode=os.getenv(
                "DUET_SHIFT_MODE", "domain_aligned_propensity"
            ),
            shift_propensity_feature=os.getenv(
                "DUET_SHIFT_PROPENSITY_FEATURE", "pca_pc2"
            ),
            shift_propensity_strength=float(
                os.getenv("DUET_SHIFT_PROPENSITY_STRENGTH", "1.0")
            ),
            shift_propensity_direction=os.getenv(
                "DUET_SHIFT_PROPENSITY_DIRECTION", "low_to_high"
            ),
        )
        runtime_config = ExperimentRuntimeConfig(
            dataset_name=os.getenv("DUET_DATASET", "particle"),
            stream=stream_config,
            train_time_limit=int(
                os.getenv("DUET_TRAIN_TIME_LIMIT", "60")
            ),
            accuracy_tolerance=float(
                os.getenv("DUET_ACCURACY_TOLERANCE", "0.01")
            ),
            drift_detector=PCAPSIConfig(
                threshold=float(os.getenv("DUET_DRIFT_THRESHOLD", "0.2")),
            ),
            autogluon=AutoGluonRuntimeConfig(
                model_keys=AUTOGLUON_MODEL_KEYS,
                fit_strategy=AUTOGLUON_FIT_STRATEGY,
                cpus_per_model=AUTOGLUON_CPUS_PER_MODEL,
                presets="good_quality",
                num_bag_folds=5,
                num_bag_sets=2,
                num_stack_levels=0,
                use_bag_holdout=True,
                dynamic_stacking=False,
            ),
            duet_autogluon=AutoGluonRuntimeConfig(
                model_keys=AUTOGLUON_MODEL_KEYS,
                fit_strategy=AUTOGLUON_FIT_STRATEGY,
                cpus_per_model=AUTOGLUON_CPUS_PER_MODEL,
                presets="medium_quality",
                num_bag_folds=0,
                num_bag_sets=1,
                num_stack_levels=0,
                use_bag_holdout=False,
                dynamic_stacking=False,
            ),
        )

    cpu_config = cpu_config or get_cpu_resource_config()
    runtime_config.validate(cpu_config.training_num_cpus)
    DATASET_NAME = runtime_config.dataset_name
    TRAIN_TIME_LIMIT = runtime_config.train_time_limit
    WINDOW_INTERVAL_SECONDS = runtime_config.window_interval_seconds
    LIMIX_GPU_IDS = (
        tuple(range(torch.cuda.device_count()))
        if limix_gpu_ids is None
        else tuple(limix_gpu_ids)
    )
    if not LIMIX_GPU_IDS:
        raise RuntimeError("DUET requires at least one GPU for LimiX")
    # DUET's student is on the online critical path.  Keep its lightweight
    # configuration separate from the heavier baseline search configuration.
    model_config = runtime_config.student_autogluon_config()
    AUTOGLUON_CPU_SETS = cpu_config.inference_cpu_sets
    AUTOGLUON_THREADS_PER_WORKER = (
        cpu_config.inference_threads_per_worker
    )
    pin_current_process(cpu_config.limix_cpu_ids)
    print(
        "CPU resource isolation: "
        "student_config=lightweight, "
        f"inference={format_cpu_sets(AUTOGLUON_CPU_SETS)}, "
        f"training={','.join(map(str, cpu_config.training_cpu_ids))}, "
        f"training_num_cpus={cpu_config.training_num_cpus}, "
        f"training_strategy={model_config.fit_strategy}, "
        f"candidates={len(model_config.model_keys)}, "
        f"cpus_per_candidate={model_config.cpus_per_model}, "
        f"preset={model_config.presets}, "
        f"bag_folds={model_config.num_bag_folds}, "
        f"parallel_cpus={len(model_config.model_keys) * model_config.cpus_per_model}, "
        f"limix={','.join(map(str, cpu_config.limix_cpu_ids))}, "
        f"training_nice=+{cpu_config.background_nice_increment}"
    )
    
    # Combined weight of shifted windows during retraining.
    DISTILL_SHIFT_WEIGHT = runtime_config.distill_shift_weight
    DISTILL_WINDOW_COUNT = runtime_config.distill_window_count
    MAX_W0_ACCURACY_DROP = runtime_config.max_w0_accuracy_drop
    
    # Scheduling accuracy estimates must not use ground-truth labels from online windows.
    EXPECTED_ACCURACY_TOLERANCE = runtime_config.accuracy_tolerance
    dataset_spec = get_dataset_spec(DATASET_NAME)
    if profile_path is None:
        profile_path = (
            PROJECT_ROOT / "output" / "profiles" / f"{DATASET_NAME}.json"
        )
    profile = load_profile(profile_path, dataset_name=DATASET_NAME)
    MIN_EXPECTED_ACCURACY = max(
        0.0,
        profile.limix.accuracy - EXPECTED_ACCURACY_TOLERANCE,
    )
    
    # Experiment configuration.
    expr_config = runtime_config.stream
    print(f"Data stream shift: {expr_config.shift_summary()}")
    
    # Stream construction.
    if data_stream is None:
        stream_builder = StreamBuilder(dataset_spec, expr_config)
        data_stream = stream_builder.build()
    
    
    # Build W0 for initialization.
    bootstrap = data_stream.bootstrap()

    drift_detector = PCAPSIDriftDetector(runtime_config.drift_detector)
    drift_detector.fit(pd.concat(
        [bootstrap.train_features, bootstrap.val_features],
        ignore_index=True,
    ))
    print(
        "PCA/PSI detector initialized from W0 features, "
        f"components={drift_detector.component_count}, "
        f"threshold={drift_detector.config.threshold:.4f}"
    )
    
    # Deploy one persistent LimiX replica on each GPU.
    limix_model = LimiXPool(
        model_path = PROJECT_ROOT / "cache" / "LimiX-2M.ckpt",
        config_path = PROJECT_ROOT / "config" / "cls_default_noretrieval.json",
        gpu_ids = LIMIX_GPU_IDS,
        seed = expr_config.seed,
        cpu_ids = cpu_config.limix_cpu_ids,
        max_shard_size = runtime_config.limix_max_shard_size,
        warmup_sample_count = runtime_config.limix_warmup_sample_count,
    )
    
    limix_model.initialize(bootstrap)
    
    # Online scheduling uses offline calibration, not online window labels.
    limix_accuracy_estimate = profile.limix.accuracy
    limix_throughput_estimate = profile.limix.throughput
    limix_startup_seconds = profile.limix.startup_seconds
    distillation_seconds_estimate = profile.distillation_seconds
    
    print(
        f"Loaded offline profiles, calibration_seed={profile.calibration_seed}, "
        f"minimum_estimated_accuracy={MIN_EXPECTED_ACCURACY:.4f}, "
        f"accuracy_tolerance={EXPECTED_ACCURACY_TOLERANCE:.4f}, "
        f"estimated_distillation_seconds={distillation_seconds_estimate:.2f} s"
    )
    
    # Use a separate directory to preserve existing models.
    if output_dir is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = PROJECT_ROOT / "output" / f"stream_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = output_dir / "results.csv"
    
    def train_w0_retrain_student() -> AutoGluonStudent:
        """Train a W0-only control with the same training budget."""
        model = _configured_student(
            output_dir / "w0_retrain_student",
            dataset_spec.problem_type,
            TRAIN_TIME_LIMIT,
            cpu_config.training_num_cpus,
            model_config,
        )
        model.fit(bootstrap)
        return model


    def materialize_background_result(
        result: BackgroundTrainingResult,
    ) -> tuple[AutoGluonStudent, float, float, float]:
        """Load only serializable metadata returned by the training process."""
        model = _configured_student(
            Path(result.model_path),
            dataset_spec.problem_type,
            TRAIN_TIME_LIMIT,
            cpu_config.training_num_cpus,
            model_config,
        )
        model.load_trained(result.model_name, result.classes)
        return (
            model,
            result.training_seconds,
            result.interval_start,
            result.interval_end,
        )


    def merged_duration(intervals: list[tuple[float, float]]) -> float:
        """Compute the union duration of training and inference intervals."""
        if not intervals:
            return 0.0

        ordered = sorted(intervals)
        current_start, current_end = ordered[0]
        total = 0.0

        for start, end in ordered[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end

        return total + current_end - current_start


    def reload_student_pool(
        pool: AutoGluonPool,
        model: AutoGluonStudent,
    ) -> float:
        """Create a parallel CPU inference pool for the trained student."""

        if model.model_name is None or model.classes is None:
            raise RuntimeError("The student model has not finished training")

        reload_start = time.perf_counter()
        pool.reload_model(
            model_path=model.model_path,
            model_name=model.model_name,
            classes=model.classes,
            warmup_features=bootstrap.val_features.iloc[:32],
        )
        return time.perf_counter() - reload_start
    
    
    def measure_w0_accuracy(
        model: AutoGluonStudent,
    ) -> float:
        """Check retention of original knowledge on the W0 validation split."""
    
        probabilities = model.predict_proba(bootstrap.val_features)
        labels = bootstrap.classes[probabilities.argmax(axis=1)]
        return float(np.mean(labels == bootstrap.val_labels))
    
    
    def predict_student(
        pool: AutoGluonPool,
        features: pd.DataFrame,
    ) -> tuple[np.ndarray, float]:
        """Run student inference and return inference-only time."""

        start = time.perf_counter()
        probabilities = pool.predict_proba(features)
        return probabilities, time.perf_counter() - start
    
    
    def update_throughput(
        previous: float,
        sample_count: int,
        inference_seconds: float,
        startup_seconds: float,
    ) -> float:
        """Smooth the throughput estimate using the latest observation."""
    
        variable_seconds = inference_seconds - startup_seconds
        if sample_count == 0 or variable_seconds <= 0:
            return previous
        observed = sample_count / variable_seconds
        return 0.8 * previous + 0.2 * observed
    
    
    def compare_students(
        initial_stu: AutoGluonStudent,
        w0_retrain_stu: AutoGluonStudent | None,
        replay_stu: AutoGluonStudent,
        limix_model: LimiXPool,
        start_window: int,
    ) -> None:
        """Compare the original student, LimiX, and distilled student on identical shifted windows."""
    
        results = []
    
        for window in data_stream:
            if window.window_id < start_window:
                continue
    
            target = data_stream.eval_target(window.window_id)
    
            initial_proba = initial_stu.predict_proba(window.features)
            w0_retrain_proba = (
                None
                if w0_retrain_stu is None
                else w0_retrain_stu.predict_proba(window.features)
            )
            limix_proba = limix_model.predict_proba(window.features)
            replay_proba = replay_stu.predict_proba(window.features)
    
            # Convert probabilities to predicted labels.
            initial_labels = bootstrap.classes[initial_proba.argmax(axis=1)]
            w0_retrain_labels = (
                None
                if w0_retrain_proba is None
                else bootstrap.classes[w0_retrain_proba.argmax(axis=1)]
            )
            limix_labels = bootstrap.classes[limix_proba.argmax(axis=1)]
            replay_labels = bootstrap.classes[replay_proba.argmax(axis=1)]

            initial_acc = accuracy_score(target.labels, initial_labels)
            w0_retrain_acc = (
                np.nan
                if w0_retrain_labels is None
                else accuracy_score(target.labels, w0_retrain_labels)
            )
            limix_acc = accuracy_score(target.labels, limix_labels)
            replay_acc = accuracy_score(target.labels, replay_labels)
    
            # Use the complete class space for consistent macro-F1 across models.
            initial_f1 = f1_score(
                target.labels,
                initial_labels,
                labels=bootstrap.classes,
                average="macro",
                zero_division=0,
            )
            w0_retrain_f1 = (
                np.nan
                if w0_retrain_labels is None
                else f1_score(
                    target.labels,
                    w0_retrain_labels,
                    labels=bootstrap.classes,
                    average="macro",
                    zero_division=0,
                )
            )
            limix_f1 = f1_score(
                target.labels,
                limix_labels,
                labels=bootstrap.classes,
                average="macro",
                zero_division=0,
            )
            replay_f1 = f1_score(
                target.labels,
                replay_labels,
                labels=bootstrap.classes,
                average="macro",
                zero_division=0,
            )
    
            results.append(
                {
                    "window_id": window.window_id,
                    "initial_acc": initial_acc,
                    "w0_retrain_acc": w0_retrain_acc,
                    "limix_acc": limix_acc,
                    "replay_acc": replay_acc,
                    "recovery_accuracy_pp": 100.0 * (
                        replay_acc - initial_acc
                    ),
                    "initial_f1": initial_f1,
                    "w0_retrain_f1": w0_retrain_f1,
                    "limix_f1": limix_f1,
                    "replay_f1": replay_f1,
                    "recovery_macro_f1_pp": 100.0 * (
                        replay_f1 - initial_f1
                    ),
                }
            )
    
        if not results:
            print("No windows remain after distillation for student comparison")
            return
    
        for result in results:
            print(
                f"W{result['window_id']}: "
                f"Accuracy old={result['initial_acc']:.4f}, "
                f"LimiX={result['limix_acc']:.4f}, "
                f"distill={result['replay_acc']:.4f}; "
                f"Macro-F1 old={result['initial_f1']:.4f}, "
                f"LimiX={result['limix_f1']:.4f}, "
                f"distill={result['replay_f1']:.4f}"
            )
    
        comparison = pd.DataFrame(results)
        comparison.to_csv(
            output_dir / "student_comparison.csv",
            index=False,
        )

        summary = pd.DataFrame(
            [
                {
                    "dataset": DATASET_NAME,
                    "metric": "Accuracy",
                    "old_student_after_shift": comparison[
                        "initial_acc"
                    ].mean(),
                    "limix_reference": comparison["limix_acc"].mean(),
                    "distilled_student": comparison["replay_acc"].mean(),
                    "recovery_over_old_student_pp": comparison[
                        "recovery_accuracy_pp"
                    ].mean(),
                },
                {
                    "dataset": DATASET_NAME,
                    "metric": "Macro-F1",
                    "old_student_after_shift": comparison[
                        "initial_f1"
                    ].mean(),
                    "limix_reference": comparison["limix_f1"].mean(),
                    "distilled_student": comparison["replay_f1"].mean(),
                    "recovery_over_old_student_pp": comparison[
                        "recovery_macro_f1_pp"
                    ].mean(),
                },
            ]
        )
        summary.to_csv(
            output_dir / "distillation_summary.csv",
            index=False,
        )
    
        print("Student comparison on identical windows:")
        print(
            "Accuracy: "
            f"old={comparison['initial_acc'].mean():.4f}, "
            f"LimiX={comparison['limix_acc'].mean():.4f}, "
            f"distill={comparison['replay_acc'].mean():.4f}, "
            "recovery="
            f"{comparison['recovery_accuracy_pp'].mean():+.2f} pp"
        )
        print(
            "Macro-F1: "
            f"old={comparison['initial_f1'].mean():.4f}, "
            f"LimiX={comparison['limix_f1'].mean():.4f}, "
            f"distill={comparison['replay_f1'].mean():.4f}, "
            "recovery="
            f"{comparison['recovery_macro_f1_pp'].mean():+.2f} pp"
        )
        print(
            "Distillation summary saved to: "
            f"{output_dir / 'distillation_summary.csv'}"
        )
    
    
    initial_stu = None
    replay_stu = None
    replay_ready_window = None
    
    # Currently active student model.
    student = None
    student_pool = AutoGluonPool(
        cpu_sets=AUTOGLUON_CPU_SETS,
        threads_per_worker=AUTOGLUON_THREADS_PER_WORKER,
    )
    student_pool.initialize()
    stu_name = None
    student_w0_accuracy = None
    student_accuracy_estimate = None
    student_throughput_estimate = None
    student_startup_seconds_estimate = None
    
    # Background training task.
    training_future = None
    training_name = None
    training_seconds_total = 0.0
    training_intervals = []
    inference_intervals = []
    
    shift_started = False
    adaptation_pending = False
    drift_episode_active = False
    student_needs_recovery = False
    replay_reference_features = None
    
    # Buffer features and LimiX probabilities from consecutive shifted windows.
    shift_features_buffer = []
    shift_probabilities_buffer = []
    shift_window_ids = []
    
    # AutoGluon/Ray runs in a dedicated process so its Python runtime and
    # process-global state cannot stall the online LimiX scheduler. Start the
    # worker before W1 so spawn/import cost is outside inference timing.
    training_executor = ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
    )
    background_training_pid = training_executor.submit(
        _prepare_background_training_worker,
        cpu_config,
    ).result()
    print(
        "AutoGluon background training process ready, "
        f"pid={background_training_pid}, isolation=spawn"
    )
    inference_executor = ThreadPoolExecutor(max_workers=1)
    
    scheduler = Scheduler(
        SchedulerPolicy(use_limix_when_gpu_idle=False)
    )
    
    results = []
    
    # Process the stream window by window.
    for window in data_stream:
        window_start = time.perf_counter()
        student_update = None
        old_w0_accuracy = None
        candidate_w0_accuracy = None
        training_started_this_window = False
        training_type_started = None
        distillation_windows = None
        training_seconds_observed = None
        training_completed_type = None
        allocation = None
        selected_student_kind = None
        used_student_model = None
        model_reload_seconds = None
    
        # 1. Check for completed background training at each window boundary.
        if training_future is not None and training_future.done():
            (
                candidate,
                training_seconds_observed,
                training_interval_start,
                training_interval_end,
            ) = materialize_background_result(training_future.result())
            training_seconds_total += training_seconds_observed
            training_intervals.append(
                (training_interval_start, training_interval_end)
            )
            training_completed_type = (
                "distillation"
                if training_name is not None
                and training_name.startswith("replay_student")
                else "initial"
            )
            candidate_w0_accuracy = measure_w0_accuracy(candidate)
    
            if (
                training_name is not None
                and training_name.startswith("replay_student")
            ):
                distillation_seconds_estimate = (
                    0.8 * distillation_seconds_estimate
                    + 0.2 * training_seconds_observed
                )
    

            if (
                training_name is not None
                and training_name.startswith("replay_student")
            ):
                old_w0_accuracy = student_w0_accuracy
                if candidate_w0_accuracy >= old_w0_accuracy - MAX_W0_ACCURACY_DROP:
                    replay_stu = candidate
                    replay_ready_window = window.window_id
                    model_reload_seconds = reload_student_pool(
                        student_pool,
                        candidate,
                    )
                    student = candidate
                    stu_name = training_name
                    student_w0_accuracy = candidate_w0_accuracy
                    student_accuracy_estimate = profile.future_student.accuracy
                    student_throughput_estimate = profile.future_student.throughput
                    student_startup_seconds_estimate = (
                        profile.future_student.startup_seconds
                    )
                    student_update = "accepted"
                    # Monitor against the data used for this accepted update,
                    # so a persistent shift does not retrigger the same replay.
                    drift_detector.reset()
                    drift_detector.fit(replay_reference_features)
                    drift_episode_active = False
                    student_needs_recovery = False
                    print("PCA/PSI reference updated to the accepted replay windows")
                else:
                    student_update = "rejected"
                adaptation_pending = False
                replay_reference_features = None
            else:
                initial_stu = candidate
                model_reload_seconds = reload_student_pool(
                    student_pool,
                    candidate,
                )
                student = candidate
                stu_name = training_name
                student_w0_accuracy = candidate_w0_accuracy
                student_accuracy_estimate = profile.stable_student.accuracy
                student_throughput_estimate = profile.stable_student.throughput
                student_startup_seconds_estimate = (
                    profile.stable_student.startup_seconds
                )
                student_update = "accepted"
    
            reload_text = (
                "not_reloaded"
                if model_reload_seconds is None
                else f"{model_reload_seconds:.2f}"
            )
            print(
                f"{training_name} training completed, "
                f"update={student_update}, "
                f"active_tree_model={student.model_name}, "
                f"candidate_W0_accuracy={candidate_w0_accuracy:.4f}, "
                f"training_seconds={training_seconds_observed:.2f}, "
                f"training_seconds_total={training_seconds_total:.2f}, "
                f"model_reload_seconds={reload_text}"
            )
    
            training_future = None
            training_name = None
    
        # The synthetic change point is not an online detection signal.
        drift_signal = drift_detector.observe(
            window.features,
            window_id=window.window_id,
        )
        drift_alert = (
            drift_signal.detected
            and not drift_episode_active
            and not adaptation_pending
        )
        if drift_alert:
            shift_started = True
            adaptation_pending = True
            drift_episode_active = True
            student_needs_recovery = True
            print(
                f"Drift detected at W{window.window_id}: "
                f"PSI={drift_signal.score:.4f}, "
                f"threshold={drift_signal.threshold:.4f}; "
                "replay collection is pending"
            )
        elif not drift_signal.detected and not adaptation_pending:
            drift_episode_active = False
    
        # 3. Collect resource status and obtain a scheduling decision.
        resources = get_resource_status()
    
        state = SystemState(
            student_ready=student is not None and student_pool is not None,
            training_running=training_future is not None,
            drift_detected=adaptation_pending,
            resources=resources
        )
    
        lifecycle_decision = scheduler.decide(state)
    
        # 4. Train the initial student asynchronously during cold start.
    
        # Execute train_initial_student when the scheduler requests initialization.
        if (
            lifecycle_decision.start_training
            and lifecycle_decision.training_type == "initial"
        ):
            training_name = "initial_student"
    
            training_future = training_executor.submit(
                _train_initial_student_in_process,
                bootstrap,
                output_dir / "initial_student",
                dataset_spec.problem_type,
                TRAIN_TIME_LIMIT,
                model_config,
                cpu_config,
            )
            training_started_this_window = True
            training_type_started = "initial"
    
    
        # 5. Compute the model allocation for this window.
        force_limix = (
            student is None
            or student_pool is None
            or (
                lifecycle_decision.start_training
                and lifecycle_decision.training_type == "distillation"
            )
        )
    
        if force_limix:
            allocation = AllocationDecision(
                limix_count=len(window.features),
                student_count=0,
                limix_ratio=1.0,
                expected_accuracy=limix_accuracy_estimate,
                expected_seconds=(
                    limix_startup_seconds
                    + len(window.features) / limix_throughput_estimate
                ),
            )
            allocation_reason = lifecycle_decision.reason
        else:
            limix_estimate = ModelEstimate(
                accuracy=limix_accuracy_estimate,
                throughput=limix_throughput_estimate,
                startup_seconds=limix_startup_seconds,
            )
            current_accuracy_estimate = (
                profile.shifted_student.accuracy
                if student_needs_recovery
                else student_accuracy_estimate
            )
            current_estimate = ModelEstimate(
                accuracy=current_accuracy_estimate,
                throughput=student_throughput_estimate,
                startup_seconds=student_startup_seconds_estimate,
            )
            allocation = solve_allocation(
                batch_size=len(window.features),
                limix=limix_estimate,
                student=current_estimate,
                min_accuracy=MIN_EXPECTED_ACCURACY,
                mixed_overhead_seconds=profile.mixed_overhead_seconds,
            )
            selected_student_kind = "current"
            allocation_reason = "accuracy_constrained_allocation"
    
        # 6. Split contiguous shards; each sample is processed by exactly one engine.
        inference_start = time.perf_counter()
        if allocation.limix_count == 0:
            limix_features = None
            student_features = window.features
        elif allocation.student_count == 0:
            limix_features = window.features
            student_features = None
        else:
            limix_features = window.features.iloc[: allocation.limix_count]
            student_features = window.features.iloc[allocation.limix_count :]
        partition_end = time.perf_counter()
        partition_seconds = partition_end - inference_start

        student_job = None
        if student_features is not None:
            student_job = inference_executor.submit(
                predict_student,
                student_pool,
                student_features,
            )
    
        limix_inference_seconds = 0.0
        limix_probabilities = None
        if limix_features is not None:
            limix_start = time.perf_counter()
            limix_probabilities = limix_model.predict_proba(
                limix_features
            )
            limix_inference_seconds = time.perf_counter() - limix_start
    
        student_inference_seconds = 0.0
        student_probabilities = None
        if student_job is not None:
            student_probabilities, student_inference_seconds = (
                student_job.result()
            )
            used_student_model = student.model_name

        engine_end = time.perf_counter()
        probabilities = np.empty(
            (len(window.features), len(bootstrap.classes)),
            dtype=np.float64,
        )
        if limix_probabilities is not None:
            probabilities[: allocation.limix_count] = limix_probabilities
        if student_probabilities is not None:
            probabilities[allocation.limix_count :] = student_probabilities
        inference_end = time.perf_counter()
        inference_seconds = inference_end - inference_start
        engine_parallel_seconds = engine_end - partition_end
        merge_seconds = inference_end - engine_end
        inference_intervals.append((inference_start, inference_end))
        limix_throughput_estimate = update_throughput(
            limix_throughput_estimate,
            allocation.limix_count,
            limix_inference_seconds,
            limix_startup_seconds,
        )
        student_throughput_estimate = (
            None
            if student_throughput_estimate is None
            else update_throughput(
                student_throughput_estimate,
                allocation.student_count,
                student_inference_seconds,
                student_startup_seconds_estimate,
            )
        )
    
        if allocation.limix_count == 0:
            route = "autogluon"
        elif allocation.student_count == 0:
            route = "limix"
        else:
            route = "mixed"
    
        # 7. Buffer shifted windows and start asynchronous distillation once enough are collected.
        if (
            lifecycle_decision.start_training
            and lifecycle_decision.training_type == "distillation"
            and training_future is None
        ):
            if route != "limix":
                raise RuntimeError(
                    "Shifted windows must first be processed by LimiX"
                )
    
            shift_features_buffer.append(window.features.copy())
            shift_probabilities_buffer.append(probabilities.copy())
            shift_window_ids.append(window.window_id)
    
            print(f"Collected shifted windows: {shift_window_ids}")
    
            if len(shift_window_ids) >= DISTILL_WINDOW_COUNT:
                combined_shift_features = pd.concat(
                    shift_features_buffer,
                    ignore_index=True,
                )
                combined_teacher_probabilities = np.concatenate(
                    shift_probabilities_buffer,
                    axis=0,
                )
                replay_reference_features = combined_shift_features
    
                first_window = shift_window_ids[0]
                last_window = shift_window_ids[-1]
                training_name = (
                    f"replay_student_w{first_window}_w{last_window}"
                )
                distillation_windows = ",".join(
                    str(window_id) for window_id in shift_window_ids
                )
    
                training_future = training_executor.submit(
                    _distill_student_in_process,
                    bootstrap,
                    combined_shift_features,
                    combined_teacher_probabilities,
                    output_dir / training_name,
                    dataset_spec.problem_type,
                    DISTILL_SHIFT_WEIGHT,
                    TRAIN_TIME_LIMIT,
                    expr_config.seed,
                    model_config,
                    cpu_config,
                )
                training_started_this_window = True
                training_type_started = "distillation"
    
                print(
                    f"Using shifted windows {distillation_windows} "
                    "to start replay distillation"
                )
    
                shift_features_buffer.clear()
                shift_probabilities_buffer.clear()
                shift_window_ids.clear()
    
        # 8. Save results.
        target = data_stream.eval_target(
            window.window_id
        )
    
        # Compute evaluation metrics.
        metrics = evaluate_window(
            true_labels=target.labels,
            probabilities=probabilities,
            classes=bootstrap.classes,
            inference_seconds=inference_seconds,
        )
    
        # Window processing time includes resource monitoring, scheduling, inference, and evaluation.
        window_seconds = time.perf_counter() - window_start
    
        # 9. Save per-window experiment records.
        result = {
            "window_id": window.window_id,
            "phase": (
                "post_shift"
                if shift_started
                else "pre_shift"
            ),
            "drift_score": drift_signal.score,
            "drift_threshold": drift_signal.threshold,
            "drift_detected": drift_signal.detected,
            "drift_alert": drift_alert,
            "adaptation_pending": adaptation_pending,
            "route": route,
            "reason": allocation_reason,
            "active_student": stu_name,
            "tree_model": used_student_model,
            "selected_student": selected_student_kind,
            "limix_gpu_count": len(LIMIX_GPU_IDS),
            "limix_gpu_ids": ",".join(map(str, LIMIX_GPU_IDS)),
            "autogluon_inference_cpu_sets": format_cpu_sets(
                AUTOGLUON_CPU_SETS
            ),
            "autogluon_threads_per_worker": AUTOGLUON_THREADS_PER_WORKER,
            "autogluon_training_cpu_ids": ",".join(
                map(str, cpu_config.training_cpu_ids)
            ),
            "autogluon_training_num_cpus": cpu_config.training_num_cpus,
            "autogluon_config_role": "duet_student",
            "autogluon_fit_strategy": model_config.fit_strategy,
            "autogluon_candidate_count": len(model_config.model_keys),
            "autogluon_cpus_per_candidate": model_config.cpus_per_model,
            "autogluon_presets": model_config.presets,
            "autogluon_num_bag_folds": model_config.num_bag_folds,
            "autogluon_num_bag_sets": model_config.num_bag_sets,
            "autogluon_num_stack_levels": model_config.num_stack_levels,
            "autogluon_use_bag_holdout": model_config.use_bag_holdout,
            "autogluon_dynamic_stacking": model_config.dynamic_stacking,
            "background_training_isolation": "spawn_process",
            "background_training_nice": (
                cpu_config.background_nice_increment
            ),
            "limix_cpu_ids": ",".join(map(str, cpu_config.limix_cpu_ids)),
            "limix_count": allocation.limix_count,
            "student_count": allocation.student_count,
            "limix_ratio": allocation.limix_ratio,
            "expected_accuracy": allocation.expected_accuracy,
            "min_expected_accuracy": MIN_EXPECTED_ACCURACY,
            "accuracy_tolerance": EXPECTED_ACCURACY_TOLERANCE,
            "expected_seconds": allocation.expected_seconds,
            "profile_mixed_overhead_seconds": profile.mixed_overhead_seconds,
            "partition_seconds": partition_seconds,
            "engine_parallel_seconds": engine_parallel_seconds,
            "merge_seconds": merge_seconds,
            "limix_inference_seconds": limix_inference_seconds,
            "student_inference_seconds": student_inference_seconds,
            "student_update": student_update,
            "old_w0_accuracy": old_w0_accuracy,
            "candidate_w0_accuracy": candidate_w0_accuracy,
            "training_started": training_started_this_window,
            "training_type": training_type_started,
            "training_completed_type": training_completed_type,
            "distillation_windows": distillation_windows,
            "distillation_seconds_estimate": distillation_seconds_estimate,
            "training_seconds_observed": training_seconds_observed,
            "training_seconds_total": training_seconds_total,
            "model_reload_seconds": model_reload_seconds,
            "gpu_utilization": (
                resources.gpu_utilization
            ),
            "gpu_memory_free_mb": (
                resources.gpu_memory_free_mb
            ),
            "cpu_utilization": (
                resources.cpu_utilization
            ),
            "window_seconds": window_seconds,
            "window_throughput": len(target.labels) / window_seconds,
            **metrics,
        }
    
        results.append(result)
    
        pd.DataFrame(results).to_csv(
            result_path,
            index=False,
        )
    
        model_text = (
            f"model={used_student_model}, "
            if used_student_model is not None
            else ""
        )
    
        print(
            f"W{window.window_id}: "
            f"route={route}, "
            f"{model_text}"
            f"limix_ratio={allocation.limix_ratio:.3f}, "
            f"reason={allocation_reason}, "
            f"PSI={drift_signal.score:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"latency={inference_seconds:.4f}s"
        )
    
        # Simulate a fixed window arrival interval when replaying a static dataset.
        # Record this interval as an experiment setting, not a measured real-world arrival rate.
        remaining_interval = WINDOW_INTERVAL_SECONDS - window_seconds
        if remaining_interval > 0:
            time.sleep(remaining_interval)
    
        
    # Wait for the final background task and propagate training failures.
    if training_future is not None:
        completed_training_name = training_name
        (
            candidate,
            training_seconds_observed,
            training_interval_start,
            training_interval_end,
        ) = materialize_background_result(training_future.result())
        training_seconds_total += training_seconds_observed
        training_intervals.append(
            (training_interval_start, training_interval_end)
        )
        training_completed_type = (
            "distillation"
            if completed_training_name is not None
            and completed_training_name.startswith("replay_student")
            else "initial"
        )
        candidate_w0_accuracy = measure_w0_accuracy(candidate)
        student_update = "accepted"
        if (
            completed_training_name is not None
            and completed_training_name.startswith("replay_student")
        ):
            adaptation_pending = False
            distillation_seconds_estimate = (
                0.8 * distillation_seconds_estimate
                + 0.2 * training_seconds_observed
            )
            old_w0_accuracy = student_w0_accuracy
            if candidate_w0_accuracy < old_w0_accuracy - MAX_W0_ACCURACY_DROP:
                student_update = "rejected"
            else:
                replay_stu = candidate
                replay_ready_window = len(data_stream)
                student = candidate
                stu_name = training_name
                student_w0_accuracy = candidate_w0_accuracy
                student_accuracy_estimate = profile.future_student.accuracy
                student_throughput_estimate = profile.future_student.throughput
                student_startup_seconds_estimate = (
                    profile.future_student.startup_seconds
                )
                student_needs_recovery = False
        else:
            student = candidate
            stu_name = training_name
            student_w0_accuracy = candidate_w0_accuracy
            student_accuracy_estimate = profile.stable_student.accuracy
            student_throughput_estimate = profile.stable_student.throughput
            student_startup_seconds_estimate = (
                profile.stable_student.startup_seconds
            )
    
        print(
            f"{training_name} training completed, "
            f"update={student_update}, "
            f"active_tree_model={student.model_name}, "
            f"candidate_W0_accuracy={candidate_w0_accuracy:.4f}, "
            f"training_seconds={training_seconds_observed:.2f}, "
            f"training_seconds_total={training_seconds_total:.2f}"
        )

        if results:
            results[-1]["training_completed_type"] = training_completed_type
            results[-1]["training_seconds_observed"] = training_seconds_observed
            results[-1]["training_seconds_total"] = training_seconds_total
            results[-1]["adaptation_pending"] = adaptation_pending

    # Exclude the final incomplete window from formal metrics, matching the aggregate reporting script.
    formal_inference_intervals = inference_intervals[:-1]
    inference_seconds_total = sum(
        end - start for start, end in formal_inference_intervals
    )
    total_seconds = merged_duration(
        training_intervals + formal_inference_intervals
    )
    overlap_seconds = max(
        0.0,
        training_seconds_total + inference_seconds_total - total_seconds,
    )

    for result in results:
        result["inference_seconds_total"] = inference_seconds_total
        result["training_inference_overlap_seconds"] = overlap_seconds
        result["total_seconds"] = total_seconds

    if results:
        pd.DataFrame(results).to_csv(result_path, index=False)
    
    
    training_executor.shutdown(wait=True)
    inference_executor.shutdown(wait=True)

    print(
        "DUET timing summary: "
        f"search_training_seconds={training_seconds_total:.2f} s, "
        f"inference_seconds={inference_seconds_total:.2f} s, "
        f"overlap_seconds={overlap_seconds:.2f} s, "
        f"total_seconds={total_seconds:.2f} s"
    )
    
    if(
        runtime_config.run_student_comparison
        and
        initial_stu is not None 
        and replay_stu is not None
        and replay_ready_window is not None 
    ):
        print("Training the W0-only retraining control")
        # Write the requested old/LimiX/distilled comparison before running
        # the optional W0 control fit.  A failure in that extra fit must not
        # erase the already valid distillation summary.
        compare_students(
            initial_stu,
            None,
            replay_stu,
            limix_model,
            replay_ready_window,
        )
        # Reset the process-global Ray session left by the background fit.
        shutdown_ray()
        w0_retrain_start = time.perf_counter()
        with cpu_affinity_scope(cpu_config.training_cpu_ids):
            w0_retrain_stu = train_w0_retrain_student()
        w0_retrain_seconds = time.perf_counter() - w0_retrain_start
    
        w0_retrain_labels = bootstrap.classes[
            w0_retrain_stu.predict_proba(
                bootstrap.val_features
            ).argmax(axis=1)
        ]
        w0_retrain_accuracy = accuracy_score(
            bootstrap.val_labels,
            w0_retrain_labels,
        )
        print(
            f"W0-only control training completed, model={w0_retrain_stu.model_name}, "
            f"W0_accuracy={w0_retrain_accuracy:.4f}, "
            f"training_seconds={w0_retrain_seconds:.2f}"
        )
    
    if student_pool is not None:
        student_pool.close()
    limix_model.close()
    shutdown_ray()
    
    print(f"Experiment complete; results saved to: {result_path}")
    return result_path


if __name__ == "__main__":
    main()

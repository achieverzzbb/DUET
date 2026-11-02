"""Generate offline model profiles for adaptive scheduling."""

from pathlib import Path
import os
import sys
import time

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.builder import StreamBuilder
from src.data.datasets import get_dataset_spec
from src.data.stream import DataStream
from src.data.types import StreamConfig
from src.experiment_runtime import (
    AutoGluonRuntimeConfig,
    ExperimentRuntimeConfig,
)
from src.engines.tbe.autogluon import (
    AUTOGLUON_CPUS_PER_MODEL,
    AUTOGLUON_FIT_STRATEGY,
    AUTOGLUON_MODEL_KEYS,
    AUTOGLUON_REQUIRED_PARALLEL_CPUS,
    AutoGluonStudent,
)
from src.engines.tbe.autogluon_parallel import AutoGluonPool
from src.engines.tbe.distillation import distill_replay_student
from src.engines.fbe.limix_parallel import LimiXPool
from src.system.cpu_resources import (
    CPUResourceConfig,
    cpu_affinity_scope,
    format_cpu_sets,
    get_cpu_resource_config,
    pin_current_process,
)
from src.system.profiling import (
    OfflineModelProfile,
    SchedulingProfile,
    conservative_accuracy,
    measure_latency_profile,
    measure_mixed_overhead,
    measure_window_accuracies,
    save_profile,
)


DATASET_NAME = os.getenv("DUET_DATASET", "particle")
print(f"Dataset: {DATASET_NAME}\n")
TRAIN_TIME_LIMIT = int(os.getenv("DUET_TRAIN_TIME_LIMIT", "60"))
DISTILL_SHIFT_WEIGHT = 0.6
DISTILL_WINDOW_COUNT = 2
ACCURACY_QUANTILE = 0.1
CALIBRATION_SEED = 3026
LATENCY_BATCH_SIZES = (400, 2000, 4000, 8000)
LIMIX_GPU_IDS = tuple(range(torch.cuda.device_count()))


def create_student_pool(
    model: AutoGluonStudent,
    cpu_config: CPUResourceConfig | None = None,
) -> AutoGluonPool:
    """Deploy the student with the same CPU data-parallel settings as the main experiment."""
    if model.model_name is None or model.classes is None:
        raise RuntimeError("The student model has not finished training")

    cpu_config = cpu_config or get_cpu_resource_config()
    pool = AutoGluonPool(
        model_path=model.model_path,
        model_name=model.model_name,
        classes=model.classes,
        cpu_sets=cpu_config.inference_cpu_sets,
        threads_per_worker=cpu_config.inference_threads_per_worker,
    )
    try:
        pool.initialize()
    except Exception:
        pool.close()
        raise
    return pool


def main(
    runtime_config: ExperimentRuntimeConfig | None = None,
    cpu_config: CPUResourceConfig | None = None,
    limix_gpu_ids: tuple[int, ...] | None = None,
    data_stream: DataStream | None = None,
    profile_path: Path | None = None,
    run_dir: Path | None = None,
    latency_batch_sizes: tuple[int, ...] = LATENCY_BATCH_SIZES,
    calibration_seed: int = CALIBRATION_SEED,
) -> Path:
    """Generate one scheduler profile with optional explicit configuration."""
    if runtime_config is None:
        stream_config = StreamConfig(
            bootstrap_size=int(os.getenv("DUET_BOOTSTRAP_SIZE", "500")),
            bootstrap_val_ratio=0.2,
            window_size=int(os.getenv("DUET_WINDOW_SIZE", "2000")),
            seed=calibration_seed,
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
    gpu_ids = LIMIX_GPU_IDS if limix_gpu_ids is None else tuple(limix_gpu_ids)
    if not gpu_ids:
        raise RuntimeError("Profiling requires at least one LimiX GPU")
    # Profile the same lightweight student that DUET will use online.  The
    # standalone baselines may intentionally use a heavier AutoGluon config.
    model_config = runtime_config.student_autogluon_config()
    dataset_name = runtime_config.dataset_name
    train_time_limit = runtime_config.train_time_limit
    pin_current_process(cpu_config.limix_cpu_ids)
    print(
        "CPU resource isolation: "
        "student_config=lightweight, "
        f"inference={format_cpu_sets(cpu_config.inference_cpu_sets)}, "
        f"training={','.join(map(str, cpu_config.training_cpu_ids))}, "
        f"training_strategy={model_config.fit_strategy}, "
        f"candidates={len(model_config.model_keys)}, "
        f"cpus_per_candidate={model_config.cpus_per_model}, "
        f"preset={model_config.presets}, "
        f"bag_folds={model_config.num_bag_folds}, "
        f"parallel_cpus={len(model_config.model_keys) * model_config.cpus_per_model}, "
        f"limix={','.join(map(str, cpu_config.limix_cpu_ids))}"
    )
    print("[1/5] Building the offline calibration stream")
    dataset_spec = get_dataset_spec(dataset_name)
    config = runtime_config.stream
    print(f"Data stream shift: {config.shift_summary()}")
    if data_stream is None:
        data_stream = StreamBuilder(dataset_spec, config).build()
    bootstrap = data_stream.bootstrap()
    windows = list(data_stream)

    stable_windows = [
        window
        for window in windows
        if window.window_id < config.shift_start_window
    ]
    shifted_windows = [
        window
        for window in windows
        if window.window_id >= config.shift_start_window
    ]
    distill_end = runtime_config.distill_window_count
    distill_windows = shifted_windows[:distill_end]
    evaluation_windows = shifted_windows[distill_end:]
    if (
        not stable_windows
        or len(distill_windows) < runtime_config.distill_window_count
    ):
        raise ValueError("The calibration stream has insufficient stable or distillation windows")
    if not evaluation_windows:
        raise ValueError("At least one independent calibration window is required after distillation windows")

    labels_by_window = {
        window.window_id: data_stream.eval_target(window.window_id).labels
        for window in windows
    }

    if run_dir is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
        run_dir = PROJECT_ROOT / "output" / f"profiling_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    limix = LimiXPool(
        model_path=PROJECT_ROOT / "cache" / "LimiX-2M.ckpt",
        config_path=PROJECT_ROOT / "config" / "cls_default_noretrieval.json",
        gpu_ids=gpu_ids,
        seed=calibration_seed,
        cpu_ids=cpu_config.limix_cpu_ids,
        max_shard_size=runtime_config.limix_max_shard_size,
        warmup_sample_count=runtime_config.limix_warmup_sample_count,
    )
    limix.initialize(bootstrap)
    limix.predict_proba(bootstrap.val_features.iloc[:32])

    print("[2/5] Initializing LimiX and training the initial student")
    initial_student = AutoGluonStudent(
        model_path=run_dir / "initial_student",
        problem_type=dataset_spec.problem_type,
        time_limit=train_time_limit,
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
    with cpu_affinity_scope(cpu_config.training_cpu_ids):
        initial_student.fit(bootstrap)
    initial_pool = create_student_pool(initial_student, cpu_config)
    initial_pool.predict_proba(bootstrap.val_features.iloc[:32])

    print("[3/5] Measuring estimated model accuracies")
    print("  measuring accuracy profiles...", flush=True)
    limix_accuracies = measure_window_accuracies(
        limix.predict_proba,
        evaluation_windows,
        labels_by_window,
        bootstrap.classes,
    )
    stable_student_accuracies = measure_window_accuracies(
        initial_pool.predict_proba,
        stable_windows,
        labels_by_window,
        bootstrap.classes,
    )
    shifted_student_accuracies = measure_window_accuracies(
        initial_pool.predict_proba,
        evaluation_windows,
        labels_by_window,
        bootstrap.classes,
    )

    shift_features = pd.concat(
        [window.features for window in distill_windows],
        ignore_index=True,
    )
    teacher_probability_parts = []
    for window in distill_windows:
        teacher_probability_parts.append(
            limix.predict_proba(window.features)
        )
    teacher_probabilities = np.concatenate(
        teacher_probability_parts,
        axis=0,
    )

    print("[4/5] Training the replay student and measuring distillation time")
    print(
        f"  distillation fit: samples={len(shift_features)}",
        flush=True,
    )
    distillation_start = time.perf_counter()
    with cpu_affinity_scope(cpu_config.training_cpu_ids):
        future_student = distill_replay_student(
            bootstrap=bootstrap,
            shift_features=shift_features,
            teacher_probabilities=teacher_probabilities,
            model_path=run_dir / "future_student",
            problem_type=dataset_spec.problem_type,
            shift_weight=runtime_config.distill_shift_weight,
            time_limit=train_time_limit,
            seed=calibration_seed,
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
    distillation_seconds = time.perf_counter() - distillation_start
    future_pool = create_student_pool(future_student, cpu_config)
    future_pool.predict_proba(bootstrap.val_features.iloc[:32])

    future_student_accuracies = measure_window_accuracies(
        future_pool.predict_proba,
        evaluation_windows,
        labels_by_window,
        bootstrap.classes,
    )

    reference_features = pd.concat(
        [window.features for window in evaluation_windows],
        ignore_index=True,
    )
    batch_sizes = tuple(
        size
        for size in latency_batch_sizes
        if size <= len(reference_features)
    )
    if len(batch_sizes) < 2:
        raise ValueError("The calibration stream has insufficient samples for latency fitting")

    print("[5/5] Fitting inference latency and saving offline profiles")
    limix_startup, limix_throughput = measure_latency_profile(
        limix.predict_proba,
        reference_features,
        batch_sizes,
    )
    initial_startup, initial_throughput = measure_latency_profile(
        initial_pool.predict_proba,
        reference_features,
        batch_sizes,
    )
    future_startup, future_throughput = measure_latency_profile(
        future_pool.predict_proba,
        reference_features,
        batch_sizes,
    )

    limix_profile = OfflineModelProfile(
        accuracy=conservative_accuracy(
            limix_accuracies,
            ACCURACY_QUANTILE,
        ),
        startup_seconds=limix_startup,
        throughput=limix_throughput,
    )
    stable_student_profile = OfflineModelProfile(
        accuracy=conservative_accuracy(
            stable_student_accuracies,
            ACCURACY_QUANTILE,
        ),
        startup_seconds=initial_startup,
        throughput=initial_throughput,
    )
    shifted_student_profile = OfflineModelProfile(
        accuracy=conservative_accuracy(
            shifted_student_accuracies,
            ACCURACY_QUANTILE,
        ),
        startup_seconds=initial_startup,
        throughput=initial_throughput,
    )
    future_student_profile = OfflineModelProfile(
        accuracy=conservative_accuracy(
            future_student_accuracies,
            ACCURACY_QUANTILE,
        ),
        startup_seconds=future_startup,
        throughput=future_throughput,
    )

    mixed_batch_size = min(config.window_size, len(reference_features))
    print(
        f"  Mixed overhead: initial student, samples={mixed_batch_size}",
        flush=True,
    )
    initial_mixed_overhead = measure_mixed_overhead(
        limix.predict_proba,
        initial_pool.predict_proba,
        reference_features,
        mixed_batch_size,
        limix_profile,
        stable_student_profile,
    )
    print(
        f"  Mixed overhead: distilled student, samples={mixed_batch_size}",
        flush=True,
    )
    future_mixed_overhead = measure_mixed_overhead(
        limix.predict_proba,
        future_pool.predict_proba,
        reference_features,
        mixed_batch_size,
        limix_profile,
        future_student_profile,
    )
    mixed_overhead_seconds = max(
        initial_mixed_overhead,
        future_mixed_overhead,
    )

    profile = SchedulingProfile(
        dataset_name=dataset_name,
        calibration_seed=calibration_seed,
        accuracy_quantile=ACCURACY_QUANTILE,
        limix=limix_profile,
        stable_student=stable_student_profile,
        shifted_student=shifted_student_profile,
        future_student=future_student_profile,
        distillation_seconds=distillation_seconds,
        mixed_overhead_seconds=mixed_overhead_seconds,
    )

    if profile_path is None:
        profile_path = (
            PROJECT_ROOT / "output" / "profiles" / f"{dataset_name}.json"
        )
    save_profile(profile, profile_path)

    accuracy_rows = []
    groups = (
        ("limix_shifted", evaluation_windows, limix_accuracies),
        ("student_stable", stable_windows, stable_student_accuracies),
        ("student_shifted", evaluation_windows, shifted_student_accuracies),
        ("future_student", evaluation_windows, future_student_accuracies),
    )
    for model_state, model_windows, accuracies in groups:
        accuracy_rows.extend(
            {
                "model_state": model_state,
                "window_id": window.window_id,
                "accuracy": accuracy,
            }
            for window, accuracy in zip(model_windows, accuracies)
        )
    pd.DataFrame(accuracy_rows).to_csv(
        run_dir / "window_accuracies.csv",
        index=False,
    )

    print(f"Offline scheduling profiles saved to: {profile_path}")
    print(
        "Estimated accuracy: "
        f"LimiX={profile.limix.accuracy:.4f}, "
        f"stable_student={profile.stable_student.accuracy:.4f}, "
        f"shifted_student={profile.shifted_student.accuracy:.4f}, "
        f"distilled_student={profile.future_student.accuracy:.4f}"
    )
    print(f"Estimated distillation time: {profile.distillation_seconds:.2f} s")
    print(f"Mixed-execution overhead: {profile.mixed_overhead_seconds:.4f} s")

    future_pool.close()
    initial_pool.close()
    limix.close()
    return profile_path


if __name__ == "__main__":
    main()

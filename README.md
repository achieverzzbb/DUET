# DUET

DUET is a dual-engine system for streaming tabular inference. It coordinates a
GPU-based foundation-model engine (LimiX) and a CPU-based tree-model engine
(AutoGluon) through accuracy-constrained scheduling and drift-triggered
asynchronous replay distillation.

## Code Structure

```text
DUET-release/
├── main.py                     # Main experiment entry point
├── src/
│   ├── main.py                 # Streaming workflow and scheduling integration
│   ├── profile_models.py       # Offline engine profiling
│   ├── experiment_runtime.py   # Runtime settings
│   ├── engines/
│   │   ├── fbe/                # Foundation-model engine and multi-GPU workers
│   │   └── tbe/                # Tree-model engine and replay distillation
│   ├── data/                   # Dataset loading and stream construction
│   ├── detection/              # Feature-only PCA/PSI drift detection
│   ├── system/                 # Scheduling, profiling, and resource management
│   └── evaluation/             # Evaluation metrics
├── inference/                  # LimiX inference implementation
├── model/                      # LimiX model implementation
├── utils/                      # LimiX utilities
├── config/                     # LimiX inference configuration
├── cache/                      # Downloaded model weights
├── dataset/                    # Input datasets
└── requirements.txt
```

## Environment Setup

Use Linux, Python 3.12, an NVIDIA GPU with a CUDA 12.8-compatible driver, and at
least 8 available CPU threads for the commands below. Run all commands from
the project root.

```bash
conda create -n duet python=3.12 -y
conda activate duet
python -m pip install -r requirements.txt
hf download stableai-org/LimiX-2M LimiX-2M.ckpt --local-dir cache
```

The [LimiX weights](https://huggingface.co/stableai-org/LimiX-2M) must be downloaded
to `cache/LimiX-2M.ckpt`.

## Main Experiment

Set the experiment parameters, generate the engine profiles, and run DUET:

```bash
export CUDA_VISIBLE_DEVICES=0
export DUET_DATASET=skin
export DUET_BOOTSTRAP_SIZE=6000
export DUET_WINDOW_SIZE=6000
export DUET_SHIFT_START_WINDOW=16
export DUET_TRAIN_TIME_LIMIT=60
export DUET_AUTOGLUON_FIT_STRATEGY=sequential
export DUET_AUTOGLUON_TRAIN_CPUS=4
export DUET_AUTOGLUON_CPUS_PER_MODEL=4

python -u src/profile_models.py
python -u main.py
```

`DUET_SHIFT_START_WINDOW` controls synthetic workload construction only;
online PCA/PSI alerts trigger replay-window collection.

Engine profiles are saved to `output/profiles/skin.json`, and experiment
results to `output/stream_*/results.csv`.

""" Collect CPU, memory, and GPU resource status. """

from dataclasses import dataclass
import subprocess
import warnings

import psutil
import torch

_nvidia_smi_disabled = False

@dataclass(frozen=True)
class ResourceStatus:
    # GPU 
    gpu_available: bool                 # Whether a GPU is available.
    gpu_name: str | None                # GPU model name.
    gpu_utilization: float | None       # GPU utilization.
    gpu_memory_used_mb: float | None    # GPU memory used.
    gpu_memory_free_mb: float | None    # Free GPU memory.

    # CPU
    cpu_utilization: float              # CPU utilization.

    # Memory.
    memory_available_mb: float


def get_resource_status(
    gpu_index: int = 0,
    cpu_sample_seconds: float = 0.2,
) -> ResourceStatus:
    """ Collect one resource snapshot. """

    global _nvidia_smi_disabled

    cpu_utilization = psutil.cpu_percent(interval=cpu_sample_seconds)

    memory_available_mb = psutil.virtual_memory().available / 1024 / 1024

    if not torch.cuda.is_available():
        return ResourceStatus(
            gpu_available=False,
            gpu_name=None,
            gpu_utilization=None,
            gpu_memory_used_mb=None,
            gpu_memory_free_mb=None,
            cpu_utilization=cpu_utilization,
            memory_available_mb=memory_available_mb,
        )


    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=name,utilization.gpu,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]

    if not _nvidia_smi_disabled:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            gpu_name, utilization, memory_used, memory_free = [
                value.strip()
                for value in result.stdout.strip().split(',')
            ]
            return ResourceStatus(
                gpu_available=True,
                gpu_name=gpu_name,
                gpu_utilization=float(utilization),
                gpu_memory_used_mb=float(memory_used),
                gpu_memory_free_mb=float(memory_free),
                cpu_utilization=cpu_utilization,
                memory_available_mb=memory_available_mb,
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            # Telemetry failure must not disable a working inference engine.
            _nvidia_smi_disabled = True
            warnings.warn(
                f"nvidia-smi status query failed ({error}). "
                "Using PyTorch CUDA memory status; GPU utilization is unknown. "
                "Further nvidia-smi queries are disabled for this process.",
                RuntimeWarning,
                stacklevel=2,
            )

    memory_free_bytes, memory_total_bytes = torch.cuda.mem_get_info(gpu_index)

    return ResourceStatus(
        gpu_available=True,
        gpu_name=torch.cuda.get_device_name(gpu_index),
        gpu_utilization=None,
        gpu_memory_used_mb=(memory_total_bytes - memory_free_bytes) / 1024 / 1024,
        gpu_memory_free_mb=memory_free_bytes / 1024 / 1024,
        cpu_utilization=cpu_utilization,
        memory_available_mb=memory_available_mb,
    )

if __name__ == "__main__":
    print(get_resource_status(
        gpu_index=0,
        cpu_sample_seconds=0.2,
    ))

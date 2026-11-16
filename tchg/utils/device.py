from __future__ import annotations

import torch


def get_device(device: str | None = None) -> torch.device:

    name = device or "auto"
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(name)


def gpu_memory_mb(device: torch.device) -> float:

    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

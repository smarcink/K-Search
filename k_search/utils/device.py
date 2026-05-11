"""Shared device abstraction for XPU/CUDA/CPU backends."""

from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> str:
    """Return the best available device string.

    Args:
        preferred: If given, validate and return it (e.g. "xpu:0", "cuda:0").
                   Falls back to auto-detect if not available.
    """
    if preferred:
        backend = preferred.split(":")[0]
        if backend == "xpu" and torch.xpu.is_available():
            return preferred
        if backend == "cuda" and torch.cuda.is_available():
            return preferred
        if backend == "cpu":
            return preferred
        # Requested device not available — fall through to auto-detect.

    if torch.xpu.is_available():
        return "xpu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_device_name(device: str | torch.device) -> str:
    """Return the human-readable device name."""
    dev = str(device)
    backend = dev.split(":")[0]
    idx = int(dev.split(":")[1]) if ":" in dev else 0
    if backend == "xpu":
        return torch.xpu.get_device_name(idx)
    if backend == "cuda":
        return torch.cuda.get_device_name(idx)
    return "cpu"


def synchronize(device: str | torch.device) -> None:
    """Synchronize the given device."""
    backend = str(device).split(":")[0]
    if backend == "xpu":
        torch.xpu.synchronize()
    elif backend == "cuda":
        torch.cuda.synchronize()


def create_event(device: str | torch.device, enable_timing: bool = True):
    """Create a device Event for timing."""
    backend = str(device).split(":")[0]
    if backend == "xpu":
        return torch.xpu.Event(enable_timing=enable_timing)
    if backend == "cuda":
        return torch.cuda.Event(enable_timing=enable_timing)
    raise ValueError(f"Timing events not supported on device: {device}")

"""Device selection for Apple Silicon (MPS), CUDA, or CPU.

This project is developed on an Apple M3 Pro where the accelerator is Metal/MPS,
not CUDA. The helper prefers CUDA when present (e.g. if run later on Colab/Kaggle),
then MPS, then CPU, so the same code path runs unchanged across environments.
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Args:
        prefer: Optional explicit device string ("cuda", "mps", "cpu"). If given and
            available, it is used; otherwise we auto-select.

    Returns:
        A ``torch.device``. Auto-selection order: CUDA > MPS > CPU.
    """
    if prefer:
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    """Human-readable description of a device for logging.

    Args:
        device: A ``torch.device``.

    Returns:
        A short string such as "cuda (NVIDIA A100)" or "mps (Apple Silicon)".
    """
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple Silicon / Metal)"
    return "cpu"

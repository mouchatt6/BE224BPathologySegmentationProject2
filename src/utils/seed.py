"""Reproducibility helpers: seed every RNG that the pipeline touches."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: Integer seed applied to ``random``, ``numpy`` and ``torch``.
        deterministic: If True, force deterministic cuDNN/algorithms. Slower, but
            required for bit-for-bit reproducible final runs. On MPS/CPU this mainly
            sets the deterministic-algorithms flag and the ``PYTHONHASHSEED`` env var.

    Returns:
        None.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # use_deterministic_algorithms can raise on ops without a deterministic
        # implementation; warn_only keeps final runs from crashing on MPS/CPU.
        torch.use_deterministic_algorithms(True, warn_only=True)

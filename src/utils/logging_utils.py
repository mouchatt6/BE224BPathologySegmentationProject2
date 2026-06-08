"""Logging setup: every run writes to both stdout and a per-experiment log file.

Per the project coding standards, each experiment logs to
``outputs/logs/<experiment_name>.log`` while also streaming to the console, so a
run can be followed live and audited afterwards.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.config import REPO_ROOT


def get_logger(experiment_name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Create (or fetch) a logger that writes to stdout and a file.

    Args:
        experiment_name: Used as the logger name and the log filename stem.
        log_dir: Directory for the log file. Defaults to ``outputs/logs``.

    Returns:
        A configured ``logging.Logger``. Calling twice with the same name returns the
        same logger without adding duplicate handlers.
    """
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # Guard against duplicate handlers if get_logger is called more than once
    # (e.g. across folds) — otherwise every log line would print N times.
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    # Console handler.
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler.
    log_dir = Path(log_dir) if log_dir else (REPO_ROOT / "outputs" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{experiment_name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

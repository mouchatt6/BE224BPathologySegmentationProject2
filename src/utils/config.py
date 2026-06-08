"""Config + path utilities.

Centralizes (a) where the repo root is, so scripts can resolve ``data/`` and
``outputs/`` regardless of the current working directory, and (b) loading the YAML
experiment configs. Keeping all path logic here means no script hardcodes an
absolute path — important because the data lives *outside* the git repo and the
same configs must also work on Colab/Kaggle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# This file is <repo>/src/utils/config.py, so the repo root is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path) -> Path:
    """Resolve a path that may be relative to the repo root.

    Absolute paths are returned unchanged; relative paths (e.g. "data/train" from a
    config) are interpreted relative to the repo root, not the current working dir.

    Args:
        p: A path string or Path, absolute or repo-relative.

    Returns:
        An absolute ``Path``.
    """
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config into a dict.

    Args:
        config_path: Path to a ``.yaml`` config file.

    Returns:
        The parsed config as a plain dict.
    """
    with open(resolve_path(config_path), "r") as f:
        return yaml.safe_load(f)


def config_hash(config: dict[str, Any], length: int = 8) -> str:
    """Short, stable hash of a config dict for tagging checkpoints/outputs.

    Args:
        config: The config dict.
        length: Number of hex characters to keep.

    Returns:
        A truncated SHA-1 hex digest of the canonical JSON form of the config.
    """
    # sort_keys makes the hash invariant to key ordering in the YAML file.
    canonical = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(canonical).hexdigest()[:length]

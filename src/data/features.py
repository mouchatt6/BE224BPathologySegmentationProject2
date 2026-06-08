"""Load/concatenate the cached frozen-backbone features.

The feature-extraction script writes one ``.npy`` per (backbone, split) plus a paths
CSV recording row order. Here we reassemble them into the single concatenated feature
matrix the MLP head consumes, asserting that all backbones share the same row order so
features never get misaligned with labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import REPO_ROOT


def features_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return the feature cache directory (default ``outputs/features``)."""
    return Path(cache_dir) if cache_dir else (REPO_ROOT / "outputs" / "features")


def load_concat_features(
    split: str,
    backbones: list[str],
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load and horizontally concatenate cached features for one split.

    Args:
        split: "train" or "test".
        backbones: Backbone names in the canonical concatenation order
            (e.g. ["resnet18", "resnet34", "efficientnet_b0"]).
        cache_dir: Where the ``.npy`` files live. Defaults to ``outputs/features``.

    Returns:
        (X, paths):
          * X: float32 array of shape (N, sum(feature_dims)) — features concatenated
            across backbones in the given order.
          * paths: list of N img_path strings in the same row order as X.
    """
    cache = features_cache_dir(cache_dir)

    # Row order is defined by the paths CSV; every backbone must match it.
    paths_csv = cache / f"{split}_paths.csv"
    paths = pd.read_csv(paths_csv)["img_path"].tolist()

    mats: list[np.ndarray] = []
    for name in backbones:
        arr = np.load(cache / f"{name}_{split}.npy")
        # Guard against a stale cache where a backbone has a different row count.
        assert arr.shape[0] == len(paths), (
            f"{name}_{split}.npy has {arr.shape[0]} rows but {len(paths)} paths expected"
        )
        mats.append(arr)

    X = np.concatenate(mats, axis=1).astype(np.float32)
    return X, paths

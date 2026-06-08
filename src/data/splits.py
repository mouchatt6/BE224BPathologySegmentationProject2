"""Cross-validation split construction.

Why StratifiedKFold (and not GroupKFold): the test for slide/patient leakage is
whether ``img_path`` encodes a slide or patient id that could place correlated
patches in both train and val. Here the filenames are bare sequential integers
(``0000.png`` ...), so there is no group structure to leak on. We therefore stratify
by label only, which also keeps the 50/50 class balance identical across folds.

The fold assignment is computed ONCE and cached to ``outputs/folds.csv`` so every
script (feature extraction, training, OOF assembly) uses the exact same split. Never
re-randomize between runs — that would invalidate OOF comparisons across experiments.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils.config import REPO_ROOT, resolve_path


def make_stratified_folds(
    train_csv: str | Path,
    n_splits: int = 5,
    seed: int = 42,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Assign each training patch to a stratified CV fold and cache the result.

    Args:
        train_csv: Path to ``train.csv`` (columns: img_path, label).
        n_splits: Number of CV folds (5 → ~5,920 train / 1,480 val per fold).
        seed: RNG seed for the fold shuffling, for reproducible splits.
        out_path: Where to write the fold table. Defaults to ``outputs/folds.csv``.

    Returns:
        DataFrame with columns ["img_path", "label", "fold"] where ``fold`` is the
        validation-fold index in [0, n_splits) that each row belongs to.
    """
    df = pd.read_csv(resolve_path(train_csv))
    # Default integer fold column; every row gets overwritten in the loop below.
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # split() yields (train_idx, val_idx) per fold; we only need the val indices,
    # because a row's "fold" is the fold in which it serves as validation data.
    for fold_idx, (_, val_idx) in enumerate(skf.split(df["img_path"], df["label"])):
        df.loc[val_idx, "fold"] = fold_idx

    # Sanity: every row must have been assigned to exactly one fold.
    assert (df["fold"] >= 0).all(), "Some rows were not assigned a fold."

    out_path = Path(out_path) if out_path else (REPO_ROOT / "outputs" / "folds.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_folds(folds_csv: str | Path | None = None) -> pd.DataFrame:
    """Load the cached fold table, erroring clearly if it doesn't exist yet.

    Args:
        folds_csv: Path to the fold CSV. Defaults to ``outputs/folds.csv``.

    Returns:
        DataFrame with columns ["img_path", "label", "fold"].
    """
    folds_csv = Path(folds_csv) if folds_csv else (REPO_ROOT / "outputs" / "folds.csv")
    if not folds_csv.exists():
        raise FileNotFoundError(
            f"{folds_csv} not found. Run make_stratified_folds() (or scripts/extract_features.py) first."
        )
    return pd.read_csv(folds_csv)

"""CPU-only smoke/unit tests for the core pipeline.

Run from the repo root:  PYTHONPATH=. ./.venv/bin/python -m pytest tests/ -q

These exercise the parts that must be correct regardless of the backbone — the Kaggle
scorer, the submission validator (which the grader fails silently on), the stratified
splits, the stain normalizer, the MLP head, the threshold sweep, and the TTA variants.
No dataset, GPU, or network needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.splits import make_stratified_folds
from src.data.stain import MacenkoNormalizer
from src.eval.metrics import compute_score, report_across_alphas
from src.eval.validate import validate_submission
from src.inference.threshold import optimize_threshold
from src.inference.tta import flip_variants, rot_variants
from src.models.heads import MLPHead


# --------------------------------------------------------------------------- scorer
def test_compute_score_perfect_separation():
    """A perfectly-ranked, correctly-labeled set scores AUROC=1, F1=1, composite=1."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    label = (proba >= 0.5).astype(int)
    comp, auroc, f1 = compute_score(y, label, proba, alpha=0.5)
    assert auroc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert comp == pytest.approx(0.5 * auroc + 0.5 * f1)


def test_compute_score_alpha_weighting():
    """Composite is exactly alpha*AUROC + (1-alpha)*F1."""
    y = np.array([0, 1, 0, 1, 1, 0])
    proba = np.array([0.3, 0.6, 0.45, 0.9, 0.55, 0.2])
    label = (proba >= 0.5).astype(int)
    for a in (0.2, 0.5, 0.8):
        comp, auroc, f1 = compute_score(y, label, proba, alpha=a)
        assert comp == pytest.approx(a * auroc + (1 - a) * f1)


def test_report_across_alphas_keys():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.95])
    rep = report_across_alphas(y, (proba >= 0.5).astype(int), proba, alphas=(0.3, 0.5, 0.7))
    assert {"auroc", "f1", "score@0.3", "score@0.5", "score@0.7"} <= set(rep)


# ------------------------------------------------------------------- submission guard
def _make_valid_pair(n: int = 2400):
    """Build a matching (submission, dummy) pair of the required size."""
    paths = [f"test/{i:04d}.png" for i in range(n)]
    dummy = pd.DataFrame({"img_path": paths, "label": 0, "probabilities": 0.0})
    rng = np.random.default_rng(0)
    sub = pd.DataFrame({
        "img_path": paths,
        "label": rng.integers(0, 2, n).astype(int),
        "probabilities": rng.random(n),
    })
    return sub, dummy


def test_validate_submission_accepts_valid():
    sub, dummy = _make_valid_pair()
    assert validate_submission(sub, dummy) is True


@pytest.mark.parametrize("mutate", ["float_label", "col_order", "prob_oob", "row_count", "reorder"])
def test_validate_submission_rejects_bad(mutate):
    """Each silent-failure mode the grader cares about must raise AssertionError."""
    sub, dummy = _make_valid_pair()
    if mutate == "float_label":
        sub["label"] = sub["label"].astype(float)
    elif mutate == "col_order":
        sub = sub[["label", "img_path", "probabilities"]]
    elif mutate == "prob_oob":
        sub.loc[0, "probabilities"] = 1.5
    elif mutate == "row_count":
        sub = sub.iloc[:-1]
    elif mutate == "reorder":
        sub["img_path"] = sub["img_path"].values[::-1]
    with pytest.raises(AssertionError):
        validate_submission(sub, dummy)


# ------------------------------------------------------------------------- CV splits
def test_make_stratified_folds(tmp_path):
    """Folds cover every row exactly once and preserve label balance."""
    n, n_splits = 100, 5
    csv = tmp_path / "train.csv"
    pd.DataFrame({
        "img_path": [f"train/{i:04d}.png" for i in range(n)],
        "label": [0, 1] * (n // 2),
    }).to_csv(csv, index=False)

    df = make_stratified_folds(str(csv), n_splits=n_splits, seed=42, out_path=tmp_path / "folds.csv")
    assert sorted(df["fold"].unique()) == list(range(n_splits))
    assert (df["fold"] >= 0).all()
    counts = df["fold"].value_counts()
    assert counts.min() == counts.max() == n // n_splits          # balanced fold sizes
    for _, g in df.groupby("fold"):
        assert g["label"].mean() == pytest.approx(0.5)            # stratification preserved


# ---------------------------------------------------------------- stain normalization
def test_macenko_returns_uint8_image():
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    out = MacenkoNormalizer().normalize(patch)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_macenko_blank_patch_is_noop():
    """A near-white (background) patch can't be stain-estimated -> returned unchanged."""
    blank = np.full((224, 224, 3), 245, np.uint8)
    out = MacenkoNormalizer().normalize(blank)
    assert np.array_equal(out, blank)


# ----------------------------------------------------------------------------- head
@pytest.mark.parametrize("in_features", [2304, 768, 1536])
def test_mlp_head_output_shape(in_features):
    """Head maps (B, in_features) -> (B,) for every backbone's feature dim."""
    head = MLPHead(in_features).eval()
    out = head(torch.randn(4, in_features))
    assert out.shape == (4,)


# ------------------------------------------------------------------------- threshold
def test_optimize_threshold():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    proba = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.2, 500), 0, 1)
    out = optimize_threshold(y, proba)
    assert {"tau", "f1", "f1_minus", "f1_plus", "f1_stable"} <= set(out)
    assert 0.05 <= out["tau"] <= 0.95


# ------------------------------------------------------------------------------ TTA
def test_tta_variant_counts_and_shape():
    x = torch.randn(2, 3, 8, 8)
    flips = flip_variants(x)
    rots = rot_variants(x)
    assert len(flips) == 4 and len(rots) == 6
    for v in flips + rots:
        assert v.shape == x.shape

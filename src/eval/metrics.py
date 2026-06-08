"""The single source of truth for local scoring — mirrors the Kaggle metric exactly.

Competition score:  Score = alpha * AUROC + (1 - alpha) * F1   (alpha undisclosed).

Because alpha is unknown, we must do well on BOTH terms:
  * AUROC depends only on the RANKING of the predicted probabilities.
  * F1 depends on the binary labels, hence on the decision threshold tau.
Every other module routes its metric calls through ``compute_score`` so that local
numbers are directly comparable to the leaderboard.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_score(
    y_true: Sequence[int] | np.ndarray,
    y_pred_label: Sequence[int] | np.ndarray,
    y_pred_proba: Sequence[float] | np.ndarray,
    alpha: float = 0.5,
) -> tuple[float, float, float]:
    """Compute the composite competition score and its two components.

    Args:
        y_true: Ground-truth binary labels, shape (N,).
        y_pred_label: Predicted binary labels in {0, 1}, shape (N,). Used for F1.
        y_pred_proba: Predicted positive-class probabilities in [0, 1], shape (N,).
            Used for AUROC (ranking only — the threshold does not affect AUROC).
        alpha: Weight on AUROC in the composite. The true value is undisclosed; we
            report across several alphas elsewhere for a sensitivity analysis.

    Returns:
        (composite, auroc, f1) as floats.
    """
    auroc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred_label)
    composite = alpha * auroc + (1.0 - alpha) * f1
    return float(composite), float(auroc), float(f1)


def report_across_alphas(
    y_true: Sequence[int] | np.ndarray,
    y_pred_label: Sequence[int] | np.ndarray,
    y_pred_proba: Sequence[float] | np.ndarray,
    alphas: Sequence[float] = (0.3, 0.5, 0.7),
) -> dict[str, float]:
    """Score at several alphas to gauge robustness to the unknown weighting.

    If the composite is roughly flat across alphas, the submission is insensitive to
    how the graders weight AUROC vs F1 — exactly what we want to report (§9.2).

    Args:
        y_true: Ground-truth binary labels.
        y_pred_label: Predicted binary labels in {0, 1}.
        y_pred_proba: Predicted positive-class probabilities in [0, 1].
        alphas: The alpha values to evaluate.

    Returns:
        A dict with "auroc", "f1", and "score@{alpha}" for each alpha. AUROC and F1
        are computed once (they do not depend on alpha).
    """
    # AUROC and F1 are alpha-independent, so compute them a single time.
    _, auroc, f1 = compute_score(y_true, y_pred_label, y_pred_proba, alpha=0.5)
    out: dict[str, float] = {"auroc": auroc, "f1": f1}
    for a in alphas:
        out[f"score@{a}"] = a * auroc + (1.0 - a) * f1
    return out

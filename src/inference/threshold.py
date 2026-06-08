"""Decision-threshold optimization for the F1 half of the composite score (§6.1).

AUROC only cares about ranking, but F1 needs a hard 0/1 decision, i.e. a threshold
tau on the predicted probability. The default tau=0.5 usually leaves 1-3 F1 points on
the table. We therefore sweep tau over a fine grid and pick the value that maximizes
F1 — but ALWAYS on out-of-fold (OOF) predictions, never on training-fold data, so the
chosen tau is an honest holdout estimate that transfers to the test set.

We also report a stability check: F1 at tau* +/- 0.02. If F1 is flat there, tau* is
robust; if it cliffs, the threshold is overfit / the model isn't smooth enough.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Find the probability threshold that maximizes F1 on OOF predictions.

    Args:
        y_true: Ground-truth binary labels, shape (N,).
        y_proba: Predicted positive-class probabilities in [0, 1], shape (N,).
        grid: Threshold candidates. Defaults to np.linspace(0.05, 0.95, 181)
            (step 0.005), per §6.1.

    Returns:
        Dict with:
          * "tau"        — the F1-maximizing threshold tau*.
          * "f1"         — F1 at tau*.
          * "f1_minus"   — F1 at tau* - 0.02 (stability check).
          * "f1_plus"    — F1 at tau* + 0.02 (stability check).
          * "f1_stable"  — True if both neighbors stay within 0.01 of the peak F1.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)

    # Evaluate F1 at every candidate threshold and take the argmax.
    f1s = np.array([f1_score(y_true, (y_proba >= t).astype(int)) for t in grid])
    best_idx = int(np.argmax(f1s))
    tau = float(grid[best_idx])
    best_f1 = float(f1s[best_idx])

    # Stability window: recompute F1 a small step on either side of tau*.
    f1_minus = float(f1_score(y_true, (y_proba >= tau - 0.02).astype(int)))
    f1_plus = float(f1_score(y_true, (y_proba >= tau + 0.02).astype(int)))
    f1_stable = bool((best_f1 - min(f1_minus, f1_plus)) <= 0.01)

    return {
        "tau": tau,
        "f1": best_f1,
        "f1_minus": f1_minus,
        "f1_plus": f1_plus,
        "f1_stable": f1_stable,
    }

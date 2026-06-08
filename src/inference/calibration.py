"""Probability calibration via temperature scaling (§6.2).

Temperature scaling fits a single scalar T on the OOF logits to minimize negative
log-likelihood, then divides logits by T before the sigmoid. It does not change the
ranking (so AUROC is unaffected) but tightens probability calibration, which makes the
chosen threshold tau* transfer more reliably to the test set and helps when averaging
models with different output distributions.

For Path A this is optional (it sits on the cut list, §11). It is implemented so it
can be switched on from the config without code changes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Fit a single temperature T minimizing BCE NLL on OOF logits.

    Args:
        logits: OOF logits (pre-sigmoid), shape (N,).
        labels: Ground-truth binary labels, shape (N,).
        max_iter: LBFGS iterations.
        lr: LBFGS learning rate.

    Returns:
        The fitted temperature T (> 0). Divide logits by T before sigmoid to calibrate.
    """
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    # Optimize log_T (unconstrained) so T = exp(log_T) stays strictly positive.
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(log_t.exp().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling and return calibrated probabilities.

    Args:
        logits: Logits to calibrate, shape (N,).
        temperature: The fitted temperature T.

    Returns:
        Calibrated probabilities in [0, 1], shape (N,).
    """
    return expit(logits / temperature)  # numerically stable sigmoid

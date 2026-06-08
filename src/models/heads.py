"""The trainable MLP classifier head for Path A.

Architecture (Appendix A) for the Frozen-feature MLP head model:

    BN -> Dropout(0.15) -> Linear(in_features -> 256) -> SiLU
       -> BN -> Dropout(0.25) -> Linear(256 -> 1)

Design notes:
  * Input is the 2304-dim concatenation of the three frozen backbones' features
    (or 768 for Phikon in Path B — the head is parameterized by ``in_features`` so the
    SAME class serves both paths, isolating the backbone as the only changed variable).
  * BatchNorm on the input standardizes the heterogeneously-scaled features coming
    from three different backbones before the first linear layer.
  * The head outputs a single LOGIT (no sigmoid) — we pair it with BCEWithLogitsLoss
    for numerical stability and apply sigmoid only at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Two-layer MLP with BatchNorm + Dropout, mapping features → a single logit."""

    def __init__(
        self,
        in_features: int,
        hidden: int = 256,
        p_drop1: float = 0.15,
        p_drop2: float = 0.25,
    ) -> None:
        """Build the head.

        Args:
            in_features: Dimension of the input feature vector (2304 for Path A).
            hidden: Width of the hidden layer.
            p_drop1: Dropout probability before the first linear layer.
            p_drop2: Dropout probability before the output linear layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),   # standardize concatenated backbone features
            nn.Dropout(p_drop1),
            nn.Linear(in_features, hidden),
            nn.SiLU(),                     # smooth activation; slight edge over ReLU here
            nn.BatchNorm1d(hidden),
            nn.Dropout(p_drop2),
            nn.Linear(hidden, 1),          # single logit (positive-class score)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a feature batch to logits.

        Args:
            x: Feature tensor of shape (B, in_features).

        Returns:
            Logit tensor of shape (B,) — squeezed so it lines up with float targets.
        """
        return self.net(x).squeeze(1)

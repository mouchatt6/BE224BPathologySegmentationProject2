"""Phikon foundation-model feature extractor (Path B).

Phikon (`owkin/phikon`) is a ViT-Base/16 self-supervised (iBOT) on ~40M TCGA
histopathology tiles — i.e. pretrained on *pathology*, not ImageNet. The hypothesis for
Path B: its features already encode tissue morphology (glands, nuclei, mitoses) that
ImageNet CNNs must approximate, so a frozen Phikon + the same MLP head may rank patches
better and/or transfer better than the Path A CNN trio.

This is the FROZEN feature extractor (the "base" Path B): no backbone training. The image
feature is Phikon's CLS token (`last_hidden_state[:, 0]`), 768-dim. Phikon expects
ImageNet normalization at 224x224, which matches ``src.data.transforms.build_eval_transform``,
so no Path-B-specific transform is needed.

Kept in its own module so Path A never imports ``transformers``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

PHIKON_MODEL: str = "owkin/phikon"
PHIKON_FEATURE_DIM: int = 768


class PhikonFeatureExtractor(nn.Module):
    """Frozen Phikon ViT-Base; returns the 768-dim CLS-token feature per patch."""

    def __init__(self, model_name: str = PHIKON_MODEL, pretrained: bool = True) -> None:
        """Load and freeze Phikon.

        Args:
            model_name: HuggingFace id (default ``owkin/phikon``).
            pretrained: Unused flag kept for a uniform extractor interface (Phikon is
                only meaningful with its pretrained weights, which AutoModel always loads).
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # frozen: pure feature function, no gradients
        self.feature_dim: int = PHIKON_FEATURE_DIM

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS-token features for a batch.

        Args:
            x: ImageNet-normalized image batch, shape (B, 3, 224, 224).

        Returns:
            Feature tensor of shape (B, 768) — the transformer's CLS token, which the
            Phikon model card recommends as the patch-level representation.
        """
        return self.model(pixel_values=x).last_hidden_state[:, 0]


def build_phikon_extractors(pretrained: bool = True) -> dict[str, PhikonFeatureExtractor]:
    """Instantiate the (single) frozen Phikon extractor, keyed for the extraction loop.

    Returns a one-entry dict so it plugs into the same per-backbone extraction/caching
    loop as the Path A CNN trio (which returns three entries).

    Args:
        pretrained: Forwarded to the extractor (Phikon always loads pretrained weights).

    Returns:
        ``{"phikon": PhikonFeatureExtractor()}``.
    """
    return {"phikon": PhikonFeatureExtractor(pretrained=pretrained)}

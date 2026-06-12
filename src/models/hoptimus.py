"""H-optimus-0 foundation-model feature extractor (Path C).

H-optimus-0 (`bioptimus/H-optimus-0`) is a **1.1B-parameter ViT-Giant/14** trained
self-supervised on >500k H&E whole-slide images — among the strongest *open* pathology
foundation models (Phikon-v1, by contrast, is a ViT-Base). Used here as a FROZEN feature
extractor → the same MLP head, isolating the backbone as the only change.

Two H-optimus specifics differ from Phikon and must be honored:
  * It loads via **timm** from the HF hub (not transformers), with ``init_values=1e-5``.
  * It expects its OWN normalization stats (below), NOT ImageNet — these are passed
    through the config (``features.norm_mean`` / ``features.norm_std``) into the transform.
It expects 224x224 tiles at ~0.5 µm/px, which matches this dataset's 20x patches.

Kept in its own module so the other paths never import it.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

HOPTIMUS_MODEL: str = "hf-hub:bioptimus/H-optimus-0"
HOPTIMUS_FEATURE_DIM: int = 1536
# H-optimus-0's required normalization (from the model card) — NOT ImageNet.
HOPTIMUS_MEAN: tuple[float, float, float] = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD: tuple[float, float, float] = (0.211883, 0.230117, 0.177517)


class HOptimusFeatureExtractor(nn.Module):
    """Frozen H-optimus-0 ViT-Giant; returns the 1536-dim feature per patch."""

    def __init__(self, pretrained: bool = True) -> None:
        """Load and freeze H-optimus-0 via timm.

        Args:
            pretrained: Load pretrained weights (always True; the model is only useful
                with its SSL-pretrained weights).
        """
        super().__init__()
        # init_values=1e-5 + dynamic_img_size=False per the model card. num_classes=0 so
        # the model returns the pooled feature vector rather than classification logits.
        self.model = timm.create_model(
            HOPTIMUS_MODEL, pretrained=pretrained, init_values=1e-5,
            dynamic_img_size=False, num_classes=0,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # frozen: pure feature function
        self.feature_dim: int = HOPTIMUS_FEATURE_DIM

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for a batch.

        Args:
            x: Image batch normalized with H-optimus stats, shape (B, 3, 224, 224).

        Returns:
            Feature tensor of shape (B, 1536).
        """
        return self.model(x)


def build_hoptimus_extractors(pretrained: bool = True) -> dict[str, HOptimusFeatureExtractor]:
    """Instantiate the (single) frozen H-optimus-0 extractor, keyed for the extraction loop.

    Args:
        pretrained: Forwarded to the extractor.

    Returns:
        ``{"h_optimus": HOptimusFeatureExtractor()}``.
    """
    return {"h_optimus": HOptimusFeatureExtractor(pretrained=pretrained)}

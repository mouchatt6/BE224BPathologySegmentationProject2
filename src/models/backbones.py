"""Frozen ImageNet feature extractors for Path A.

Path A is the Frozen-feature MLP head model: take three ImageNet-pretrained
backbones (ResNet-18, ResNet-34, EfficientNet-B0), FREEZE them, and use their
penultimate (global-average-pooled) feature vectors as fixed descriptors of each
patch. The only thing we train is a small MLP head on top of the concatenated
features (see ``src.models.heads``). Freezing means:
  * the backbones are never updated — they are pure feature functions;
  * features can be extracted ONCE and cached to disk, after which the rest of the
    pipeline (MLP training, CV, threshold search) is extremely fast.

Feature dims:  ResNet-18 = 512, ResNet-34 = 512, EfficientNet-B0 = 1280
Concatenated   = 512 + 512 + 1280 = 2304.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

# The three Path A backbones and their pooled feature dimensions. timm names are used
# so all three load uniformly via timm.create_model(...). num_classes=0 + global_pool
# "avg" makes each model return a pooled feature vector instead of class logits.
PATH_A_BACKBONES: dict[str, int] = {
    "resnet18": 512,
    "resnet34": 512,
    "efficientnet_b0": 1280,
}
PATH_A_FEATURE_DIM: int = sum(PATH_A_BACKBONES.values())  # 2304


class FrozenFeatureExtractor(nn.Module):
    """Wraps a single timm backbone as a frozen, pooled feature extractor."""

    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        """Build and freeze a timm backbone.

        Args:
            model_name: A timm model name, e.g. "resnet18".
            pretrained: Load ImageNet-pretrained weights (always True for Path A).
        """
        super().__init__()
        # num_classes=0 drops the classifier; global_pool="avg" returns a pooled
        # (B, feat_dim) vector — exactly the penultimate-layer descriptor we want.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.model.eval()  # disable dropout/batchnorm updates

        # Freeze every parameter — no gradients, the backbone is a fixed function.
        for p in self.model.parameters():
            p.requires_grad = False

        # Record the output feature dimension for downstream concatenation bookkeeping.
        self.feature_dim: int = PATH_A_BACKBONES.get(model_name, self.model.num_features)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features for a batch (no grad, backbone frozen).

        Args:
            x: Normalized image batch, shape (B, 3, 224, 224).

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        return self.model(x)


def build_path_a_extractors(
    pretrained: bool = True,
) -> dict[str, FrozenFeatureExtractor]:
    """Instantiate all three frozen Path A backbones.

    Args:
        pretrained: Load ImageNet weights (True for the real run).

    Returns:
        Dict mapping backbone name → FrozenFeatureExtractor, in the canonical order
        used for feature concatenation (resnet18, resnet34, efficientnet_b0).
    """
    return {name: FrozenFeatureExtractor(name, pretrained) for name in PATH_A_BACKBONES}

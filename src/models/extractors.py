"""Backend dispatcher for frozen feature extractors.

Single place that maps a config ``features.backend`` to the right frozen extractor(s),
so feature extraction and the stress eval build backbones the same way. The Phikon import
is lazy so Path A (timm CNNs) never pulls in ``transformers``.

Backends:
  * ``timm_cnn``  (default, Path A): ResNet-18 + ResNet-34 + EfficientNet-B0
  * ``phikon``    (Path B):          frozen Phikon ViT-Base (768-dim CLS token)
  * ``h_optimus`` (Path C):          frozen H-optimus-0 ViT-Giant (1536-dim)
"""

from __future__ import annotations

import torch.nn as nn


def build_extractors(backend: str = "timm_cnn", pretrained: bool = True) -> dict[str, nn.Module]:
    """Return the frozen extractor(s) for a backend, keyed by name.

    Args:
        backend: "timm_cnn", "phikon", or "h_optimus".
        pretrained: Load pretrained weights (always True in practice).

    Returns:
        Dict of {name: extractor module}. One entry for the foundation models; three for
        timm_cnn. Each extractor exposes ``.feature_dim`` and
        ``forward(batch) -> (B, feature_dim)``.
    """
    if backend == "phikon":
        from src.models.phikon import build_phikon_extractors  # lazy: avoids transformers for Path A
        return build_phikon_extractors(pretrained)
    if backend == "h_optimus":
        from src.models.hoptimus import build_hoptimus_extractors
        return build_hoptimus_extractors(pretrained)
    from src.models.backbones import build_path_a_extractors
    return build_path_a_extractors(pretrained)

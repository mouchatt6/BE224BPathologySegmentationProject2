"""Test-time augmentation variants.

H&E patches are rotationally and reflectionally invariant — a flipped/rotated tumor
patch is still a tumor patch — so averaging predictions (or features) over these
deterministic transforms reduces variance with no risk of changing the label.

Two variant sets are provided:
  * ``flip_variants``  — 4-way (identity, h-flip, v-flip, h+v-flip). Used at FEATURE
    EXTRACTION time for Path A (Appendix A: "4-way ... averaged").
  * ``rot_variants``   — 6-way (identity, h-flip, v-flip, rot90/180/270). Used for
    image-level inference TTA in the improvement layer (§7).

All functions operate on a batched tensor of shape (B, C, H, W). Flips/rotations act
on the spatial dims (H=-2, W=-1).
"""

from __future__ import annotations

import torch


def flip_variants(x: torch.Tensor) -> list[torch.Tensor]:
    """Return the 4 flip variants of a batch (for feature-extraction TTA).

    Args:
        x: Batch tensor of shape (B, C, H, W).

    Returns:
        List of 4 tensors: [identity, hflip, vflip, hflip+vflip].
    """
    return [
        x,                              # original
        torch.flip(x, dims=[-1]),       # horizontal flip (mirror left-right)
        torch.flip(x, dims=[-2]),       # vertical flip (mirror top-bottom)
        torch.flip(x, dims=[-2, -1]),   # both (== 180-degree rotation)
    ]


def rot_variants(x: torch.Tensor) -> list[torch.Tensor]:
    """Return the 6 rotation/flip variants of a batch (for inference-time TTA, §7).

    Args:
        x: Batch tensor of shape (B, C, H, W).

    Returns:
        List of 6 tensors: [identity, hflip, vflip, rot90, rot180, rot270].
    """
    return [
        x,                                  # original
        torch.flip(x, dims=[-1]),           # horizontal flip
        torch.flip(x, dims=[-2]),           # vertical flip
        torch.rot90(x, k=1, dims=[-2, -1]),  # 90 degrees
        torch.rot90(x, k=2, dims=[-2, -1]),  # 180 degrees
        torch.rot90(x, k=3, dims=[-2, -1]),  # 270 degrees
    ]

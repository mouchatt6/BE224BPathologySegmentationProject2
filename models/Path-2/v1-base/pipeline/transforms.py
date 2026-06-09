"""Albumentations transform pipelines.

For Path A the backbones are FROZEN and we only extract features, so the image-side
transform is intentionally minimal: convert to the backbone's expected normalization
and tensor layout. We do NOT apply random augmentation during feature extraction —
augmentation there would inject noise into the cached features. Instead, robustness
comes from deterministic flip TTA (applied in ``src.inference.tta`` and averaged).

A richer ``build_train_transform`` is provided for the future end-to-end paths
(Path B fine-tuning, §3.3) but is unused by Path A's frozen-feature workflow.

Normalization note: all three Path A backbones (ResNet-18/34, EfficientNet-B0) are
ImageNet-pretrained, so we use ImageNet mean/std. timm models expect the same stats.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet channel statistics (RGB). Standard for all ImageNet-pretrained backbones.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_eval_transform(
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """Deterministic transform for feature extraction / inference.

    Just normalize to the backbone's stats and convert HWC uint8 → CHW float tensor.
    Patches are already 224x224, so no resize/crop is needed.

    Args:
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        An Albumentations ``Compose`` taking ``image=<HxWx3 uint8 RGB>`` and returning
        a normalized float ``torch.Tensor`` of shape (3, 224, 224).
    """
    return A.Compose([
        A.Normalize(mean=mean, std=std),  # scales to ImageNet distribution
        ToTensorV2(),                     # HWC numpy → CHW torch tensor
    ])


def build_train_transform(
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """Augmentation pipeline for end-to-end TRAINING (Path B fine-tuning, §3.3).

    Unused by Path A (which trains on frozen features), kept here so the later
    fine-tuning path has a ready, pathology-appropriate pipeline. Key idea: H&E
    patches are rotationally invariant (unlike natural images), so 90-degree rotations
    and flips are label-preserving and safe to use aggressively. Heavy hue shifts are
    avoided because color carries biological meaning in H&E.

    Args:
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.

    Returns:
        An Albumentations ``Compose`` with geometric + mild photometric augmentation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),  # 0/90/180/270 — valid because H&E is rotation-invariant
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

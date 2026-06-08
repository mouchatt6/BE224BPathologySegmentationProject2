"""PatchDataset — loads a 224x224 H&E patch, applies a transform, returns a tensor.

The ``img_path`` strings come straight from ``train.csv`` / ``dummyTest.csv`` and
already include the ``train/`` or ``test/`` prefix (e.g. "train/0000.png"). They are
resolved against ``data_root`` (the repo's ``data/`` folder, which symlinks to the
real image directories living outside the git repo).

For the test set there are no labels, so a dummy 0 is returned; ``return_path=True``
keeps the original img_path string flowing through so submission rows stay aligned.

Image loading uses PIL, not ``cv2.imread``. With ``num_workers > 0`` (spawn workers on
macOS), OpenCV's internal threadpool contends with the worker processes and
``cv2.imread`` can intermittently return ``None`` on a perfectly valid file. PIL is
fork/spawn-safe (the torchvision standard), and we additionally pin OpenCV — which
Albumentations still uses internally for some transforms — to a single thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.config import REPO_ROOT

# Disable OpenCV's internal threading. This runs at import time in every spawned
# DataLoader worker too, preventing the imread-returns-None contention described above.
cv2.setNumThreads(0)


class PatchDataset(Dataset):
    """A torch Dataset over H&E patches referenced by relative img_path strings."""

    def __init__(
        self,
        img_paths: Sequence[str],
        labels: Sequence[int] | None,
        transform: A.Compose,
        data_root: str | Path | None = None,
        return_path: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            img_paths: Relative patch paths, e.g. "train/0000.png" — as stored in the
                CSVs. Resolved against ``data_root``.
            labels: Integer labels aligned with ``img_paths``. Pass None for the test
                set (a dummy 0 label is returned instead).
            transform: An Albumentations Compose that returns a CHW float tensor.
            data_root: Directory the relative paths are resolved against. Defaults to
                ``<repo>/data`` (which symlinks to the external image folders).
            return_path: If True, __getitem__ also returns the original img_path string
                (needed for the test set to keep submission rows aligned).
        """
        self.img_paths = list(img_paths)
        # Store labels as float32 — BCEWithLogitsLoss expects float targets.
        self.labels = (
            np.zeros(len(self.img_paths), dtype=np.float32)
            if labels is None
            else np.asarray(labels, dtype=np.float32)
        )
        self.transform = transform
        self.data_root = Path(data_root) if data_root else (REPO_ROOT / "data")
        self.return_path = return_path

    def __len__(self) -> int:
        """Number of patches in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str | int]:
        """Load and transform one patch.

        Args:
            idx: Sample index.

        Returns:
            (image, label, path_or_zero):
              * image: float tensor (3, 224, 224), normalized.
              * label: scalar float tensor (the patch's label, or 0 for test).
              * path_or_zero: the img_path string if ``return_path`` else integer 0.
        """
        rel_path = self.img_paths[idx]
        full_path = self.data_root / rel_path

        # Load with PIL and force 3-channel RGB. np.asarray gives an HWC uint8 RGB
        # array — exactly what Albumentations expects, with no BGR->RGB swap needed.
        with Image.open(full_path) as im:
            img = np.asarray(im.convert("RGB"))

        # Albumentations returns the transformed image under the "image" key.
        img = self.transform(image=img)["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        path_out: str | int = rel_path if self.return_path else 0
        return img, label, path_out

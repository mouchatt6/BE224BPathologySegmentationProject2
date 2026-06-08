"""Extract and cache frozen-backbone features for Path A (the Frozen-feature MLP head model).

For every train and test patch we run the three FROZEN ImageNet backbones
(ResNet-18, ResNet-34, EfficientNet-B0) and save their pooled feature vectors to disk.
This is the expensive step (it touches all 9,800 images), but it runs ONCE — after
this, training the MLP head is near-instant because it operates on the cached vectors.

4-way flip TTA: each image is passed through the backbone in 4 deterministic flip
orientations and the resulting feature vectors are averaged, which denoises the cached
descriptor at no risk to the label (H&E patches are flip-invariant).

Usage:
    python scripts/extract_features.py --config configs/path_a.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PatchDataset
from src.data.features import features_cache_dir
from src.data.transforms import build_eval_transform
from src.inference.tta import flip_variants
from src.models.backbones import build_path_a_extractors
from src.utils.config import load_config, resolve_path
from src.utils.device import device_name, get_device
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_everything


def extract_for_split(
    split: str,
    img_paths: list[str],
    extractors: dict,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_dir: Path,
    logger,
) -> None:
    """Extract 4-way-flip-TTA features for one split and cache them to disk.

    Args:
        split: "train" or "test" (used in output filenames).
        img_paths: Relative patch paths in the desired (fixed) row order.
        extractors: Dict of {backbone_name: FrozenFeatureExtractor}.
        transform: The deterministic eval transform (normalize + to-tensor).
        device: Torch device to run the backbones on.
        batch_size: Images per forward pass.
        num_workers: DataLoader workers.
        cache_dir: Directory to write ``{backbone}_{split}.npy`` and ``{split}_paths.csv``.
        logger: Logger for progress messages.

    Returns:
        None. Writes one ``.npy`` per backbone plus a paths CSV.
    """
    # shuffle=False is critical: it keeps batches in img_paths order so the saved
    # feature rows line up exactly with the saved paths CSV (and later, the labels).
    ds = PatchDataset(img_paths, labels=None, transform=transform, return_path=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Accumulate per-backbone feature chunks, concatenated at the end.
    feats: dict[str, list[np.ndarray]] = {name: [] for name in extractors}

    t0 = time.time()
    for imgs, _, _ in tqdm(loader, desc=f"extract[{split}]"):
        imgs = imgs.to(device, non_blocking=True)
        # 4 deterministic flip orientations of the batch.
        variants = flip_variants(imgs)
        for name, extractor in extractors.items():
            # Average the feature vectors over the 4 flip variants (TTA).
            acc = None
            for v in variants:
                f = extractor(v)  # (B, feat_dim), no grad (backbone is frozen)
                acc = f if acc is None else acc + f
            acc = acc / len(variants)
            feats[name].append(acc.float().cpu().numpy())

    # Persist each backbone's features and the shared row-order index.
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in extractors:
        arr = np.concatenate(feats[name], axis=0)  # (N, feat_dim)
        np.save(cache_dir / f"{name}_{split}.npy", arr)
        logger.info(f"  saved {name}_{split}.npy  shape={arr.shape}")
    pd.DataFrame({"img_path": img_paths}).to_csv(cache_dir / f"{split}_paths.csv", index=False)
    logger.info(f"  [{split}] done in {time.time() - t0:.1f}s for {len(img_paths)} patches")


def main() -> None:
    """Entry point: extract cached features for both train and test splits."""
    parser = argparse.ArgumentParser(description="Extract frozen-backbone features (Path A).")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(f"{cfg['experiment_name']}_extract")
    seed_everything(cfg["seed"], cfg.get("deterministic", False))

    device = get_device()
    logger.info(f"Device: {device_name(device)}")

    # Build and move the three frozen extractors onto the device.
    extractors = build_path_a_extractors(pretrained=True)
    for name, ex in extractors.items():
        ex.to(device).eval()
        logger.info(f"Loaded frozen backbone: {name} (feat_dim={ex.feature_dim})")

    transform = build_eval_transform()
    cache_dir = features_cache_dir(cfg["features"]["cache_dir"])
    bs = cfg["features"]["batch_size"]
    nw = cfg["features"]["num_workers"]

    # --- TRAIN split: img_paths come from train.csv (in file order). ---
    train_df = pd.read_csv(resolve_path(cfg["data"]["train_csv"]))
    logger.info(f"Extracting TRAIN features for {len(train_df)} patches ...")
    extract_for_split(
        "train", train_df["img_path"].tolist(), extractors, transform,
        device, bs, nw, cache_dir, logger,
    )

    # --- TEST split: img_paths come from dummyTest.csv (preserve its exact order). ---
    dummy_df = pd.read_csv(resolve_path(cfg["data"]["dummy_csv"]))
    logger.info(f"Extracting TEST features for {len(dummy_df)} patches ...")
    extract_for_split(
        "test", dummy_df["img_path"].tolist(), extractors, transform,
        device, bs, nw, cache_dir, logger,
    )

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()

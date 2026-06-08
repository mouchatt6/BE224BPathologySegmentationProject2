"""Cheap ablations on the cached v1 features (no re-extraction needed).

Answers two questions before committing v2 design choices:

  1. Late-fusion ensemble gate (action plan §8.2): train a separate MLP head per backbone,
     measure inter-backbone OOF prediction correlation, and compare a probability-averaged
     late-fusion ensemble against the v1 feature-concatenation model. If the backbones'
     OOF predictions correlate > 0.97, late fusion adds no diversity and we skip it.

  2. Label-smoothing sweep: retrain the concatenation head with smoothing in {0, 0.05, 0.1}
     and compare OOF composite, to decide the v2 setting.

This trains only the tiny MLP head on cached features, so it runs in ~1-2 minutes.

Usage:
    python scripts/ablate_v1.py --config configs/path_a.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.features import load_concat_features
from src.data.splits import load_folds
from src.eval.metrics import compute_score
from src.inference.threshold import optimize_threshold
from src.models.heads import MLPHead
from src.utils.config import load_config, resolve_path
from src.utils.device import get_device
from src.utils.seed import seed_everything


def train_cv_oof(
    X: np.ndarray, y: np.ndarray, fold_of: np.ndarray, n_splits: int,
    device: torch.device, smoothing: float = 0.0, seed: int = 42,
    lr: float = 2e-3, wd: float = 1e-3, bs: int = 256,
    max_epochs: int = 100, patience: int = 15,
) -> np.ndarray:
    """Train the MLP head with 5-fold CV and return OOF probabilities.

    Mirrors scripts/train_path_a.py's per-fold logic in compact form (this is an
    analysis-only helper). Early-stops each fold on validation AUROC.

    Args:
        X: Feature matrix (N, D). y: labels (N,). fold_of: fold index per row.
        n_splits: number of folds. device: torch device. smoothing: label smoothing.
        seed: RNG seed. lr/wd/bs/max_epochs/patience: training hyperparameters.

    Returns:
        OOF probability array (N,), each row filled from the fold where it was held out.
    """
    seed_everything(seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for fold in range(n_splits):
        val_idx = np.where(fold_of == fold)[0]
        tr_idx = np.where(fold_of != fold)[0]
        loader = DataLoader(
            TensorDataset(torch.tensor(X[tr_idx], dtype=torch.float32),
                          torch.tensor(y[tr_idx], dtype=torch.float32)),
            batch_size=bs, shuffle=True, drop_last=True,
        )
        Xval = torch.tensor(X[val_idx], dtype=torch.float32)
        model = MLPHead(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        crit = nn.BCEWithLogitsLoss()
        best_auroc, best_probs, noimp = -1.0, None, 0
        for _ in range(max_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if smoothing > 0:
                    yb = yb * (1.0 - smoothing) + 0.5 * smoothing
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                probs = expit(model(Xval.to(device)).cpu().numpy())
            auroc = roc_auc_score(y[val_idx], probs)
            if auroc > best_auroc:
                best_auroc, best_probs, noimp = auroc, probs, 0
            else:
                noimp += 1
                if noimp >= patience:
                    break
        oof[val_idx] = best_probs
    return oof


def _composite(y: np.ndarray, oof: np.ndarray) -> tuple[float, float, float]:
    """Return (AUROC, F1@tau*, composite@0.5) for an OOF prediction vector."""
    tau = optimize_threshold(y, oof)["tau"]
    comp, auroc, f1 = compute_score(y, (oof >= tau).astype(int), oof, alpha=0.5)
    return auroc, f1, comp


def main() -> None:
    """Run the late-fusion correlation gate and the label-smoothing sweep."""
    parser = argparse.ArgumentParser(description="Cheap ablations on v1 features.")
    parser.add_argument("--config", default="configs/path_a.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    backbones = cfg["features"]["backbones"]
    cache_dir = cfg["features"]["cache_dir"]

    # Labels + folds aligned to the feature row order.
    X, paths = load_concat_features("train", backbones, cache_dir)
    label_map = dict(zip(*[pd.read_csv(resolve_path(cfg["data"]["train_csv"]))[c]
                           for c in ("img_path", "label")]))
    y = np.array([label_map[p] for p in paths], dtype=np.int64)
    fold_map = dict(zip(load_folds()["img_path"], load_folds()["fold"]))
    fold_of = np.array([fold_map[p] for p in paths], dtype=np.int64)
    n = cfg["data"]["n_splits"]

    print("=" * 70)
    print("ABLATION 1 — Late-fusion ensemble gate (per-backbone OOF correlation)")
    print("=" * 70)
    per_bb_oof = {}
    for bb in backbones:
        Xbb, _ = load_concat_features("train", [bb], cache_dir)
        oof_bb = train_cv_oof(Xbb, y, fold_of, n, device)
        per_bb_oof[bb] = oof_bb
        a, f, c = _composite(y, oof_bb)
        print(f"  {bb:16s} OOF AUROC={a:.4f}  F1={f:.4f}  composite={c:.4f}")

    # Inter-backbone OOF prediction correlation matrix.
    mat = np.stack([per_bb_oof[bb] for bb in backbones])
    corr = np.corrcoef(mat)
    print("\n  Inter-backbone OOF correlation:")
    print("           " + "  ".join(f"{bb[:8]:>8s}" for bb in backbones))
    for i, bb in enumerate(backbones):
        print(f"  {bb[:8]:>8s} " + "  ".join(f"{corr[i,j]:8.4f}" for j in range(len(backbones))))
    max_off = max(corr[i, j] for i in range(len(backbones)) for j in range(len(backbones)) if i != j)

    # Late fusion = simple average of per-backbone OOF probabilities.
    late = np.mean(mat, axis=0)
    la, lf, lc = _composite(y, late)
    # Concatenation baseline (v1 model) OOF from the saved npz.
    v1 = np.load(resolve_path(cfg["outputs"]["oof_path"]))
    ca, cf, cc = _composite(y, v1["oof_probs"])
    print(f"\n  Concatenation (v1) : AUROC={ca:.4f} F1={cf:.4f} composite={cc:.4f}")
    print(f"  Late-fusion (mean) : AUROC={la:.4f} F1={lf:.4f} composite={lc:.4f}")
    print(f"  Max off-diagonal correlation = {max_off:.4f}  "
          f"-> {'SKIP late fusion (>0.97, no diversity)' if max_off > 0.97 else 'late fusion may add diversity'}")
    print(f"  Late-fusion vs concat composite delta = {lc - cc:+.4f}")

    print("\n" + "=" * 70)
    print("ABLATION 2 — Label-smoothing sweep (concatenation head)")
    print("=" * 70)
    for s in (0.0, 0.05, 0.1):
        oof_s = train_cv_oof(X, y, fold_of, n, device, smoothing=s)
        a, f, c = _composite(y, oof_s)
        print(f"  label_smoothing={s:<4}  OOF AUROC={a:.4f}  F1={f:.4f}  composite={c:.4f}")


if __name__ == "__main__":
    main()

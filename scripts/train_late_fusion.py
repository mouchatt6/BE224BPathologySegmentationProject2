"""Late-fusion ensemble of per-backbone MLP heads (Path-1 v2 ensemble variant).

Instead of one head on the 2304-dim concatenation (early fusion), train a separate head
on EACH backbone's features and average their probabilities (late / decision fusion).
The v1 ablation showed the backbones' OOF predictions correlate only ~0.83 — below the
0.97 skip threshold (action plan §8.2) — and late fusion beat concatenation, so the
diversity is real. Weighted by per-backbone OOF AUROC.

Writes an .npz + sidecar JSON in the SAME format as scripts/train_path_a.py, so
scripts/make_submission.py and the run-log consume it unchanged.

Usage:
    python scripts/train_late_fusion.py --config configs/path_a_v2_ensemble.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.features import load_concat_features
from src.data.splits import load_folds, make_stratified_folds
from src.eval.metrics import compute_score, report_across_alphas
from src.inference.threshold import optimize_threshold
from src.models.heads import MLPHead
from src.utils.config import REPO_ROOT, config_hash, load_config, resolve_path
from src.utils.device import device_name, get_device
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_everything


def _git_sha() -> str:
    """Best-effort short git SHA for provenance."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return ""


def train_backbone_cv(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, fold_of: np.ndarray,
    n_splits: int, cfg: dict, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one backbone's head with 5-fold CV; return (OOF probs, fold-averaged test probs).

    Args:
        X: this backbone's train features (N, d). y: labels (N,). X_test: test features (M, d).
        fold_of: fold index per train row. n_splits: number of folds. cfg: config. device: device.

    Returns:
        (oof_probs (N,), test_probs (M,)) — test_probs averaged over the folds' heads.
    """
    tcfg, hcfg = cfg["train"], cfg["head"]
    smoothing = float(tcfg.get("label_smoothing", 0.0))
    oof = np.zeros(len(y), dtype=np.float64)
    test_acc = np.zeros(len(X_test), dtype=np.float64)
    Xtest_t = torch.tensor(X_test, dtype=torch.float32)

    for fold in range(n_splits):
        seed_everything(cfg["seed"] + fold)  # vary per fold but reproducible
        val_idx = np.where(fold_of == fold)[0]
        tr_idx = np.where(fold_of != fold)[0]
        loader = DataLoader(
            TensorDataset(torch.tensor(X[tr_idx], dtype=torch.float32),
                          torch.tensor(y[tr_idx], dtype=torch.float32)),
            batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
        )
        Xval = torch.tensor(X[val_idx], dtype=torch.float32)
        model = MLPHead(X.shape[1], hcfg["hidden"], hcfg["p_drop1"], hcfg["p_drop2"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
        crit = nn.BCEWithLogitsLoss()
        best_auroc, best_state, noimp = -1.0, None, 0
        for _ in range(tcfg["max_epochs"]):
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
                vp = expit(model(Xval.to(device)).cpu().numpy())
            a = roc_auc_score(y[val_idx], vp)
            if a > best_auroc:
                best_auroc, best_state, noimp = a, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                noimp += 1
                if noimp >= tcfg["early_stop_patience"]:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[val_idx] = expit(model(Xval.to(device)).cpu().numpy())
            test_acc += expit(model(Xtest_t.to(device)).cpu().numpy())
    return oof, test_acc / n_splits


def main() -> None:
    """Train per-backbone heads, AUROC-weight-average them, and save OOF + test predictions."""
    parser = argparse.ArgumentParser(description="Late-fusion ensemble training (Path-1 v2).")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = get_logger(f"{cfg['experiment_name']}_train")
    seed_everything(cfg["seed"], cfg.get("deterministic", False))
    device = get_device()
    logger.info(f"Device: {device_name(device)} | late-fusion of {cfg['features']['backbones']}")

    backbones = cfg["features"]["backbones"]
    cache_dir = cfg["features"]["cache_dir"]

    # Labels + folds aligned to feature row order.
    _, train_paths = load_concat_features("train", backbones, cache_dir)
    _, test_paths = load_concat_features("test", backbones, cache_dir)
    train_df = pd.read_csv(resolve_path(cfg["data"]["train_csv"]))
    label_map = dict(zip(train_df["img_path"], train_df["label"]))
    y = np.array([label_map[p] for p in train_paths], dtype=np.int64)
    if not (REPO_ROOT / "outputs" / "folds.csv").exists():
        make_stratified_folds(cfg["data"]["train_csv"], cfg["data"]["n_splits"], cfg["seed"])
    fold_map = dict(zip(load_folds()["img_path"], load_folds()["fold"]))
    fold_of = np.array([fold_map[p] for p in train_paths], dtype=np.int64)
    n = cfg["data"]["n_splits"]

    # Per-backbone CV predictions.
    oof_by_bb, test_by_bb, auroc_by_bb = {}, {}, {}
    for bb in backbones:
        Xtr, _ = load_concat_features("train", [bb], cache_dir)
        Xte, _ = load_concat_features("test", [bb], cache_dir)
        oof_bb, test_bb = train_backbone_cv(Xtr, y, Xte, fold_of, n, cfg, device)
        oof_by_bb[bb], test_by_bb[bb] = oof_bb, test_bb
        auroc_by_bb[bb] = float(roc_auc_score(y, oof_bb))
        logger.info(f"  {bb:16s} OOF AUROC = {auroc_by_bb[bb]:.4f}")

    # AUROC-weighted average over backbones (probability space).
    w = np.array([auroc_by_bb[bb] for bb in backbones], dtype=np.float64)
    w = w / w.sum()
    oof_probs = np.sum([w[i] * oof_by_bb[bb] for i, bb in enumerate(backbones)], axis=0)
    test_probs = np.sum([w[i] * test_by_bb[bb] for i, bb in enumerate(backbones)], axis=0)
    logger.info(f"Fusion weights: {dict(zip(backbones, np.round(w, 3)))}")

    # Threshold + reporting (identical protocol to train_path_a.py).
    tau = optimize_threshold(y, oof_probs)["tau"] if cfg["postprocess"].get("optimize_threshold", True) else 0.5
    thr = optimize_threshold(y, oof_probs)
    oof_labels = (oof_probs >= tau).astype(int)
    report = report_across_alphas(y, oof_labels, oof_probs, cfg["report_alphas"])
    _, auroc, f1 = compute_score(y, oof_labels, oof_probs, alpha=0.5)
    logger.info("=" * 64)
    logger.info(f"LATE-FUSION OOF AUROC : {auroc:.4f}")
    logger.info(f"OOF F1 @ tau={tau:.3f}  : {f1:.4f}  (stable={thr['f1_stable']})")
    for a in cfg["report_alphas"]:
        logger.info(f"Composite @ alpha={a}: {report[f'score@{a}']:.4f}")
    logger.info("=" * 64)

    oof_path = resolve_path(cfg["outputs"]["oof_path"])
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        oof_path, y_true=y, oof_logits=np.zeros_like(oof_probs), oof_probs=oof_probs,
        test_probs=test_probs, test_paths=np.array(test_paths), tau=tau, temperature=1.0,
    )
    sidecar = {
        "experiment_name": cfg["experiment_name"], "config_hash": config_hash(cfg), "git_sha": _git_sha(),
        "oof_auroc": auroc, "oof_f1": f1, "tau": tau, "temperature": 1.0,
        "fold_aurocs": [auroc_by_bb[bb] for bb in backbones],  # per-backbone here
        "fusion_weights": dict(zip(backbones, w.tolist())),
        "report": report, "threshold_stability": thr,
    }
    oof_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    logger.info(f"Saved late-fusion OOF + test predictions to {oof_path}")


if __name__ == "__main__":
    main()

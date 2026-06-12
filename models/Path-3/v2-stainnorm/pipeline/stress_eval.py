"""Stain-stress robustness eval — an offline proxy for the train->test gap.

Same-distribution OOF cannot validate stain robustness (it pays off only on the *shifted*
test set). So we simulate a cross-lab stain shift on held-out patches and measure how much
each pipeline's AUROC degrades:

  * v1 pipeline: shifted patch -> frozen backbones (no normalization) -> v1 head.
  * v2 pipeline: shifted patch -> Macenko normalization -> frozen backbones -> v2 head.

The stain shift recomposes each patch with imbalanced H&E concentrations (stronger
hematoxylin, weaker eosin) against the canonical stain matrix — a deterministic, realistic
"different scanner/lab" perturbation. A more stain-robust model degrades LESS. If v2
(Macenko) degrades less than v1, that is evidence the stain layer closes the gap even when
clean OOF looks similar.

Predictions use each patch's own CV-fold head (loaded from outputs/checkpoints), single
view (no TTA) for a fast, fair comparison; we report relative degradation, not absolutes.

Usage:
    python scripts/stress_eval.py --v1 configs/path_a.yaml --v2 configs/path_a_v2.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.data.splits import load_folds
from src.data.stain import MacenkoNormalizer, _HEREF
from src.data.transforms import build_eval_transform
from src.models.extractors import build_extractors
from src.models.heads import MLPHead
from src.utils.config import REPO_ROOT, load_config, resolve_path
from src.utils.device import get_device
from src.utils.seed import seed_everything

# Fixed cross-lab stain perturbation: amplify hematoxylin, attenuate eosin.
SHIFT_H, SHIFT_E = 1.30, 0.70


def stain_shift(img: np.ndarray, Io: int = 240) -> np.ndarray:
    """Recompose a patch with imbalanced H&E concentrations (simulated lab shift).

    Args:
        img: HxWx3 uint8 RGB patch.
        Io: transmitted-light intensity for the OD transform.

    Returns:
        HxWx3 uint8 RGB patch with hematoxylin scaled by SHIFT_H and eosin by SHIFT_E.
    """
    h, w, _ = img.shape
    OD = -np.log((img.reshape(-1, 3).astype(np.float64) + 1.0) / Io)
    C = np.linalg.lstsq(_HEREF, OD.T, rcond=None)[0]        # concentrations vs canonical H/E
    C = C * np.array([SHIFT_H, SHIFT_E])[:, None]           # imbalance the two stains
    out = Io * np.exp(-_HEREF @ C)
    return np.clip(out, 0, 255).T.reshape(h, w, 3).astype(np.uint8)


def load_heads(experiment_name: str, n_splits: int, in_features: int, device) -> list[MLPHead]:
    """Load the per-fold trained MLP heads for an experiment."""
    ckpt_dir = REPO_ROOT / "outputs" / "checkpoints"
    heads = []
    for k in range(n_splits):
        head = MLPHead(in_features).to(device)
        head.load_state_dict(torch.load(ckpt_dir / f"{experiment_name}_fold{k}.pt", map_location=device))
        head.eval()
        heads.append(head)
    return heads


@torch.no_grad()
def extract_concat(imgs: list[np.ndarray], extractors, transform, device,
                   normalizer=None, batch_size: int = 128) -> np.ndarray:
    """Single-view concatenated features for a list of uint8 patches (optional Macenko).

    Processes in mini-batches so a large subsample never forms one giant tensor (which
    would exhaust MPS memory).
    """
    out: list[np.ndarray] = []
    for start in range(0, len(imgs), batch_size):
        chunk = imgs[start:start + batch_size]
        tensors = []
        for im in chunk:
            if normalizer is not None:
                im = normalizer.normalize(im)
            tensors.append(transform(image=im)["image"])
        x = torch.stack(tensors).to(device)
        feats = [ex(x).float().cpu().numpy() for ex in extractors.values()]
        out.append(np.concatenate(feats, axis=1))
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_by_fold(X: np.ndarray, fold_of: np.ndarray, heads: list[MLPHead], device) -> np.ndarray:
    """Predict each row with its own fold's head; return probabilities (N,)."""
    probs = np.zeros(len(X), dtype=np.float64)
    for k, head in enumerate(heads):
        idx = np.where(fold_of == k)[0]
        if len(idx) == 0:
            continue
        logits = head(torch.tensor(X[idx], dtype=torch.float32).to(device)).cpu().numpy()
        probs[idx] = expit(logits)
    return probs


def main() -> None:
    """Run the v1-vs-v2 stain-stress comparison on a held-out subsample."""
    parser = argparse.ArgumentParser(description="Stain-stress robustness eval.")
    parser.add_argument("--v1", default="configs/path_a.yaml")
    parser.add_argument("--v2", default="configs/path_a_v2.yaml")
    parser.add_argument("--n", type=int, default=2400, help="subsample size for speed")
    args = parser.parse_args()

    seed_everything(42)
    device = get_device()
    cfg1, cfg2 = load_config(args.v1), load_config(args.v2)
    n_splits = cfg1["data"]["n_splits"]
    dim = cfg1["features"]["feature_dim"]

    # Subsample held-out patches with their fold ids and labels.
    df = pd.read_csv(resolve_path(cfg1["data"]["train_csv"]))
    folds = load_folds()
    fold_map = dict(zip(folds["img_path"], folds["fold"]))
    rng = np.random.default_rng(42)
    sel = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    paths = df["img_path"].to_numpy()[sel]
    y = df["label"].to_numpy()[sel].astype(int)
    fold_of = np.array([fold_map[p] for p in paths], dtype=np.int64)
    imgs = [np.asarray(Image.open(resolve_path(cfg1["data"]["data_root"]) / p).convert("RGB")) for p in paths]
    shifted = [stain_shift(im) for im in imgs]
    print(f"Subsample: {len(imgs)} patches | stain shift H×{SHIFT_H} E×{SHIFT_E}")

    extractors = build_extractors(cfg1["features"].get("backend", "timm_cnn"), pretrained=True)
    for ex in extractors.values():
        ex.to(device).eval()
    transform = build_eval_transform()
    normalizer = MacenkoNormalizer()

    heads_v1 = load_heads(cfg1["experiment_name"], n_splits, dim, device)
    heads_v2 = load_heads(cfg2["experiment_name"], n_splits, dim, device)

    def auroc(imglist, normalize, heads):
        X = extract_concat(imglist, extractors, transform, device, normalizer if normalize else None)
        return roc_auc_score(y, predict_by_fold(X, fold_of, heads, device))

    # v1: no normalization, v1 heads.  v2: Macenko, v2 heads.
    v1_clean = auroc(imgs, False, heads_v1)
    v1_shift = auroc(shifted, False, heads_v1)
    v2_clean = auroc(imgs, True, heads_v2)
    v2_shift = auroc(shifted, True, heads_v2)

    print("=" * 60)
    print(f"{'pipeline':<22}{'clean':>8}{'shifted':>9}{'drop':>8}")
    print(f"{'v1 (no stain norm)':<22}{v1_clean:>8.4f}{v1_shift:>9.4f}{v1_clean - v1_shift:>8.4f}")
    print(f"{'v2 (Macenko norm)':<22}{v2_clean:>8.4f}{v2_shift:>9.4f}{v2_clean - v2_shift:>8.4f}")
    print("=" * 60)
    print(f"Robustness gain (smaller drop is better): "
          f"v1 drop {v1_clean - v1_shift:.4f} vs v2 drop {v2_clean - v2_shift:.4f}")
    print("Interpretation: if v2's drop < v1's drop, Macenko normalization improves "
          "robustness to stain shift — direct evidence for closing the OOF->LB gap.")


if __name__ == "__main__":
    main()

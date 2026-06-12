# Path-2 — v1 (frozen Phikon base)

**Submission:** `Path-2-phikon-base_v1.csv`  ·  **Config:** `configs/path_b.yaml`

The base Path B model: **frozen Phikon** (`owkin/phikon`, ViT-Base/16, iBOT pretrained on
~40M TCGA histopathology tiles) as a feature extractor → the same MLP head as Path A. A
clean backbone swap — **no** stain normalization, late fusion, label smoothing, or
fine-tuning (those are the deferred improvement layer).

## What was executed

| Component | Setting |
|---|---|
| Backbone (frozen, **TCGA-pathology**) | Phikon ViT-Base/16 → **768-dim** CLS token |
| Feature-extraction TTA | 4-way flip, averaged (matches Path A v1) |
| Normalization | ImageNet mean/std (Phikon's expected preprocessing) |
| Cross-validation | 5-fold stratified |
| Head | `BN → Dropout(0.15) → Linear(768→256) → SiLU → BN → Dropout(0.25) → Linear(256→1)` |
| Optimizer / loss | AdamW lr=2e-3 wd=1e-3, batch 256, BCE, label smoothing 0.0 |
| Schedule | ≤100 epochs, early stop patience 15 on val AUROC |
| Post-processing | OOF-optimized threshold τ\*=0.325 (stable); no calibration |

Compute: frozen extraction ~7 min on Apple M3 Pro / MPS (97 img/s single-view); head
training ~30 s. **No Kaggle/Colab needed** for the frozen base.

## Results (out-of-fold)

| Metric | Value |
|---|---|
| OOF AUROC | **0.9929** (per-fold 0.992 / 0.9942 / 0.9931 / 0.9952 / 0.9912) |
| Best F1 @ τ\*=0.325 | **0.9645** (stable) |
| Composite @ α=0.3 / 0.5 / 0.7 | 0.9730 / 0.9787 / 0.9844 |
| Predicted test positive rate | 0.564 |

### vs Path A
| Model | OOF AUROC | OOF composite @0.5 |
|---|---|---|
| Path-1 v1 (CNN trio) | 0.9393 | 0.9055 |
| Path-1 v2 (stain + late fusion) | 0.9470 | 0.9143 |
| **Path-2 v1 (frozen Phikon)** | **0.9929** | **0.9787** |

The pathology-pretrained transformer outranks the ImageNet CNN trio by **+0.046 OOF
AUROC** — a large gap, consistent with domain-specific foundation models encoding tissue
morphology the CNNs must approximate.

## Caveats / watch-list
- **OOF is same-distribution; the LB is the real test.** Path A had a sizable OOF→LB gap
  (stain/scanner shift). Phikon's domain pretraining may shrink it (more stain-robust), but
  that is unconfirmed until upload. Even with Path A's gap, the 0.979 OOF should comfortably
  clear the CNN baselines on the leaderboard.
- Probabilities are very confident (features highly separable) and the predicted positive
  rate (0.564) sits a bit above the balanced train prior — if LB F1 underperforms AUROC,
  the threshold is the first thing to revisit.

## Reproduce

```bash
python scripts/extract_features.py --config configs/path_b.yaml   # frozen Phikon features (~7 min MPS)
python scripts/train_path_a.py     --config configs/path_b.yaml   # 5-fold MLP head (backbone-agnostic)
python scripts/make_submission.py  --config configs/path_b.yaml   # validated CSV + run log
```

`pipeline/` holds frozen copies of the scripts/config for this version.

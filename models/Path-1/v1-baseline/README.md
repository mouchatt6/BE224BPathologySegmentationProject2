# Path-1 — v1 (baseline)

**Submission:** `Path-1-baseline_v1.csv`  ·  **Public LB:** **0.811**  ·  **Code state:** git `ccc69f2`

The first Path A submission: the **Frozen-feature MLP head model**. Three frozen
ImageNet CNNs produce fixed feature vectors that an MLP head classifies, with 5-fold CV
and an OOF-optimized decision threshold.

## What was executed

| Component | Setting |
|---|---|
| Backbones (frozen, ImageNet) | ResNet-18 (512) + ResNet-34 (512) + EfficientNet-B0 (1280) → **2304-dim** concat |
| Feature-extraction TTA | 4-way flip (orig + hflip + vflip + h+v), averaged |
| Image loading | PIL (+ `cv2.setNumThreads(0)`) |
| Normalization | ImageNet mean/std |
| Cross-validation | 5-fold stratified (by label; no slide grouping in filenames) |
| Head | `BN → Dropout(0.15) → Linear(2304→256) → SiLU → BN → Dropout(0.25) → Linear(256→1)` |
| Optimizer | AdamW, lr=2e-3, weight_decay=1e-3, batch=256 |
| Loss | `BCEWithLogitsLoss`, **label smoothing 0.0** |
| Schedule | ≤100 epochs, early-stop patience 15 on val AUROC |
| Post-processing | threshold τ optimized on OOF (F1); calibration off |
| Seed | 42 |

## Results (out-of-fold)

| Metric | Value |
|---|---|
| OOF AUROC | **0.9393** (per-fold 0.9397 / 0.9409 / 0.9325 / 0.9414 / 0.9462, mean 0.9401) |
| Best F1 @ τ\*=0.520 | **0.8716** (stable) |
| Composite @ α=0.3 / 0.5 / 0.7 | 0.8920 / 0.9055 / 0.9190 |
| Predicted positive rate (test) | 0.488 (1172/2400) |
| **Public LB** | **0.811** |

### The headline finding: a ~9-point OOF→LB gap
OOF composite ~0.905 vs LB 0.811. Because the OOF score is an honest 5-fold holdout,
this gap is **distribution shift** (stain/scanner/lab variation between train and test),
not overfitting — which is what motivates the v2 stain-normalization work.

## Reproduce

The exact library code for this version is pinned at git commit **`ccc69f2`**. The
entry scripts + config that define the run are snapshotted in [`pipeline/`](pipeline/).

```bash
git checkout ccc69f2          # full library code state for v1
python scripts/extract_features.py --config configs/path_a.yaml
python scripts/train_path_a.py     --config configs/path_a.yaml
python scripts/make_submission.py  --config configs/path_a.yaml
```

> `pipeline/` holds reference copies of the v1 entry scripts (`extract_features.py`,
> `train_path_a.py`, `make_submission.py`), the data modules that later change
> (`transforms.py`, `dataset.py`), and the v1 config (`path_a.yaml`). They import the
> `src/` package, so run them from the pinned commit above for a faithful reproduction.

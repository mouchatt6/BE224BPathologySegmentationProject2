# BE224B Project 2 — Prostate Pathology Patch Classification

Binary patch-level classification of 224×224 H&E-stained histopathology patches from
radical prostatectomy whole-slide images: **cancerous (`label=1`) vs non-cancerous
(`label=0`)**.

Competition scoring:

```
Score = alpha * AUROC + (1 - alpha) * F1     (alpha undisclosed)
```

Because `alpha` is unknown, the pipeline optimizes **both** the ranking quality of the
predicted probabilities (AUROC) and a tuned decision threshold for the binary labels
(F1).

## Path A — Frozen-feature MLP head model

The first model is the **Frozen-feature MLP head model**:

1. Three **frozen** ImageNet backbones — ResNet-18, ResNet-34, EfficientNet-B0 —
   produce pooled feature vectors (512 + 512 + 1280 = **2304-dim** concatenated).
2. A small **MLP head** (`BN → Dropout → Linear(256) → SiLU → BN → Dropout → Linear(1)`)
   is trained on the cached features with **5-fold stratified CV**.
3. The decision threshold `tau` is optimized on **out-of-fold (OOF)** predictions to
   maximize F1; the 5 fold models are averaged for the test predictions.

Feature extraction uses 4-way flip test-time augmentation (averaged), and the
backbones are extracted **once** and cached, so head training is near-instant.

## Repository layout

```
src/
  data/      dataset, transforms, stratified splits, feature loading
  models/    frozen backbones, MLP head
  eval/      compute_score (mirrors Kaggle), validate_submission
  inference/ TTA, threshold optimization, temperature calibration
  utils/     seed, device (CUDA/MPS/CPU), config, logging
scripts/
  extract_features.py   # cache frozen-backbone features (train + test)
  train_path_a.py       # 5-fold MLP CV, OOF assembly, threshold opt
  make_submission.py    # build + validate submission.csv
configs/
  path_a.yaml           # all hyperparameters for Path A
outputs/                # features, checkpoints, OOF preds, logs (gitignored)
submissions/            # {name}_{version}.csv + submission_logs/{name}_{version}.md (gitignored)
data/                   # symlinks to the dataset (gitignored — never committed)
```

## Setup

The data lives **outside** this repo and is never committed. `data/` holds symlinks to
the dataset; raw images, caches, checkpoints, and submissions are all gitignored.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Developed on Apple Silicon (M3 Pro); the device helper auto-selects **CUDA → MPS → CPU**,
so the same code runs unchanged on Colab/Kaggle GPUs.

## Run Path A end-to-end

```bash
python scripts/extract_features.py --config configs/path_a.yaml   # cache features (one-time)
python scripts/train_path_a.py     --config configs/path_a.yaml   # 5-fold CV + OOF + tau
python scripts/make_submission.py  --config configs/path_a.yaml   # write + validate submission
```

Outputs land in `outputs/`: OOF predictions and metrics (`outputs/oof_preds/`) and
per-fold checkpoints (`outputs/checkpoints/`). The validated submission is written to
`submissions/{name}_{version}.csv` (e.g. `submissions/Path-1-baseline_v1.csv`), cataloged
by model and iteration — bump `submission.version` in the config for each resubmission.
Each run also writes `submissions/submission_logs/{name}_{version}.md` cataloging the OOF
AUROC, best F1, composite per α, and a summary of what was executed (for run-to-run diffs).

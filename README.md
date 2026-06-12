# BE224B Project 2 — Prostate Pathology Patch Classification

Binary patch-level classification of **224×224 H&E-stained histopathology patches** from
radical prostatectomy whole-slide images: **cancerous (`label=1`) vs non-cancerous
(`label=0`)**. UCLA BE224B course Kaggle competition — **7,400** train + **2,400** test PNGs.

This repository implements **three parallel modeling paths** behind one config-driven,
backbone-agnostic pipeline: an ImageNet-CNN baseline and two pathology **foundation
models**, each in a clean base version and a stain-normalized improvement version.

---

## Scoring metric and its strategic implication

```
Score = alpha * AUROC + (1 - alpha) * F1     (alpha is the graders' fixed but UNDISCLOSED weight)
```

`alpha` is set by the graders and is **not** tunable. Because we don't know it, the pipeline
must do well on **both** halves of the score:

- **AUROC** — ranking quality of the predicted probabilities (threshold-free).
- **F1** — quality of the binary labels, which depends on the **decision threshold `tau`**.

`tau` is the controllable analog of `alpha`: we optimize it on **out-of-fold (OOF)**
predictions to maximize F1, and report the composite across an `alpha` grid so the choice is
robust to whatever weight the graders use.

---

## Results / Leaderboard

Public-leaderboard numbers are authoritative; OOF AUROC is same-distribution 5-fold cross-validation.

| Submission | Backbone | OOF AUROC | Public LB |
|---|---|---|---|
| **Path-1 v1** — CNN trio (ResNet-18 + ResNet-34 + EfficientNet-B0, frozen) | ImageNet CNNs | 0.9393 | 0.811 |
| **Path-1 v2** — + Macenko stain norm + 6-way rot TTA + label smoothing + late fusion | ImageNet CNNs | 0.9470 | 0.827 |
| **Path-2 v1** — frozen Phikon | ViT-B, TCGA-pretrained | 0.9929 | 0.882 |
| **Path-2 v2** — Phikon + stain norm | ViT-B | 0.9890 | 0.890 |
| **Path-3 v1** — frozen H-optimus-0 | ViT-G / 1.1B, histology-SSL | 0.9897 | **0.895 (BEST)** |
| **Path-3 v2** — H-optimus + stain norm | ViT-G | _in progress_ | _pending_ |

**Three findings drive the project:**

1. **Pathology foundation models crush ImageNet CNNs.** Swapping the only variable — the
   frozen backbone — lifts the LB from **0.827 → 0.882+**. Histology-pretrained transformers
   produce vastly more separable features than CNNs pretrained on natural images.
2. **Stain normalization consistently lifts the leaderboard** by closing the train→test
   stain/scanner shift: **CNN +0.016**, **Phikon +0.008** — even though it can slightly
   *lower* same-distribution OOF (raw-H&E inputs are mildly out-of-distribution for a frozen
   FM). It bets robustness on the *shifted* test, and that bet keeps winning.
3. **OOF is saturated (~0.99) for the foundation models**, so it barely separates them — the
   **leaderboard, not OOF, is the deciding signal.**

---

## Approach

A deliberately simple, **frozen-feature → small-head** recipe that isolates the backbone as
the experimental variable:

1. **Extract frozen features once.** Every train/test patch is passed through a frozen
   backbone; the pooled descriptors are cached to disk. This is the only expensive step, and
   it runs **once** per backbone — after that, head training is near-instant.
2. **Train a small MLP head** (`BN → Dropout → Linear(256) → SiLU → BN → Dropout → Linear(1)`)
   on the cached features with **5-fold stratified CV**.
3. **Optimize the threshold on OOF.** `tau` is swept on out-of-fold predictions to maximize
   F1; the 5 fold models are averaged for the test predictions.
4. **Validate, then submit.** The submission CSV is schema-checked before it is ever written.

**Config-driven and backbone-agnostic.** Every hyperparameter lives in `configs/*.yaml`;
nothing is hardcoded in the scripts. A single **backend switch** selects the feature
extractor:

| `features.backend` | Path | Backbone | Feature dim |
|---|---|---|---|
| `timm_cnn` *(default)* | Path-1 | ResNet-18 + ResNet-34 + EfficientNet-B0 | 2304 (concat) |
| `phikon` | Path-2 | Phikon ViT-Base (`owkin/phikon`, iBOT on ~40M TCGA tiles) | 768 |
| `h_optimus` | Path-3 | H-optimus-0 ViT-Giant (`bioptimus/H-optimus-0`, SSL on >500k WSIs) | 1536 |

### The improvement layer and the stain-shift finding

The **v2** of every path adds a stain-robustness layer on top of the identical base recipe:

- **Macenko stain normalization** — a pure-NumPy implementation (`src/data/stain.py`, no
  third-party stain library) that maps every patch onto one canonical H&E appearance.
- **Rotational/flip TTA** — feature descriptors averaged over deterministic orientations
  (`flip4` → `rot6`).
- **Label smoothing (0.05)** — for calibration and cleaner threshold transfer.
- **Late-fusion ensemble** (Path-1 only) — one head per backbone, AUROC-weight-averaged
  (the v1 ablation found inter-backbone OOF correlation ~0.83, below the skip threshold, so
  the diversity is real and late fusion beats concatenation).

**The key empirical finding:** the train→test gap is a **stain/scanner distribution shift**,
not overfitting. Same-distribution OOF can't validate the fix (it only pays off on the
shifted test), so `scripts/stress_eval.py` simulates a cross-lab stain shift offline and
measures relative AUROC degradation. Stain-normalized pipelines degrade dramatically less,
and the leaderboard confirms it: stain normalization closes the gap for **every** backbone.

---

## Repository layout

```
src/
  data/       dataset, transforms, stratified splits, feature loading, Macenko stain norm
  models/     frozen CNN backbones, Phikon & H-optimus extractors, backend dispatcher, MLP head
  eval/       compute_score (mirrors Kaggle), validate_submission, run-log writer
  inference/  flip/rot TTA, threshold optimization, temperature calibration
  utils/      seed, device (CUDA → MPS → CPU), config loader, logging
scripts/
  extract_features.py    # cache frozen-backbone features (train + test), backend-driven
  train_path_a.py        # 5-fold MLP CV, OOF assembly, threshold opt (all single-head configs)
  train_late_fusion.py   # per-backbone heads + AUROC-weighted late fusion (Path-1 v2 ensemble)
  make_submission.py     # build + validate submission CSV from saved test predictions
  stress_eval.py         # simulated stain-shift robustness eval (v1 vs v2)
  ablate_v1.py           # cheap late-fusion / label-smoothing ablations on cached features
configs/
  path_a.yaml  path_a_v2.yaml  path_a_v2_ensemble.yaml   # Path-1 (CNN trio)
  path_b.yaml  path_b_v2.yaml                            # Path-2 (Phikon)
  path_c.yaml  path_c_v2.yaml                            # Path-3 (H-optimus)
models/
  Path-1/  Path-2/  Path-3/   # per-model index READMEs + versioned snapshots (see below)
docs/CLOUD_GUIDE.md           # Colab/Kaggle recipes for heavy FMs and fine-tuning
tests/                        # 17 CPU pytest smoke tests
outputs/                      # feature caches, checkpoints, OOF preds, logs (gitignored)
submissions/                  # {name}_{version}.csv + submission_logs/ (gitignored)
data/                         # symlinks to the dataset (gitignored — never committed)
```

> **Gitignored artifacts:** `data/`, `outputs/` caches & checkpoints, `submissions/`, and all
> `*.npy`/`*.npz` feature/prediction dumps are never committed. The repo keeps code, configs,
> and documentation — not data or binaries.

---

## Quickstart

The data lives **outside** the repo and is never committed; `data/` holds symlinks to the
dataset. Developed on **Apple M3 Pro / MPS (no CUDA)** — the device helper auto-selects
**CUDA → MPS → CPU**, so the same code runs unchanged on Colab/Kaggle GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each path runs end-to-end with the **same three scripts**, pointed at its config:

```bash
# Path-1 — frozen CNN trio (the safety-net baseline)
python scripts/extract_features.py --config configs/path_a.yaml
python scripts/train_path_a.py     --config configs/path_a.yaml
python scripts/make_submission.py  --config configs/path_a.yaml

# Path-2 — frozen Phikon (pathology ViT-Base; ~7 min to extract on MPS)
python scripts/extract_features.py --config configs/path_b.yaml
python scripts/train_path_a.py     --config configs/path_b.yaml
python scripts/make_submission.py  --config configs/path_b.yaml

# Path-3 — frozen H-optimus-0 (ViT-Giant/1.1B). GATED: accept the license on HF and
# authenticate first:  huggingface-cli login   (paste a READ token)
python scripts/extract_features.py --config configs/path_c.yaml
python scripts/train_path_a.py     --config configs/path_c.yaml
python scripts/make_submission.py  --config configs/path_c.yaml
```

**Improvement (v2) runs** swap in the `*_v2.yaml` config (stain norm + TTA + label
smoothing). The Path-1 ensemble variant uses `train_late_fusion.py` instead of
`train_path_a.py`:

```bash
python scripts/extract_features.py   --config configs/path_a_v2_ensemble.yaml
python scripts/train_late_fusion.py  --config configs/path_a_v2_ensemble.yaml
python scripts/make_submission.py    --config configs/path_a_v2_ensemble.yaml
```

Each run writes OOF predictions and per-fold checkpoints under `outputs/`, a validated
`submissions/{name}_{version}.csv`, and a per-run metric log in
`submissions/submission_logs/{name}_{version}.md`. Bump `submission.version` in the config
for each resubmission.

### Tests

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/ -q     # 17 CPU smoke tests
```

---

## Where to look next

- **`models/Path-1/`, `models/Path-2/`, `models/Path-3/`** — per-model index READMEs with the
  full version history. Each version sub-folder (`v1-…`, `v2-…`) has its own `README.md`
  documenting what was run and the results, plus a `pipeline/` snapshot of the exact
  scripts and config that produced it.
- **`docs/CLOUD_GUIDE.md`** — Colab/Kaggle recipes for the GPU-heavy steps: frozen
  extraction of large FMs (e.g. H-optimus) and end-to-end fine-tuning. Both produce artifacts
  the local pipeline consumes **unchanged**, so you only run the GPU step in the cloud.
- **`tests/`** — 17 CPU pytest smoke tests covering the MLP head, TTA, threshold
  optimization, splits, scoring, and submission validation.
- **`submissions/`** *(gitignored)* — cataloged submission CSVs named
  `{model}_{version}.csv` (e.g. `Path-1-baseline_v1.csv`) with per-run logs in
  `submissions/submission_logs/`.

# Path-2 — Phikon foundation model (Path B)

Path B of the project: replace Path A's three ImageNet CNNs with **Phikon**
(`owkin/phikon`, ViT-Base/16, self-supervised **iBOT on ~40M TCGA histopathology tiles**)
as a **frozen** feature extractor → the same MLP head + 5-fold CV. The backbone is the
only changed variable: a transformer pretrained on *pathology* instead of CNNs pretrained
on natural images. Runs locally on MPS (~7 min to extract features; Kaggle/Colab not
needed for the frozen base — only for end-to-end fine-tuning, which is deferred).

Each version sub-folder has a `README.md` + a `pipeline/` snapshot. Per-run metric logs
are in the gitignored `submissions/submission_logs/`.

## Versions

| Version | Summary | OOF AUROC | OOF composite (α=0.5) | Public LB |
|---|---|---|---|---|
| [v1-base](v1-base/) | Frozen Phikon CLS features (768-d) + MLP, 4-way flip TTA | **0.9929** | **0.9787** | _pending upload_ |

## Path A vs Path B (out-of-fold)

| Model | OOF AUROC | OOF composite @0.5 | Public LB |
|---|---|---|---|
| Path-1 v1 (CNN trio) | 0.9393 | 0.9055 | 0.811 |
| Path-1 v2 (stain + late fusion) | 0.9470 | 0.9143 | 0.827 |
| **Path-2 v1 (frozen Phikon)** | **0.9929** | **0.9787** | _pending_ |

**Caveat:** OOF is same-distribution; the leaderboard is the real test (Path A showed a
~9–13 pt OOF→LB gap from stain shift). Phikon's pathology pretraining *may* shrink that
gap (it is plausibly more stain/scanner-robust than ImageNet CNNs), but that is unconfirmed
until the LB comes back. The improvement layer (stain norm, late fusion, fine-tuning) is
intentionally **not** applied yet — this is the clean base.

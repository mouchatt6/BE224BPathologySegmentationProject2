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
| [v1-base](v1-base/) | Frozen Phikon CLS features (768-d) + MLP, 4-way flip TTA | **0.9929** | **0.9787** | **0.882** |
| [v2-stainnorm](v2-stainnorm/) | + Macenko stain norm, rot6 TTA, label smoothing 0.05 | 0.9890 | ~0.974 | **0.890** |

## Path A vs Path B (out-of-fold)

| Model | OOF AUROC | OOF composite @0.5 | Public LB |
|---|---|---|---|
| Path-1 v1 (CNN trio) | 0.9393 | 0.9055 | 0.811 |
| Path-1 v2 (stain + late fusion) | 0.9470 | 0.9143 | 0.827 |
| **Path-2 v1 (frozen Phikon)** | **0.9929** | **0.9787** | **0.882** |
| Path-2 v2 (Phikon + stain norm) | 0.9890 | ~0.974 | **0.890** |

**Phikon's ~10-pt OOF→LB gap (0.979 → 0.882) confirms the test set is genuinely shifted.**
The improvement layer matters: for Phikon, Macenko stain norm slightly lowers clean OOF
(raw-H&E inputs are out-of-distribution for it) but makes it **5.8× more robust** to stain
shift in the stress eval — so v2 bets that robustness wins on the shifted LB, mirroring
Path A v1→v2 (0.811→0.827). See [v2-stainnorm/](v2-stainnorm/). v1-base (0.882) is the
Path B fallback. Fine-tuning is the next deferred step.

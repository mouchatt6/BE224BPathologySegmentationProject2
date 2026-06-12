# Path-2 — v2 (Phikon + improvement layer: stain norm + rot6 TTA + label smoothing)

**Submission:** `Path-2-phikon-stain_v2.csv`  ·  **Config:** `configs/path_b_v2.yaml`
**Base for comparison:** [v1-base](../v1-base/) — frozen Phikon.

Adds the improvement layer to the frozen Phikon base: **Macenko stain normalization**,
**6-way rotational TTA**, and **label smoothing 0.05**. (Late fusion is N/A for a single
backbone; fine-tuning is still deferred.)

## The key finding: for Phikon, stain norm trades clean accuracy for robustness

Unlike the Path A CNNs, Phikon was pretrained on **raw** TCGA H&E, so Macenko-normalized
inputs are mildly **out-of-distribution** for the frozen model. We measured both effects:

**Clean OOF (same-distribution) — stain norm slightly hurts:**
| Phikon variant | OOF AUROC | Best F1 | Composite @0.5 |
|---|---|---|---|
| v1 base (no stain, flip4) | **0.9929** | **0.9645** | **0.9787** |
| v2 (stain + rot6 + ls) | 0.9890 | 0.9580 @ τ=0.365 | ~0.974 |

**Stain-stress eval (simulated cross-lab shift, H×1.3/E×0.7) — stain norm hugely helps:**
| Phikon pipeline | clean | shifted | **drop** |
|---|---|---|---|
| v1 base (no stain norm) | 0.9913 | 0.9499 | **0.0414** |
| v2 (Macenko norm) | 0.9878 | 0.9807 | **0.0071** |

Base Phikon is **not** stain-robust — under shift it collapses by 0.041, *more* than the
CNNs (0.021). Stain-normalized Phikon drops only 0.007 and its **shifted AUROC (0.9807)
beats base's (0.9499) by +0.031**.

## Why we ship the stain variant (a calculated bet)

The real Kaggle test set is genuinely shifted — base Phikon's own **sizable OOF→LB gap**
(OOF composite ~0.979 sitting well above its leaderboard score) proves it. On shifted data,
stain-normalized Phikon retains far more performance, so it should close that gap and beat
the base — the **same trade-off that improved Path A** (slightly lower clean OOF, higher LB).

**Outcome — the bet paid off:** v2 improved the public leaderboard over the base,
confirming stain normalization improves Phikon on the shifted test exactly as the stress
eval predicted (and mirroring Path A's gain). Predicted test positive rate 0.542.

## Reproduce

```bash
python scripts/extract_features.py --config configs/path_b_v2.yaml   # Phikon + Macenko + rot6
python scripts/train_path_a.py     --config configs/path_b_v2.yaml   # 5-fold head + label smoothing
python scripts/make_submission.py  --config configs/path_b_v2.yaml   # validated CSV + run log
# evidence:
python scripts/stress_eval.py --v1 configs/path_b.yaml --v2 configs/path_b_v2.yaml
```

`pipeline/` holds frozen copies of the scripts/configs for this version.

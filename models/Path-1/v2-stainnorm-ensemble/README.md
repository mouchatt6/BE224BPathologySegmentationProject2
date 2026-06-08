# Path-1 — v2 (stain normalization + late-fusion ensemble)

**Submission:** `Path-1-stainnorm-ens_v2.csv`  ·  **Public LB:** _pending upload_  ·  **Configs:** `configs/path_a_v2_ensemble.yaml`

v2 attacks the **~9-point OOF→LB gap** that v1 exposed (OOF composite 0.905 → LB 0.811),
which is the signature of train/test **stain distribution shift**. Four evidence-backed
changes, all keeping v1's frozen ResNet-18 + ResNet-34 + EfficientNet-B0 backbones:

| Change | Why | Evidence |
|---|---|---|
| **Macenko stain normalization** | map every patch to one canonical H&E appearance so the *frozen* nets see consistent color (we can't augment a frozen net stain-invariant) | [Tellez 2019](https://arxiv.org/abs/1902.06543); our stress eval below |
| **6-way rotational TTA** (was 4-way flip) | more deterministic views averaged; H&E is rotation-invariant | — |
| **Label smoothing 0.05** | calibration + threshold transfer | [Müller 2019](https://arxiv.org/abs/1906.02629); +0.006 OOF (ablation) |
| **Late-fusion ensemble** (head per backbone, AUROC-weighted avg) | per-backbone OOF correlation only ~0.83 (< 0.97 skip gate, §8.2) → real diversity | +0.0055 OOF (ablation) |

## Results (out-of-fold, 5-fold stratified CV)

| Model | OOF AUROC | Best F1 | Composite @0.5 |
|---|---|---|---|
| v1 (concat, no stain) | 0.9393 | 0.8716 @ τ=0.520 | 0.9055 |
| v2 concat (stain) | 0.9422 | 0.8751 @ τ=0.460 | 0.9086 |
| **v2 late-fusion (stain) — shipped** | **0.9470** | **0.8815 @ τ=0.450** | **0.9143** |

Composite is stable across α∈[0.1, 0.9] (see `submissions/submission_logs/Path-1-stainnorm-ens_v2.md`).
Predicted test positive rate 0.520.

## Why this is expected to beat v1 on the LB: the stain-stress eval

Same-distribution OOF *cannot* show stain robustness (it pays off only on the shifted
test set). We simulated a cross-lab stain shift (hematoxylin ×1.3, eosin ×0.7) on
held-out patches and measured AUROC degradation (`scripts/stress_eval.py`):

| pipeline | clean | shifted | **drop** |
|---|---|---|---|
| v1 (no stain norm) | 0.9220 | 0.9010 | **0.0210** |
| v2 (Macenko) | 0.9164 | 0.9115 | **0.0048** |

Under shift, v2 degrades **4.4× less** and its *shifted* AUROC (0.9115) beats v1's (0.9010)
by +0.0105. Macenko slightly lowers clean AUROC (it discards some discriminative color)
but is far more robust — and the Kaggle test set is precisely the shifted scenario.

## Ablations (`scripts/ablate_v1.py`)

- **Label smoothing sweep** (concat head): 0.0 → 0.9055, **0.05 → 0.9117**, 0.1 → 0.9095.
- **Late-fusion gate**: inter-backbone OOF correlation 0.82–0.85 (< 0.97), and averaging
  3 per-backbone heads beat concatenation (0.9110 vs 0.9055). Late fusion kept.

## What we did NOT do (and why)
- **"Tune α"**: α is the graders' fixed hidden weight — not tunable. We report α-sensitivity
  instead (composite stays strong across α). The controllable analog, threshold τ, is
  OOF-optimized (τ\*=0.450, stable).
- **Heavier head regularization**: low-leverage; the gap is distribution shift, not head
  overfitting. The high-value "regularizer" was stain normalization, applied at the input.

## Reproduce

```bash
python scripts/extract_features.py   --config configs/path_a_v2.yaml          # stain + rot6 features
python scripts/train_late_fusion.py  --config configs/path_a_v2_ensemble.yaml # per-backbone heads + fusion
python scripts/make_submission.py    --config configs/path_a_v2_ensemble.yaml # validated CSV + run log
# supporting analysis:
python scripts/ablate_v1.py --config configs/path_a.yaml      # ablations
python scripts/stress_eval.py --v1 configs/path_a.yaml --v2 configs/path_a_v2.yaml
```

`pipeline/` holds frozen copies of the v2 scripts/configs. Tip: prefix extraction with
`VECLIB_MAXIMUM_THREADS=1` (now set automatically in the script) to avoid BLAS
oversubscription slowing the per-patch Macenko step.

# Path-3 — v1 (frozen H-optimus-0 base)

**Submission:** `Path-3-hoptimus-base_v1.csv`  ·  **Public LB:** _pending upload_  ·  **Config:** `configs/path_c.yaml`

Frozen **H-optimus-0** (`bioptimus/H-optimus-0`, ViT-Giant/14, **1.1B params**, SSL on
>500k WSIs) as a feature extractor → the same MLP head as the other paths. Clean base:
no stain norm / TTA / fine-tuning.

## What was executed

| Component | Setting |
|---|---|
| Backbone (frozen, **histology SSL**) | H-optimus-0 ViT-Giant/14 → **1536-dim** feature |
| Access | **Gated** — HF license + token required |
| Feature-extraction TTA | **none** (single view) — ViT-Giant is ~5.9 img/s on MPS; 4-way would be ~110 min |
| Normalization | H-optimus's own stats mean=(0.7072,0.5787,0.7036) std=(0.2119,0.2301,0.1775), **not ImageNet** |
| Cross-validation | 5-fold stratified |
| Head | `BN → Dropout(0.15) → Linear(1536→256) → SiLU → BN → Dropout(0.25) → Linear(256→1)` |
| Optimizer / loss | AdamW lr=2e-3 wd=1e-3, batch 256, BCE, label smoothing 0.0 |
| Post-processing | OOF-optimized threshold τ\*=0.535 (stable); no calibration |

Compute: frozen extraction ~25 min on Apple M3 Pro / MPS (single view, ~6 img/s);
head training ~30 s. Extraction logged after the user accepted the H-optimus license and
ran `hf auth login`.

## Results (out-of-fold)

| Metric | Value |
|---|---|
| OOF AUROC | **0.9897** (per-fold 0.9862 / 0.9918 / 0.9897 / 0.9923 / 0.9886) |
| Best F1 @ τ\*=0.535 | **0.9541** (stable) |
| Predicted test positive rate | 0.505 |

### vs the other paths
| Model | OOF AUROC | OOF F1 | Public LB |
|---|---|---|---|
| Path-1 v2 (CNN + stain + fusion) | 0.9470 | 0.8815 | 0.827 |
| Path-2 v1 (Phikon) | 0.9929 | 0.9645 | 0.882 |
| **Path-3 v1 (H-optimus)** | 0.9897 | 0.9541 | _pending_ |

## Caveats
- **OOF is saturated** (~0.99) and **H-optimus used no TTA** vs Phikon's flip4, so OOF
  does not show H-optimus ahead. The **leaderboard** is the deciding signal — whether the
  larger 1.1B model's features close the shifted-test gap better than Phikon's.
- If the LB is promising: add 4-way TTA (~110 min) and/or the stain-norm robustness layer
  (`features.tta: rot6`, `features.stain_norm: true`), which helped every other backbone.

## Reproduce

Requires HF access to H-optimus-0 (accept license + `hf auth login`).

```bash
python scripts/extract_features.py --config configs/path_c.yaml   # frozen H-optimus, single view (~25 min MPS)
python scripts/train_path_a.py     --config configs/path_c.yaml   # 5-fold MLP head
python scripts/make_submission.py  --config configs/path_c.yaml   # validated CSV + run log
```

`pipeline/` holds frozen copies of the scripts/config for this version.

# Path-3 — H-optimus-0 foundation model (Path C)

A second pathology foundation model, swapped in as a **frozen** feature extractor →
the same MLP head + 5-fold CV. **H-optimus-0** (`bioptimus/H-optimus-0`) is a
**ViT-Giant/14, 1.1B-param** model self-supervised on >500k H&E whole-slide images —
much larger than Phikon-v1 (ViT-Base), and a top-tier *open* (license-gated) model on
2025 benchmarks.

**Access:** gated — requires accepting the license on HF + an auth token. **Compute:**
heavy on MPS (~5.9 img/s), so the base uses single-view (no TTA) to keep extraction ~25
min; its non-ImageNet normalization is config-driven (`features.norm_mean/std`).

## Versions

| Version | Summary | OOF AUROC | OOF F1 | Public LB |
|---|---|---|---|---|
| [v1-base](v1-base/) | Frozen H-optimus CLS features (1536-d) + MLP, single-view | 0.9897 | 0.9541 | _pending upload_ |

## Foundation-model comparison (out-of-fold)

| Model | Backbone | OOF AUROC | OOF F1 | TTA | Public LB |
|---|---|---|---|---|---|
| Path-2 v1 (Phikon) | ViT-B, 768-d | 0.9929 | 0.9645 | flip4 | **0.882** |
| **Path-3 v1 (H-optimus)** | ViT-G/1.1B, 1536-d | 0.9897 | 0.9541 | none | _pending_ |

**Read this carefully:** on this patch task the OOF is **saturated** (~0.99 for both
foundation models), so OOF barely separates them — and H-optimus here is mildly
handicapped by using **no TTA** (vs Phikon's 4-way). So OOF does **not** show H-optimus
beating Phikon. Whether its larger/stronger features generalize better is decided only by
the **leaderboard** (the shifted-test gap). Predicted test positive rate 0.505 (the most
balanced of any model so far). If H-optimus is promising on the LB, the obvious next
squeezes are 4-way TTA (~110 min) and the stain-norm robustness layer.

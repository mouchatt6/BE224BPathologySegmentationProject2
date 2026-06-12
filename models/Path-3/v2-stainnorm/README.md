# Path-3 — v2 (H-optimus-0 + stain-normalization layer)

**Submission:** `Path-3-hoptimus-stain_v2.csv`  ·  **Config:** `configs/path_c_v2.yaml`
**Base for comparison:** [v1-base](../v1-base/) — frozen H-optimus, no stain norm.

Applies the improvement layer to the leaderboard-leading frozen H-optimus base: **Macenko
stain normalization** + **label smoothing 0.05**, single-view (no TTA — H-optimus is
~5.9 img/s on MPS, and stain norm, not TTA, is the proven robustness lever). Stain
normalization has lifted every other backbone's leaderboard score, so we test it here.

## Clean OOF — neutral (unlike Phikon)

| Variant | OOF AUROC | Best F1 |
|---|---|---|
| v1 base (no stain) | 0.9897 | 0.9541 @ τ=0.535 |
| v2 (stain + label smoothing) | 0.9894 | 0.9535 @ τ=0.340 |

Stain normalization is essentially **OOF-neutral** for H-optimus (0.9897 → 0.9894) — it
does **not** lower clean OOF the way it did for Phikon (which dropped ~0.004). H-optimus's
larger, more diverse pretraining may make its features less sensitive to the normalization.
As always, same-distribution OOF can't measure stain robustness — that's the stress eval.

## Stain-stress eval (simulated cross-lab shift, n=1500)

`scripts/stress_eval.py` applies a fixed H&E shift (H×1.3 / E×0.7) to held-out patches and
measures AUROC degradation:

| H-optimus pipeline | clean | shifted | **drop** |
|---|---|---|---|
| base (no stain norm) | 0.9804 | 0.9689 | **0.0115** |
| v2 (Macenko norm) | 0.9836 | 0.9825 | **0.0011** |

Two things stand out. First, **base H-optimus is already noticeably more stain-robust than
the smaller models** (drop 0.0115 vs CNN 0.021, Phikon 0.041) — the 1.1B model generalizes
better out of the box. Second, **stain normalization still makes it ~10× more robust**
(drop 0.0011), and its *shifted* AUROC beats base's by ~0.014.

## Why we ship v2 (a cleaner bet than Phikon's)

For Phikon, stain norm cost clean OOF but bought robustness. For **H-optimus it's
OOF-neutral *and* far more robust** — so there's no trade-off to weigh: v2 dominates the
base on the shifted-distribution proxy at no same-distribution cost. Since the real test
set is shifted, v2 should hold up at least as well as the base and likely better. Stain
normalization has now improved robustness for **every** backbone tried.

## Reproduce

Requires HF access to H-optimus-0 (accept license + `hf auth login`).

```bash
python scripts/extract_features.py --config configs/path_c_v2.yaml   # H-optimus + Macenko, single view
python scripts/train_path_a.py     --config configs/path_c_v2.yaml   # 5-fold head + label smoothing
python scripts/make_submission.py  --config configs/path_c_v2.yaml   # validated CSV + run log
python scripts/stress_eval.py --v1 configs/path_c.yaml --v2 configs/path_c_v2.yaml --n 1500
```

`pipeline/` holds frozen copies of the scripts/configs for this version.

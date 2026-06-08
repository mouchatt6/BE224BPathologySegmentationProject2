# Path-1 — Frozen-feature MLP head model (Path A)

Path A of the project: three **frozen** ImageNet CNNs (ResNet-18 + ResNet-34 +
EfficientNet-B0) → concatenated features → MLP head, 5-fold CV, OOF threshold
optimization. This is the safety-net baseline; each iteration is documented as its own
version below.

Each version sub-folder contains a `README.md` (what was run + results) and a
`pipeline/` snapshot of the scripts/config that produced it. Auto-generated per-run
metric logs live separately in the gitignored `submissions/submission_logs/`.

## Versions

| Version | Summary | OOF AUROC | OOF composite (α=0.5) | Public LB |
|---|---|---|---|---|
| [v1-baseline](v1-baseline/) | Frozen 3-CNN features + MLP, 4-way flip TTA | 0.9393 | 0.9055 | **0.811** |
| [v2-stainnorm-ensemble](v2-stainnorm-ensemble/) | + Macenko stain norm, rot6 TTA, label smoothing 0.05, late fusion | **0.9470** | **0.9143** | _pending upload_ |

**The v1→v2 story:** v1's ~9-point OOF→LB gap (0.905 → 0.811) is train/test stain
distribution shift. v2 targets it with Macenko stain normalization (a stress test shows
v2 degrades **4.4× less** under simulated stain shift), and stacks label smoothing +
late fusion (both validated by ablation) for a higher OOF too. See
[v2-stainnorm-ensemble/](v2-stainnorm-ensemble/) for the full evidence.

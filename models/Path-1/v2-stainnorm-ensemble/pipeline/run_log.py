"""Per-run markdown logs — one file per model run, cataloged alongside submissions.

Each run writes a self-contained markdown report to ``submissions/submission_logs/``
capturing (a) the headline metrics — AUROC, best F1, and the composite at every alpha
— and (b) a summary of *what was executed* (backbones, CV, head, hyperparameters,
post-processing). Diffing two run logs shows exactly what changed between runs, which
is what the final report's ablation narrative needs.

Note: AUROC and F1 do not depend on alpha (alpha only weights the AUROC vs F1 mix), so
they are reported once; only the composite is tabulated per alpha.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _summary_lines(cfg: dict[str, Any], fusion_weights: dict | None = None) -> list[str]:
    """Build the bullet list describing what the run executed, from the config.

    Args:
        cfg: The parsed experiment config.
        fusion_weights: If present, this run used late fusion (a head per backbone,
            probability-averaged with these weights) rather than feature concatenation.

    Returns:
        A list of markdown bullet strings summarizing the pipeline for this run.
    """
    feat, head, train, post = cfg["features"], cfg["head"], cfg["train"], cfg["postprocess"]
    stain = "on (Macenko)" if feat.get("stain_norm", False) else "off"
    if fusion_weights:
        fusion = ", ".join(f"{k} {v:.2f}" for k, v in fusion_weights.items())
        backbone_line = (f"- **Backbones:** {' + '.join(feat['backbones'])} (frozen, ImageNet) "
                         f"→ per-backbone heads, **late fusion** (AUROC-weighted: {fusion})")
    else:
        backbone_line = (f"- **Backbones:** {' + '.join(feat['backbones'])} "
                         f"(frozen, ImageNet) → {feat['feature_dim']}-dim concatenation")
    return [
        backbone_line,
        f"- **Stain normalization:** {stain}",
        f"- **Feature-extraction TTA:** {feat['tta']}",
        f"- **Cross-validation:** {cfg['data']['n_splits']}-fold stratified",
        f"- **Head:** BN → Dropout({head['p_drop1']}) → Linear({feat['feature_dim']}→{head['hidden']}) "
        f"→ SiLU → BN → Dropout({head['p_drop2']}) → Linear({head['hidden']}→1)",
        f"- **Optimizer:** AdamW lr={train['lr']} wd={train['weight_decay']}, "
        f"batch={train['batch_size']}, ≤{train['max_epochs']} epochs, "
        f"early-stop patience={train['early_stop_patience']} on val {train['monitor'].upper()}",
        f"- **Label smoothing:** {train.get('label_smoothing', 0.0)}",
        f"- **Post-processing:** threshold opt={post.get('optimize_threshold', True)}, "
        f"calibration={post.get('calibrate', False)}",
        f"- **Seed:** {cfg['seed']} (deterministic={cfg.get('deterministic', False)})",
    ]


def write_run_log(
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    submission: dict[str, Any],
    out_dir: str | Path,
    timestamp: str,
) -> Path:
    """Write a markdown run log and return its path.

    Args:
        cfg: Parsed experiment config.
        metrics: The training sidecar dict (oof_auroc, oof_f1, tau, fold_aurocs,
            report{auroc,f1,score@a...}, threshold_stability, temperature, git_sha,
            config_hash).
        submission: Submission stats: {"filename", "n_rows", "positive_rate"}.
        out_dir: Directory for the log (e.g. submissions/submission_logs).
        timestamp: Human-readable run timestamp string.

    Returns:
        Path to the written ``.md`` file (named after the submission stem).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name, version = cfg["submission"]["name"], cfg["submission"]["version"]
    report = metrics.get("report", {})
    fold_aurocs = metrics.get("fold_aurocs", [])
    stab = metrics.get("threshold_stability", {})

    L: list[str] = []
    L.append(f"# Run Log — {name} ({version})")
    L.append("")
    L.append(f"- **Run timestamp:** {timestamp}")
    L.append(f"- **Submission file:** `{submission['filename']}`")
    L.append(f"- **Model / path:** Path A — Frozen-feature MLP head model")
    L.append(f"- **Git SHA:** `{metrics.get('git_sha') or 'n/a'}`  |  "
             f"**Config hash:** `{metrics.get('config_hash', 'n/a')}`")
    L.append("")

    fusion_weights = metrics.get("fusion_weights")
    L.append("## What was executed")
    L.extend(_summary_lines(cfg, fusion_weights))
    L.append("")

    L.append("## Results (out-of-fold)")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| OOF AUROC | {metrics.get('oof_auroc', float('nan')):.4f} |")
    tau = metrics.get("tau", float("nan"))
    f1 = metrics.get("oof_f1", float("nan"))
    stable = stab.get("f1_stable")
    L.append(f"| Best F1 @ τ\\*={tau:.3f} | {f1:.4f}"
             + (f" (stable={stable})" if stable is not None else "") + " |")
    if fold_aurocs:
        per_fold = ", ".join(f"{a:.4f}" for a in fold_aurocs)
        mean = sum(fold_aurocs) / len(fold_aurocs)
        label = "Per-backbone OOF AUROC" if fusion_weights else "Per-fold AUROC"
        L.append(f"| {label} | {per_fold} (mean {mean:.4f}) |")
    L.append("")

    # AUROC and F1 are alpha-independent; only the composite varies with alpha.
    L.append("### Composite score by α  (α·AUROC + (1−α)·F1)")
    L.append("")
    L.append("| α | AUROC | Best F1 | Composite |")
    L.append("|---|---|---|---|")
    auroc_r = report.get("auroc", metrics.get("oof_auroc", float("nan")))
    f1_r = report.get("f1", f1)
    for a in cfg["report_alphas"]:
        comp = report.get(f"score@{a}", a * auroc_r + (1 - a) * f1_r)
        L.append(f"| {a} | {auroc_r:.4f} | {f1_r:.4f} | {comp:.4f} |")
    L.append("")

    L.append("## Submission")
    L.append("")
    L.append(f"- **Rows:** {submission['n_rows']}")
    L.append(f"- **Decision threshold τ\\*:** {tau:.3f}")
    L.append(f"- **Predicted positive rate:** {submission['positive_rate']:.3f}")
    if metrics.get("temperature", 1.0) != 1.0:
        L.append(f"- **Temperature (calibration):** {metrics['temperature']:.4f}")
    L.append("")

    out_path = out_dir / f"{name}_{version}.md"
    out_path.write_text("\n".join(L))
    return out_path

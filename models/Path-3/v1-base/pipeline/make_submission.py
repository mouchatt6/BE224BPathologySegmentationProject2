"""Build submission.csv from the saved test predictions, then validate it.

Reads the .npz written by train_path_a.py (fold-averaged test probabilities, the
chosen threshold tau, and the test img_paths), aligns rows to dummyTest.csv exactly,
derives integer labels with tau, and writes a schema-correct CSV. The submission is
ALWAYS passed through validate_submission() before being written for upload — the
Kaggle grader fails silently on dtype/column/order mistakes.

Usage:
    python scripts/make_submission.py --config configs/path_a.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.eval.run_log import write_run_log
from src.eval.validate import validate_submission
from src.utils.config import REPO_ROOT, load_config, resolve_path
from src.utils.logging_utils import get_logger


def main() -> None:
    """Entry point: assemble, validate and write the Path A submission CSV."""
    parser = argparse.ArgumentParser(description="Make Path A submission.csv.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(f"{cfg['experiment_name']}_submit")

    # --- Load saved predictions from training. ---
    oof = np.load(resolve_path(cfg["outputs"]["oof_path"]), allow_pickle=True)
    test_probs = oof["test_probs"].astype(float)
    test_paths = [str(p) for p in oof["test_paths"]]
    tau = float(oof["tau"])
    logger.info(f"Loaded {len(test_probs)} test predictions | tau={tau:.4f}")

    # --- Load the official template; its img_path column defines the required order. ---
    dummy_df = pd.read_csv(resolve_path(cfg["data"]["dummy_csv"]))

    # Align predictions to the template's exact row order. We keyed predictions by
    # img_path, so reindex by the template's paths rather than trusting positional order.
    prob_by_path = dict(zip(test_paths, test_probs))
    missing = [p for p in dummy_df["img_path"] if p not in prob_by_path]
    assert not missing, f"{len(missing)} template paths have no prediction, e.g. {missing[:3]}"
    aligned_probs = np.array([prob_by_path[p] for p in dummy_df["img_path"]], dtype=float)

    # Clip away exact 0/1 to avoid numerical edge cases in the grader.
    aligned_probs = np.clip(aligned_probs, 1e-6, 1.0 - 1e-6)
    # Derive integer labels from tau; cast explicitly to int (NOT bool/float).
    labels = (aligned_probs >= tau).astype(int)

    # --- Build the submission frame in the exact required column order. ---
    sub_df = pd.DataFrame({
        "img_path": dummy_df["img_path"].tolist(),
        "label": labels,
        "probabilities": aligned_probs,
    })

    # --- Non-negotiable: validate before writing. Raises on any schema problem. ---
    validate_submission(sub_df, dummy_df)
    logger.info("validate_submission() passed.")

    # Catalog the submission by model name + iteration, e.g. "Path-1-baseline_v1.csv",
    # in the gitignored top-level submissions/ folder (CSVs are never pushed to GitHub).
    sub_dir = resolve_path(cfg["outputs"]["submission_dir"])
    sub_dir.mkdir(parents=True, exist_ok=True)
    out_path = sub_dir / f"{cfg['submission']['name']}_{cfg['submission']['version']}.csv"
    if out_path.exists():
        logger.warning(f"Overwriting existing submission {out_path.name} "
                       f"(bump submission.version in the config to keep both).")
    # index=False -> CSV has exactly [img_path, label, probabilities], matching dummyTest.csv.
    sub_df.to_csv(out_path, index=False)
    logger.info(f"Wrote submission to {out_path}")
    logger.info(f"Positive-rate in submission: {labels.mean():.3f} ({int(labels.sum())}/{len(labels)})")

    # --- Write a per-run markdown log cataloging metrics + what was executed. ---
    # Metrics come from the training sidecar JSON (written next to the OOF .npz).
    sidecar_path = resolve_path(cfg["outputs"]["oof_path"]).with_suffix(".json")
    metrics = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    submission_stats = {
        "filename": str(out_path.relative_to(REPO_ROOT)),
        "n_rows": len(sub_df),
        "positive_rate": float(labels.mean()),
    }
    log_path = write_run_log(
        cfg,
        metrics,
        submission_stats,
        out_dir=sub_dir / "submission_logs",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    logger.info(f"Wrote run log to {log_path}")


if __name__ == "__main__":
    main()

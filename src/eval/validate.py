"""Pre-submission validation — run on EVERY submission before uploading (§10.3).

The Kaggle grader fails *silently* on dtype/column-order mistakes, so this guard is
non-negotiable. It checks the submission against ``dummyTest.csv`` (the official
template) so that columns, row count, and img_path ordering match byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_submission(sub_df: pd.DataFrame, dummy_df: pd.DataFrame) -> bool:
    """Assert that a submission DataFrame is structurally valid.

    Args:
        sub_df: The submission to upload, with columns
            ["img_path", "label", "probabilities"] in that order.
        dummy_df: The reference ``dummyTest.csv`` loaded as a DataFrame. Defines the
            required row count and exact img_path ordering.

    Returns:
        True if every check passes. Raises AssertionError otherwise.
    """
    # 1) Columns present and in the exact required order.
    assert list(sub_df.columns) == ["img_path", "label", "probabilities"], (
        f"Wrong columns: {sub_df.columns.tolist()}"
    )

    # 2) label must be a true integer dtype (NOT bool, NOT float — the grader breaks
    #    silently otherwise). is_integer_dtype is robust across numpy/pandas versions
    #    and, unlike a literal `dtype in (np.int64, np.int32)` check, treats bool as
    #    invalid (bool is not an integer dtype here).
    assert pd.api.types.is_integer_dtype(sub_df["label"]) and not pd.api.types.is_bool_dtype(
        sub_df["label"]
    ), f"label must be an integer dtype, got {sub_df['label'].dtype}"
    assert set(np.unique(sub_df["label"])).issubset({0, 1}), "label values must be in {0, 1}"

    # 3) probabilities must be finite floats in [0, 1] with no NaN.
    assert sub_df["probabilities"].notna().all(), "probabilities must not contain NaN"
    assert sub_df["probabilities"].between(0.0, 1.0).all(), "probabilities must be in [0, 1]"

    # 4) Row count must match the template exactly (2,400 test patches).
    assert len(sub_df) == len(dummy_df) == 2400, (
        f"Row count mismatch: {len(sub_df)} vs {len(dummy_df)} (expected 2400)"
    )

    # 5) img_path ordering must match the template byte-for-byte. The grader joins on
    #    position/path; any reordering or path edit corrupts the alignment silently.
    assert sub_df["img_path"].tolist() == dummy_df["img_path"].tolist(), (
        "img_path order must match dummyTest.csv exactly (byte-for-byte)"
    )

    # 6) No duplicate test paths.
    assert sub_df["img_path"].is_unique, "img_path values must be unique"

    return True

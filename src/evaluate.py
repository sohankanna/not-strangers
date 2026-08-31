"""Temporal split and the metrics reported in README.md, and only that.

This module must never be modified to improve reported numbers -- see
CLAUDE.md. It must also stay a pure metrics library with no CLI or
orchestration code: src/run_pipeline.py owns sequencing the pipeline and
calls into these functions. Mixing orchestration into this file would
eventually get it edited for orchestration reasons, eroding the
"never modify" guarantee.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def temporal_train_test_split(
    transactions: pd.DataFrame,
    dt_col: str = "TransactionDT",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into train/test by time, never randomly.

    Args:
        transactions: Raw transaction rows.
        dt_col: Name of the column to split on.
        test_size: Fraction of the most recent transactions (by dt_col) to
            hold out as the test set.

    Returns:
        A (train, test) tuple of DataFrames, split strictly on dt_col so that
        every train-set TransactionDT is earlier than every test-set
        TransactionDT.
    """
    raise NotImplementedError


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the precision-recall curve.

    Args:
        y_true: Ground-truth labels (chargeback-reported, noisy).
        y_score: Predicted abuse scores.

    Returns:
        The PR-AUC.
    """
    raise NotImplementedError


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01
) -> float:
    """Compute recall at a fixed false-positive rate.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted abuse scores.
        target_fpr: The false-positive rate to hold fixed while measuring
            recall.

    Returns:
        The recall achieved at the given target FPR.
    """
    raise NotImplementedError


def cost_per_10k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """Compute the expected business cost per 10,000 transactions.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted abuse scores.
        threshold: Score threshold above which a transaction is flagged.
        cost_fn: Cost of a missed abuse case (false negative).
        cost_fp: Cost of a false alarm (false positive).

    Returns:
        The expected cost per 10,000 transactions at the given threshold.
    """
    raise NotImplementedError


def evaluate_model(
    model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute the full reported metric set for a trained model.

    Args:
        model: A trained LightGBM booster.
        X_test: Held-out (temporally later) feature matrix.
        y_test: Held-out labels.

    Returns:
        A dict with keys matching the README results table (e.g. "pr_auc",
        "recall_at_1pct_fpr", "cost_per_10k").
    """
    raise NotImplementedError

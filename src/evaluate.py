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
from sklearn.metrics import average_precision_score, roc_curve


def temporal_train_test_split(
    transactions: pd.DataFrame,
    dt_col: str = "TransactionDT",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into train/test by time, never randomly.

    Walks backward from the most recent timestamp, accumulating whole
    dt_col groups until at least `test_size` of the rows are covered, so a
    group of rows sharing one TransactionDT value is never split across the
    train/test boundary -- even though that means the realized test
    fraction can differ slightly from `test_size` when large groups share a
    timestamp.

    Args:
        transactions: Raw transaction rows.
        dt_col: Name of the column to split on.
        test_size: Approximate fraction of the most recent transactions (by
            dt_col) to hold out as the test set.

    Returns:
        A (train, test) tuple of DataFrames, split strictly on dt_col so that
        every train-set TransactionDT is earlier than every test-set
        TransactionDT.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    sorted_df = transactions.sort_values(dt_col, kind="mergesort")
    n = len(sorted_df)
    target_test_n = round(n * test_size)

    counts_by_dt = sorted_df[dt_col].value_counts()
    unique_dts = np.sort(sorted_df[dt_col].unique())

    cumulative = 0
    split_dt = unique_dts[-1]
    for dt in unique_dts[::-1]:
        cumulative += counts_by_dt[dt]
        split_dt = dt
        if cumulative >= target_test_n:
            break

    train = sorted_df[sorted_df[dt_col] < split_dt]
    test = sorted_df[sorted_df[dt_col] >= split_dt]

    if len(train) and len(test):
        assert train[dt_col].max() < test[dt_col].min()

    return train, test


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the precision-recall curve.

    Args:
        y_true: Ground-truth labels (chargeback-reported, noisy).
        y_score: Predicted abuse scores.

    Returns:
        The PR-AUC (average precision).
    """
    return float(average_precision_score(y_true, y_score))


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01
) -> float:
    """Compute recall at a fixed false-positive rate.

    FPR is FP / (FP + TN), measured over the negative class. This dataset's
    base positive rate is roughly 3.5%, so a 1% FPR operating point means
    flagging real abuse while disturbing only a small slice of legitimate
    traffic -- it's a meaningful low-friction operating point at this base
    rate, not an arbitrary number.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted abuse scores.
        target_fpr: The false-positive rate to hold fixed while measuring
            recall.

    Returns:
        The highest recall (TPR) achievable at or below target_fpr.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    achievable = tpr[fpr <= target_fpr]
    if achievable.size == 0:
        return 0.0
    return float(achievable.max())


def cost_per_10k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """Compute the expected business cost per 10,000 transactions.

    cost = (FN * cost_fn + FP * cost_fp) / n * 10000

    Costs are fully parameterised (cost_fn, cost_fp) rather than hardcoded,
    since the real cost of a missed abuse case vs. a false alarm is a
    business decision, not a modeling constant.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted abuse scores.
        threshold: Score at or above which a transaction is flagged.
        cost_fn: Cost of a missed abuse case (false negative).
        cost_fp: Cost of a false alarm (false positive).

    Returns:
        The expected cost per 10,000 transactions at the given threshold.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    predicted_positive = y_score >= threshold

    fn = np.sum((y_true == 1) & ~predicted_positive)
    fp = np.sum((y_true == 0) & predicted_positive)
    n = len(y_true)

    return float((fn * cost_fn + fp * cost_fp) / n * 10000)


def evaluate_model(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
) -> dict[str, float]:
    """Compute the full reported metric set for a trained model.

    Calls model.predict directly (the LightGBM Booster API) rather than
    going through src.model, so this module has no dependency on model.py.

    Args:
        model: A trained LightGBM booster.
        X_test: Held-out (temporally later) feature matrix.
        y_test: Held-out labels.
        threshold: Score threshold passed to cost_per_10k.
        cost_fn: Cost of a missed abuse case, passed to cost_per_10k.
        cost_fp: Cost of a false alarm, passed to cost_per_10k. Both cost
            arguments default to a neutral 1:1 ratio; callers should pass
            real business costs when known.

    Returns:
        A dict with keys "pr_auc", "recall_at_1pct_fpr", "cost_per_10k".
    """
    y_score = model.predict(X_test)

    return {
        "pr_auc": pr_auc(y_test, y_score),
        "recall_at_1pct_fpr": recall_at_fpr(y_test, y_score, target_fpr=0.01),
        "cost_per_10k": cost_per_10k(y_test, y_score, threshold, cost_fn, cost_fp),
    }

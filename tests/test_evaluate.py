"""Tests for src/evaluate.py.

evaluate.py is frozen after this: it can never be edited later to make
numbers look better, so it has to be correct now. These tests are the real
coverage that backs that guarantee.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    cost_per_10k,
    evaluate_model,
    pr_auc,
    recall_at_fpr,
    temporal_train_test_split,
)


# --- temporal_train_test_split -----------------------------------------


def test_temporal_split_train_strictly_before_test():
    df = pd.DataFrame({"TransactionDT": np.arange(100), "value": np.arange(100)})
    train, test = temporal_train_test_split(df, test_size=0.2)

    assert len(train) + len(test) == len(df)
    assert train["TransactionDT"].max() < test["TransactionDT"].min()


def test_temporal_split_never_splits_a_tied_timestamp_group():
    # 45 uniquely-timestamped rows, then 20 rows all sharing TransactionDT=50
    # (straddling the naive 80/20 boundary), then 20 more unique rows.
    dts = list(range(45)) + [50] * 20 + list(range(70, 90))
    df = pd.DataFrame({"TransactionDT": dts})

    train, test = temporal_train_test_split(df, test_size=0.3)

    tied_in_train = int((train["TransactionDT"] == 50).sum())
    tied_in_test = int((test["TransactionDT"] == 50).sum())

    assert tied_in_train == 0 or tied_in_test == 0
    assert tied_in_train + tied_in_test == 20
    assert len(train) and len(test)
    assert train["TransactionDT"].max() < test["TransactionDT"].min()


def test_temporal_split_rejects_invalid_test_size():
    df = pd.DataFrame({"TransactionDT": np.arange(10)})
    with pytest.raises(ValueError):
        temporal_train_test_split(df, test_size=0.0)
    with pytest.raises(ValueError):
        temporal_train_test_split(df, test_size=1.0)


# --- pr_auc ---------------------------------------------------------------


def test_pr_auc_perfect_ranking_is_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_pr_auc_uncorrelated_scores_are_near_base_rate():
    rng = np.random.default_rng(0)
    n = 20_000
    base_rate = 0.035
    y_true = (rng.random(n) < base_rate).astype(int)
    y_score = rng.random(n)  # uncorrelated with y_true

    score = pr_auc(y_true, y_score)

    assert score == pytest.approx(base_rate, abs=0.02)


# --- recall_at_fpr ----------------------------------------------------------


def test_recall_at_fpr_perfect_classifier_gets_full_recall():
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.array([0.0] * 50 + [1.0] * 50)

    assert recall_at_fpr(y_true, y_score, target_fpr=0.01) == pytest.approx(1.0)
    assert recall_at_fpr(y_true, y_score, target_fpr=0.5) == pytest.approx(1.0)


def test_recall_at_fpr_inverted_classifier_gets_near_zero_recall():
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.array([1.0] * 50 + [0.0] * 50)  # positives score lowest

    assert recall_at_fpr(y_true, y_score, target_fpr=0.01) == pytest.approx(0.0)


# --- cost_per_10k -----------------------------------------------------------


def test_cost_per_10k_hand_computed():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.6, 0.4, 0.9])
    threshold = 0.5
    cost_fn = 100.0
    cost_fp = 1.0

    # predicted_positive = [False, True, False, True]
    # -> 1 false positive (idx 1), 1 false negative (idx 2)
    expected = (1 * cost_fn + 1 * cost_fp) / 4 * 10000
    assert expected == pytest.approx(252_500.0)

    assert cost_per_10k(y_true, y_score, threshold, cost_fn, cost_fp) == pytest.approx(
        expected
    )


def test_cost_per_10k_is_parameterised_not_hardcoded():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.6, 0.4, 0.9])
    threshold = 0.5

    cheap = cost_per_10k(y_true, y_score, threshold, cost_fn=1.0, cost_fp=1.0)
    expensive = cost_per_10k(y_true, y_score, threshold, cost_fn=1000.0, cost_fp=1.0)

    assert cheap != expensive


# --- evaluate_model ---------------------------------------------------------


def _tiny_booster(seed: int) -> tuple[lgb.Booster, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 200
    X = pd.DataFrame({"f1": rng.random(n), "f2": rng.random(n)})
    y = (X["f1"] > 0.5).astype(int)
    dataset = lgb.Dataset(X, label=y)
    model = lgb.train(
        {"objective": "binary", "verbosity": -1}, dataset, num_boost_round=10
    )
    return model, X, y


def test_evaluate_model_returns_expected_keys():
    model, X, y = _tiny_booster(seed=1)

    result = evaluate_model(model, X, y)

    assert set(result.keys()) == {"pr_auc", "recall_at_1pct_fpr", "cost_per_10k"}
    assert all(isinstance(v, float) for v in result.values())
    assert result["pr_auc"] > 0.9  # f1 > 0.5 is a near-trivial separator


def test_evaluate_model_accepts_threshold_and_cost_overrides():
    model, X, y = _tiny_booster(seed=2)

    default = evaluate_model(model, X, y)
    custom = evaluate_model(model, X, y, threshold=0.9, cost_fn=500.0, cost_fp=2.0)

    assert "cost_per_10k" in default and "cost_per_10k" in custom

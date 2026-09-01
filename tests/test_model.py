"""Tests for src/model.py: feature matrix construction and identical
treatment between the baseline and cluster training paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import (
    LGBM_PARAMS,
    build_feature_matrix,
    predict,
    train_baseline_model,
    train_cluster_model,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "ProductCD": ["W", "C", "W", "C"],
            "isFraud": [0, 1, 0, 1],
            "uid": ["u1", "u2", "u1", "u2"],
        }
    )


def test_build_feature_matrix_drops_label_and_uid():
    X = build_feature_matrix(_transactions())

    assert "isFraud" not in X.columns
    assert "uid" not in X.columns
    assert X.index.name == "TransactionID"
    assert X["ProductCD"].dtype.name == "category"


def test_build_feature_matrix_joins_cluster_features():
    cluster_features = pd.DataFrame(
        {"cluster_size_uids": [5, 5, 5, 5]},
        index=pd.Index([1, 2, 3, 4], name="TransactionID"),
    )

    X_baseline = build_feature_matrix(_transactions())
    X_cluster = build_feature_matrix(_transactions(), cluster_features)

    assert "cluster_size_uids" not in X_baseline.columns
    assert list(X_cluster.columns) == list(X_baseline.columns) + ["cluster_size_uids"]
    assert (X_cluster["cluster_size_uids"] == 5).all()


def test_baseline_and_cluster_training_use_identical_params_and_seed():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"f1": rng.random(n), "f2": rng.random(n)})
    y = pd.Series((X["f1"] > 0.5).astype(int))

    baseline_model = train_baseline_model(X, y)
    cluster_model = train_cluster_model(X, y)  # same X on purpose here

    # Identical training paths on identical data must produce identical
    # trees -- this is the ablation's entire premise (same seed, same
    # params, same boosting rounds).
    assert baseline_model.model_to_string() == cluster_model.model_to_string()
    assert LGBM_PARAMS["seed"] == 42


def test_predict_returns_array_of_scores():
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({"f1": rng.random(n)})
    y = pd.Series((X["f1"] > 0.5).astype(int))
    model = train_baseline_model(X, y)

    scores = predict(model, X)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (n,)
    assert ((scores >= 0) & (scores <= 1)).all()

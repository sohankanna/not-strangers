"""Train and score baseline (transaction-only) vs. cluster-augmented models."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def build_feature_matrix(
    transactions: pd.DataFrame, cluster_features: pd.DataFrame
) -> pd.DataFrame:
    """Join transaction features with entity cluster features.

    Args:
        transactions: Raw transaction rows.
        cluster_features: Per-entity cluster features, as produced by
            graph.compute_cluster_features.

    Returns:
        A feature matrix indexed by TransactionID, ready for training or
        scoring.
    """
    raise NotImplementedError


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    """Train a model using transaction features only (no cluster features).

    Args:
        X_train: Training feature matrix, transaction features only.
        y_train: Training labels (chargeback-reported, noisy).

    Returns:
        A trained LightGBM booster.
    """
    raise NotImplementedError


def train_cluster_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    """Train a model using transaction features plus cluster features.

    Args:
        X_train: Training feature matrix, including cluster features from
            build_feature_matrix.
        y_train: Training labels (chargeback-reported, noisy).

    Returns:
        A trained LightGBM booster.
    """
    raise NotImplementedError


def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Score a feature matrix with a trained model.

    Args:
        model: A trained LightGBM booster.
        X: Feature matrix to score.

    Returns:
        An array of predicted abuse scores, one per row of X.
    """
    raise NotImplementedError

"""Train and score baseline (transaction-only) vs. cluster-augmented models.

train_baseline_model and train_cluster_model both call the same private
_fit() helper with the same LGBM_PARAMS dict, the same NUM_BOOST_ROUND and
the same seed -- structurally, not just by value, so the only way the two
models can differ is in the columns of X. That is the entire point of the
ablation: if the two training functions diverged in any other way, the
comparison would measure that divergence, not the cluster features.

build_feature_matrix never includes the raw uid string (or an internal
cluster_id) as a feature, in either model. Only the engineered cluster
statistics (cluster size, density, velocity, ...) are added for the cluster
model -- not the identity string itself. Including the raw uid would let a
model memorize "this exact uid was fraud in training" rather than learn
from the aggregated signal, which would inflate the ablation for reasons
that have nothing to do with the cluster features being tested.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 42
NUM_BOOST_ROUND = 300

LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "None",
    "verbosity": -1,
    "seed": SEED,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
}

_NON_FEATURE_COLUMNS = ("isFraud", "uid")


def build_feature_matrix(
    transactions: pd.DataFrame, cluster_features: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join transaction features with entity cluster features.

    Args:
        transactions: Raw transaction rows (TransactionID as a column or as
            the index). The label column (isFraud) and a raw "uid" column,
            if present, are dropped -- neither belongs in the feature set.
        cluster_features: Per-entity cluster features, as produced by
            graph.compute_cluster_features, already broadcast from per-uid
            to per-TransactionID (i.e. indexed the same way as
            `transactions`) by the caller. None for the baseline model.

    Returns:
        A feature matrix indexed by TransactionID, with every remaining
        object-dtype column cast to category (LightGBM's native categorical
        handling), ready for lgb.Dataset.
    """
    X = transactions.copy()
    if "TransactionID" in X.columns:
        X = X.set_index("TransactionID")
    X = X.drop(columns=[c for c in _NON_FEATURE_COLUMNS if c in X.columns])

    for col in X.select_dtypes(include=["object", "str"]).columns:
        X[col] = X[col].astype("category")

    if cluster_features is not None:
        X = X.join(cluster_features, how="left")

    return X


def _fit(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    categorical = X_train.select_dtypes(include="category").columns.tolist()
    dataset = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical or "auto"
    )
    return lgb.train(LGBM_PARAMS, dataset, num_boost_round=NUM_BOOST_ROUND)


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    """Train a model using transaction features only (no cluster features).

    Args:
        X_train: Training feature matrix, transaction features only.
        y_train: Training labels (chargeback-reported, noisy).

    Returns:
        A trained LightGBM booster.
    """
    return _fit(X_train, y_train)


def train_cluster_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    """Train a model using transaction features plus cluster features.

    Args:
        X_train: Training feature matrix, including cluster features from
            build_feature_matrix.
        y_train: Training labels (chargeback-reported, noisy).

    Returns:
        A trained LightGBM booster.
    """
    return _fit(X_train, y_train)


def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Score a feature matrix with a trained model.

    Args:
        model: A trained LightGBM booster.
        X: Feature matrix to score.

    Returns:
        An array of predicted abuse scores, one per row of X.
    """
    return model.predict(X)

"""Task 2: score attribution -- SHAP contribution, threshold position, and
the transaction-vs-cluster contribution split.

Every value here comes from a real transaction row scored by the real
trained cluster model (src/model.py, frozen, unmodified) and a real SHAP
explainer built against it -- nothing is synthesized. Thresholds are read
directly from src/policy.py's own constants (STEP_UP_THRESHOLD,
REVIEW_THRESHOLD); this module never hardcodes their values.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import policy

from dashboard_theme import (
    ACCENT,
    BORDER,
    RISK_ALLOW,
    RISK_REVIEW,
    RISK_STEPUP,
    SURFACE,
    TEXT,
    TEXT_MUTED,
)


@st.cache_resource(show_spinner="Building the SHAP explainer against the cluster model (once)...")
def get_shap_explainer(_cluster_model):
    """Built once per process against the real trained booster (src/model.py's
    train_cluster_model output) and cached -- never reconstructed per
    cluster selection, per the task's own instruction.
    """
    import shap

    return shap.TreeExplainer(_cluster_model)


def compute_shap_row(explainer, x_row: pd.DataFrame) -> tuple[pd.Series, float]:
    """SHAP values for exactly one transaction's feature row.

    Returns (shap_values, expected_value) in the tree explainer's native
    log-odds/margin space -- verified against this project's own trained
    model: sum(shap_values) + expected_value equals the model's raw-margin
    prediction for this row.
    """
    raw = explainer.shap_values(x_row)
    values = raw[1] if isinstance(raw, list) else raw
    values = np.asarray(values).reshape(-1)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    return pd.Series(values, index=x_row.columns), float(expected_value)


def build_shap_bar_figure(shap_row: pd.Series, top_n: int = 12):
    """Horizontal bar chart, top `top_n` features by |SHAP value|. Positive
    (pushes toward fraud) in the risk-review color, negative (pulls away)
    in the risk-allow color -- the same two colors this dashboard already
    uses for those two ends of its risk palette, not a new saturated pair.
    """
    top_idx = shap_row.abs().nlargest(top_n).index
    ordered = shap_row.loc[top_idx].sort_values(key=lambda s: s.abs())

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    colors = [RISK_REVIEW if v > 0 else RISK_ALLOW for v in ordered.values]
    ax.barh(ordered.index, ordered.values, color=colors, height=0.65)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_xlabel("SHAP value (log-odds / margin space)", color=TEXT_MUTED, fontsize=8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="y", colors=TEXT, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    fig.tight_layout()
    return fig


def build_threshold_figure(score: float, action: str):
    """A 0-1 score axis with STEP_UP_THRESHOLD and REVIEW_THRESHOLD marked
    (read live from src/policy.py, never hardcoded here) and the actual
    score plotted against them -- the mechanical "this scored X, review
    begins at Y, therefore Z" the task asks to make visible.
    """
    step_up_t = policy.STEP_UP_THRESHOLD
    review_t = policy.REVIEW_THRESHOLD
    action_colors = {"review": RISK_REVIEW, "step_up": RISK_STEPUP, "allow": RISK_ALLOW}

    fig, ax = plt.subplots(figsize=(7.5, 1.7))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.axhline(0, color=BORDER, linewidth=1)

    ax.axvline(step_up_t, color=RISK_STEPUP, linestyle="--", linewidth=1.2)
    ax.axvline(review_t, color=RISK_REVIEW, linestyle="--", linewidth=1.2)
    ax.annotate(
        f"step_up\n{step_up_t:.4f}", (step_up_t, 0.22), color=RISK_STEPUP,
        fontsize=7.5, ha="center", va="bottom",
    )
    ax.annotate(
        f"review\n{review_t:.4f}", (review_t, 0.22), color=RISK_REVIEW,
        fontsize=7.5, ha="center", va="bottom",
    )

    score_clamped = min(max(score, 0.0), 1.0)
    ax.scatter(
        [score_clamped], [0], s=170, color=action_colors.get(action, ACCENT),
        zorder=5, edgecolor=TEXT, linewidth=1.2,
    )
    ax.annotate(
        f"score {score:.4f} -> {action}", (score_clamped, -0.3),
        color=TEXT, fontsize=9, fontweight="bold", ha="center", va="top",
    )

    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    fig.tight_layout()
    return fig


def txn_vs_cluster_split(shap_row: pd.Series, cluster_feature_columns: set) -> dict:
    """How much of this transaction's score came from cluster-level
    features vs. transaction-level ones, summed from the real SHAP row --
    the project's central claim (clusters carry signal beyond the raw
    transaction) made visible per decision, not just in aggregate ablation
    numbers.
    """
    cluster_cols = [c for c in shap_row.index if c in cluster_feature_columns]
    txn_cols = [c for c in shap_row.index if c not in cluster_feature_columns]

    cluster_sum = float(shap_row[cluster_cols].sum())
    txn_sum = float(shap_row[txn_cols].sum())
    cluster_abs = float(shap_row[cluster_cols].abs().sum())
    txn_abs = float(shap_row[txn_cols].abs().sum())
    total_abs = cluster_abs + txn_abs

    return {
        "cluster_sum": cluster_sum,
        "txn_sum": txn_sum,
        "total": cluster_sum + txn_sum,
        "cluster_abs_pct": (cluster_abs / total_abs * 100) if total_abs else float("nan"),
        "txn_abs_pct": (txn_abs / total_abs * 100) if total_abs else float("nan"),
        "n_cluster_features": len(cluster_cols),
        "n_txn_features": len(txn_cols),
    }

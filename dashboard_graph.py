"""Task 1 (network graph) and Task 3 (contrast view) rendering for app.py.

Every node, edge, and value here comes straight from the real pipeline
objects passed in -- pipeline_data.entity_graph.graph (the actual entity
graph build_entity_graph produced) and the real transactions belonging to
a cluster's members. This module only lays out and colors what's already
there; it never invents a node, an edge, or a transaction.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_theme import (
    BORDER,
    DEFAULT_EDGE_COLOR,
    MULTI_RULE_COLOR,
    RISK_ALLOW,
    RISK_REVIEW,
    RULE_COLORS,
    SURFACE,
    TEXT,
    TEXT_MUTED,
)

MAX_NODES_BEFORE_SAMPLING = 60
_RULE_LABELS = {
    "device_info": "device_info",
    "addr1_email": "addr1 + email",
    "card_bank_addr": "card + bank + addr",
}


def _edge_color_and_label(rules: set) -> tuple[str, str]:
    if len(rules) > 1:
        return MULTI_RULE_COLOR, "multiple rules"
    if len(rules) == 1:
        rule = next(iter(rules))
        return RULE_COLORS.get(rule, DEFAULT_EDGE_COLOR), _RULE_LABELS.get(rule, rule)
    return DEFAULT_EDGE_COLOR, "unlabeled edge"


@st.cache_data(show_spinner=False)
def _layout_for_members(_graph: nx.Graph, member_key: tuple[str, ...]) -> dict:
    """Deterministic node positions for one cluster's subgraph, cached by
    the exact (sorted) member tuple -- re-selecting the same cluster reuses
    this instead of recomputing spring_layout.

    nx.spring_layout degenerates for 1-2 nodes (everything collapses at or
    near the origin), so those sizes are placed explicitly instead.
    """
    sub = _graph.subgraph(member_key)
    n = sub.number_of_nodes()
    if n <= 2:
        xs = np.linspace(-1, 1, n) if n > 1 else np.array([0.0])
        return {node: (float(x), 0.0) for node, x in zip(sub.nodes(), xs)}
    return {node: (float(x), float(y)) for node, (x, y) in nx.spring_layout(sub, seed=42).items()}


def _per_uid_stats(transactions_sub: pd.DataFrame) -> pd.DataFrame:
    """txn count, amount range, and fraud-label status per uid -- read
    straight from the real transaction rows belonging to this cluster.
    """
    return transactions_sub.groupby("uid").agg(
        txn_count=("TransactionAmt", "size"),
        amt_min=("TransactionAmt", "min"),
        amt_max=("TransactionAmt", "max"),
        any_fraud=("isFraud", "max"),
    )


def build_cluster_network_figure(
    graph: nx.Graph,
    members: list[str],
    transactions_sub: pd.DataFrame,
    height: int = 480,
) -> tuple[go.Figure, dict]:
    """The real entity-graph subgraph for one cluster's members.

    Returns (figure, meta). meta["sampled_note"] is set (and non-None) when
    the cluster was too large to render whole and got sampled down to its
    MAX_NODES_BEFORE_SAMPLING highest-transaction-count members -- the
    caller should surface that note visibly, per the task's requirement
    that sampling never happen silently.
    """
    per_uid = _per_uid_stats(transactions_sub)

    member_set = list(members)
    sampled_note = None
    if len(member_set) > MAX_NODES_BEFORE_SAMPLING:
        by_txns = per_uid.reindex(member_set)["txn_count"].fillna(0).sort_values(ascending=False)
        member_set = by_txns.head(MAX_NODES_BEFORE_SAMPLING).index.tolist()
        sampled_note = (
            f"This cluster has {len(members)} members -- showing the "
            f"{MAX_NODES_BEFORE_SAMPLING} with the most transactions, plus "
            "the real edges among just that subset. Not the full cluster."
        )

    sub = graph.subgraph(member_set)
    pos = _layout_for_members(graph, tuple(sorted(member_set)))

    fig = go.Figure()

    edges_by_color: dict[str, tuple[str, list]] = {}
    for u, v, data in sub.edges(data=True):
        rules = data.get("rules", set())
        color, label = _edge_color_and_label(rules)
        edges_by_color.setdefault(color, (label, []))[1].append((u, v))

    for color, (label, edges) in edges_by_color.items():
        edge_x, edge_y = [], []
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color=color, width=1.6),
                hoverinfo="none",
                name=label,
                showlegend=True,
            )
        )

    nodes = list(sub.nodes())
    txn_counts = [int(per_uid.loc[u, "txn_count"]) if u in per_uid.index else 0 for u in nodes]
    fraud_flags = [bool(per_uid.loc[u, "any_fraud"]) if u in per_uid.index else False for u in nodes]
    amt_mins = [float(per_uid.loc[u, "amt_min"]) if u in per_uid.index else float("nan") for u in nodes]
    amt_maxs = [float(per_uid.loc[u, "amt_max"]) if u in per_uid.index else float("nan") for u in nodes]

    node_sizes = [10 + 6 * np.sqrt(t) for t in txn_counts]
    node_colors = [RISK_REVIEW if f else RISK_ALLOW for f in fraud_flags]
    hover_text = [
        f"uid {u}<br>transactions: {t}<br>amount range: ${amin:,.2f}-${amax:,.2f}"
        f"<br>{'carried a fraud-labelled txn' if f else 'no fraud-labelled txn'}"
        for u, t, amin, amax, f in zip(nodes, txn_counts, amt_mins, amt_maxs, fraud_flags)
    ]

    fig.add_trace(
        go.Scatter(
            x=[pos[u][0] for u in nodes],
            y=[pos[u][1] for u in nodes],
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color=BORDER),
            ),
            text=hover_text,
            hoverinfo="text",
            customdata=nodes,
            name="uids",
            showlegend=False,
        )
    )

    # Dummy traces purely so the fraud/clean node coloring gets a legend
    # entry -- plotly only legends traces, not individual marker colors.
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=RISK_REVIEW),
            name="uid carried a fraud-labelled txn", showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=RISK_ALLOW),
            name="no fraud-labelled txn", showlegend=True,
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=TEXT_MUTED, size=11),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        hovermode="closest",
    )

    meta = {
        "sampled_note": sampled_note,
        "per_uid": per_uid,
        "n_nodes_shown": len(nodes),
        "n_edges_shown": sub.number_of_edges(),
        "n_members_total": len(members),
    }
    return fig, meta


def find_contrast_cluster(queue: pd.DataFrame, selected_uid_count: int, exclude_cluster_id: int) -> pd.Series | None:
    """The 'allow' (non-flagged) cluster in the current queue closest in
    size to the selected one -- Task 3's contrast view. Only searches the
    queue that's already been computed for real (build_cluster_queue's
    output), never a synthesized comparison cluster.
    """
    candidates = queue[(queue["action"] == "allow") & (queue["cluster_id"] != exclude_cluster_id)]
    if candidates.empty:
        return None
    idx = (candidates["uid_count"] - selected_uid_count).abs().idxmin()
    return candidates.loc[idx]

"""Live replay tab: replays real held-out test-split transactions in their
actual TransactionDT order, showing the pipeline operating as it would in
production. Every transaction, uid, edge, score, and decision here is read
from the existing pipeline outputs (pipeline_data, trained) -- nothing is
generated, simulated, or synthesised. Additive only: this module and the
new tab in app.py that calls it do not touch any existing tab or helper.

Precompute discipline: build_replay_sequence runs ONCE (st.cache_data) and
does the entire pass over the replay window -- scoring (a single vectorized
model.predict call, already-trained model), policy decisions (policy.py,
vectorized), and the incremental graph-reveal bookkeeping. Nothing in this
module calls model.predict, rebuilds the entity graph, or recomputes
cluster features per frame; per-frame rendering (build_incremental_figure)
only ever indexes into the precomputed sequence and the one fixed, upfront
node layout.

Window choice, verified against the real pipeline before writing any UI
code (see the session's diagnostic, not asserted from guesswork): the
first WINDOW_SIZE test-period transactions (chronological order) contain 2
real REVIEW_THRESHOLD crossings (of 10 that occur across the full
118,108-row test split) -- chosen because it's a small, fast, honest slice
that already satisfies "at least one crossing" without reaching for a
larger, slower window.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import investigator, policy
from src.graph import get_connected_components

from dashboard_graph import _edge_color_and_label
from dashboard_theme import (
    BORDER,
    MODEL_COLOR,
    RISK_ALLOW,
    RISK_REVIEW,
    RISK_STEPUP,
    SURFACE,
    TEXT_MUTED,
)

WINDOW_SIZE = 2000


@st.cache_data(show_spinner="Precomputing the replay sequence (once) -- scoring, policy decisions, and a one-time graph layout...")
def build_replay_sequence(_pipeline_data, _trained) -> dict:
    """The entire replay, precomputed once. Returns a plain dict; the UI
    only ever indexes into it (see build_incremental_figure and app.py's
    render_replay_tab) -- no feature recomputation, no graph rebuild, no
    model call happens per displayed frame.
    """
    graph = _pipeline_data.entity_graph.graph
    components = get_connected_components(graph)
    cluster_of: dict[str, int] = {}
    for i, comp in enumerate(components):
        if len(comp) >= 2:
            for u in comp:
                cluster_of[u] = i

    test_df = _pipeline_data.test_df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    window_df = test_df.head(WINDOW_SIZE).copy()
    test_ids = window_df["TransactionID"]

    uids = _pipeline_data.entity_ids.reindex(test_ids).to_numpy()
    scores = _trained.cluster_model.predict(_trained.X_test_cluster.reindex(test_ids))
    decisions = policy.apply_policy(
        pd.DataFrame({"score": np.asarray(scores)}, index=test_ids.to_numpy())
    ).reset_index(drop=True)

    # Distinct uids that are BOTH transacting in this window AND already a
    # node in the causal entity graph (i.e. had train-period history) --
    # only these get drawn in the incremental graph. A uid absent from the
    # graph has no train-period history and therefore no position in it;
    # that's a real, documented property of this pipeline (see
    # ablation.md's sanity check #4), not something to fake a position for.
    distinct_graph_uids = sorted({
        u for u in pd.unique(uids)
        if u is not None and not (isinstance(u, float) and np.isnan(u)) and u in graph
    })
    node_positions = _compute_layout(graph, distinct_graph_uids)

    frames: list[dict] = []
    seen_uids: set[str] = set()
    seen_graph_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    cluster_running_max: dict[int, float] = {}
    cluster_revealed_members: dict[int, set[str]] = {}
    step_up_fired: set[int] = set()
    review_fired: set[int] = set()
    review_events: list[dict] = []

    for i in range(len(window_df)):
        row = window_df.iloc[i]
        uid = uids[i]
        has_uid = uid is not None and not (isinstance(uid, float) and np.isnan(uid))
        score = float(scores[i])
        action = str(decisions.iloc[i]["action"])
        threshold_applied = float(decisions.iloc[i]["threshold_applied"])

        new_node = False
        new_edges: list[tuple[str, str, str, str]] = []
        cid = None
        crossed_step_up_now = False
        crossed_review_now = False

        if has_uid:
            seen_uids.add(uid)
            if uid in graph:
                if uid not in seen_graph_nodes:
                    new_node = True
                    seen_graph_nodes.add(uid)
                for nbr in graph.neighbors(uid):
                    if nbr in seen_graph_nodes:
                        edge_key = tuple(sorted((uid, nbr)))
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            rules = graph.get_edge_data(uid, nbr).get("rules", set())
                            color, label = _edge_color_and_label(rules)
                            new_edges.append((uid, nbr, color, label))

            cid = cluster_of.get(uid)
            if cid is not None:
                cluster_revealed_members.setdefault(cid, set()).add(uid)
                prev_max = cluster_running_max.get(cid, 0.0)
                if score > prev_max:
                    cluster_running_max[cid] = score
                    if prev_max < policy.STEP_UP_THRESHOLD <= score and cid not in step_up_fired:
                        step_up_fired.add(cid)
                        crossed_step_up_now = True
                    if prev_max < policy.REVIEW_THRESHOLD <= score and cid not in review_fired:
                        review_fired.add(cid)
                        crossed_review_now = True

        frames.append(
            {
                "step": i,
                "transaction_id": int(row["TransactionID"]),
                "transaction_dt": float(row["TransactionDT"]),
                "uid": uid if has_uid else None,
                "amount": float(row["TransactionAmt"]),
                "is_fraud": bool(row["isFraud"]),
                "score": score,
                "action": action,
                "threshold_applied": threshold_applied,
                "new_node": new_node,
                "new_edges": new_edges,
                "cluster_id": int(cid) if cid is not None else None,
                "crossed_step_up_now": crossed_step_up_now,
                "crossed_review_now": crossed_review_now,
                "cum_txns": i + 1,
                "cum_uids_seen": len(seen_uids),
                "cum_clusters_formed": sum(1 for m in cluster_revealed_members.values() if len(m) >= 2),
                "cum_step_up_fired": len(step_up_fired),
                "cum_review_fired": len(review_fired),
                "review_fired_clusters_so_far": frozenset(review_fired),
            }
        )

        if crossed_review_now:
            review_events.append({"step": i, "cluster_id": int(cid)})

    # Precompute the LLM narrative for each review-firing cluster ONCE here
    # -- reusing investigator.explain_cluster exactly as the existing
    # Review queue tab does (same function, same graceful LLM-or-fallback
    # behavior) -- never called per frame during playback.
    df_full = _pipeline_data.df.set_index("TransactionID")
    df_full = pd.concat([df_full, _pipeline_data.entity_ids.rename("uid")], axis=1)
    narratives: dict[int, investigator.ClusterExplanation] = {}
    for ev in review_events:
        cid = ev["cluster_id"]
        if cid in narratives:
            continue
        members = sorted(components[cid])
        cf_sub = _pipeline_data.cluster_features.loc[
            [u for u in members if u in _pipeline_data.cluster_features.index]
        ]
        transactions_sub = df_full[df_full["uid"].isin(members)]
        narratives[cid] = investigator.explain_cluster(
            cluster_id=f"cluster-{cid}", cluster_features=cf_sub, transactions=transactions_sub,
        )

    return {
        "frames": frames,
        "node_positions": node_positions,
        "components": components,
        "narratives": narratives,
        "review_events": review_events,
        "window_size": len(window_df),
        "total_test_size": len(test_df),
        "n_review_crossings_in_window": len(review_events),
    }


def _compute_layout(graph: nx.Graph, distinct_graph_uids: list[str]) -> dict[str, tuple[float, float]]:
    """One fixed spring_layout over every uid that will ever be revealed in
    the window (restricted to real graph edges among just this node set) --
    computed once, upfront, so revealing nodes/edges over time never moves
    an already-placed node. Isolated nodes (the common case here -- most
    transacting uids in this window belong to no multi-member cluster)
    naturally scatter via the same force-directed layout, which is an
    honest picture: sparse real structure inside a much larger population
    of unconnected activity, not a fabricated arrangement.
    """
    if not distinct_graph_uids:
        return {}
    sub = nx.Graph()
    sub.add_nodes_from(distinct_graph_uids)
    node_set = set(distinct_graph_uids)
    for u in distinct_graph_uids:
        for v in graph.neighbors(u):
            if v in node_set:
                sub.add_edge(u, v)
    pos = nx.spring_layout(sub, seed=42)
    return {u: (float(x), float(y)) for u, (x, y) in pos.items()}


def build_incremental_figure(sequence: dict, step: int, height: int = 560) -> go.Figure:
    """The entity graph as revealed through frame `step` -- reads only the
    precomputed sequence and the fixed node_positions; no layout, no graph
    query, no model call happens here.
    """
    frames = sequence["frames"]
    node_positions = sequence["node_positions"]

    revealed_nodes: dict[str, dict] = {}
    revealed_edges: dict[tuple[str, str], tuple[str, str]] = {}
    review_fired_clusters: frozenset = frozenset()

    for f in frames[: step + 1]:
        uid = f["uid"]
        if uid is not None and uid in node_positions:
            entry = revealed_nodes.setdefault(uid, {"txn_count": 0, "any_fraud": False, "cluster_id": f["cluster_id"]})
            entry["txn_count"] += 1
            entry["any_fraud"] = entry["any_fraud"] or f["is_fraud"]
        for (u, v, color, label) in f["new_edges"]:
            revealed_edges[(u, v)] = (color, label)
        review_fired_clusters = f["review_fired_clusters_so_far"]

    fig = go.Figure()

    edges_by_color: dict[str, tuple[str, list]] = {}
    for (u, v), (color, label) in revealed_edges.items():
        edges_by_color.setdefault(color, (label, []))[1].append((u, v))
    for color, (label, edges) in edges_by_color.items():
        ex, ey = [], []
        for u, v in edges:
            x0, y0 = node_positions[u]
            x1, y1 = node_positions[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=1.6), hoverinfo="none", name=label, showlegend=True))

    if revealed_nodes:
        nodes = list(revealed_nodes.keys())
        xs = [node_positions[u][0] for u in nodes]
        ys = [node_positions[u][1] for u in nodes]
        highlighted = [revealed_nodes[u]["cluster_id"] in review_fired_clusters for u in nodes]
        colors = [RISK_REVIEW if revealed_nodes[u]["any_fraud"] else RISK_ALLOW for u in nodes]
        sizes = [16 if h else 11 for h in highlighted]
        line_colors = [RISK_REVIEW if h else BORDER for h in highlighted]
        line_widths = [3 if h else 1 for h in highlighted]
        hover = [
            f"uid {u}<br>revealed transactions: {revealed_nodes[u]['txn_count']}"
            + (f"<br>cluster {revealed_nodes[u]['cluster_id']} -- REVIEW fired" if highlighted[i] else "")
            for i, u in enumerate(nodes)
        ]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=sizes, color=colors, line=dict(width=line_widths, color=line_colors)),
                text=hover, hoverinfo="text", name="uids", showlegend=False,
            )
        )

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=RISK_REVIEW), name="carried a fraud-labelled txn", showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=RISK_ALLOW), name="no fraud-labelled txn", showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=13, color=RISK_ALLOW, line=dict(width=3, color=RISK_REVIEW)), name="cluster fired REVIEW", showlegend=True))

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=TEXT_MUTED, size=11),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-1.3, 1.3]),
    )
    return fig


def feed_dataframe(sequence: dict, step: int, max_rows: int = 20) -> pd.DataFrame:
    """The most recent `max_rows` processed transactions through `step`, for
    the scrolling feed -- every field read straight from the precomputed
    frame, nothing recomputed.
    """
    frames = sequence["frames"][max(0, step - max_rows + 1) : step + 1]
    rows = [
        {
            "step": f["step"],
            "TransactionDT": f["transaction_dt"],
            "uid": f["uid"] if f["uid"] is not None else "(no uid)",
            "amount": f["amount"],
            "score": round(f["score"], 4),
            "action": f["action"],
        }
        for f in reversed(frames)
    ]
    return pd.DataFrame(rows)


def counters_at(sequence: dict, step: int) -> dict:
    f = sequence["frames"][step]
    return {
        "cum_txns": f["cum_txns"],
        "cum_uids_seen": f["cum_uids_seen"],
        "cum_clusters_formed": f["cum_clusters_formed"],
        "cum_step_up_fired": f["cum_step_up_fired"],
        "cum_review_fired": f["cum_review_fired"],
    }


def review_events_through(sequence: dict, step: int) -> list[dict]:
    return [ev for ev in sequence["review_events"] if ev["step"] <= step]

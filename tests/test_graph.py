"""Tests for src/graph.py: linkage rules, hub guard, and causal cluster
features -- including the explicit leakage test the task called for.
"""

from __future__ import annotations

import itertools

import networkx as nx
import pandas as pd
import pytest

from src.graph import (
    build_entity_graph,
    compute_cluster_features,
    get_connected_components,
)


def _entity_ids(transaction_ids, uids) -> pd.Series:
    return pd.Series(
        uids,
        index=pd.Index(transaction_ids, name="TransactionID"),
        name="uid",
    )


# --- build_entity_graph: linkage rules -------------------------------------


def test_device_info_links_shared_device():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 200, 300],
            "DeviceInfo": ["ZTE-Z956", "ZTE-Z956", "iPhone"],
        }
    )
    entity_ids = _entity_ids([1, 2, 3], ["uidA", "uidB", "uidC"])

    result = build_entity_graph(df, entity_ids)

    assert result.graph.has_edge("uidA", "uidB")
    assert not result.graph.has_edge("uidA", "uidC")
    assert result.graph["uidA"]["uidB"]["rules"] == {"device_info"}


def test_addr1_email_requires_both_to_match():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 200, 300],
            "addr1": [325, 325, 325],
            "P_emaildomain": ["yahoo.com", "yahoo.com", "hotmail.com"],
        }
    )
    entity_ids = _entity_ids([1, 2, 3], ["uidA", "uidB", "uidC"])

    result = build_entity_graph(df, entity_ids)

    assert result.graph.has_edge("uidA", "uidB")
    # Same addr1 but a different email domain -- addr1 alone isn't enough.
    assert not result.graph.has_edge("uidA", "uidC")


def test_card_bank_addr_requires_all_three():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 200, 300],
            "card3": [150, 150, 150],
            "card5": [226, 226, 226],
            "addr1": [325, 325, 200],
        }
    )
    entity_ids = _entity_ids([1, 2, 3], ["uidA", "uidB", "uidC"])

    result = build_entity_graph(df, entity_ids)

    assert result.graph.has_edge("uidA", "uidB")
    assert not result.graph.has_edge("uidA", "uidC")


def test_missing_linkage_columns_are_skipped_not_errors():
    df = pd.DataFrame({"TransactionID": [1, 2], "TransactionDT": [1, 2]})
    entity_ids = _entity_ids([1, 2], ["uidA", "uidB"])

    result = build_entity_graph(df, entity_ids)

    assert result.graph.number_of_edges() == 0
    assert list(result.graph.nodes) == ["uidA", "uidB"]


# --- hub guard ---------------------------------------------------------------


def test_hub_guard_excludes_common_values_and_reports_them():
    n = 10
    df = pd.DataFrame(
        {
            "TransactionID": list(range(n)),
            "TransactionDT": list(range(n)),
            "DeviceInfo": ["Windows"] * n,
        }
    )
    entity_ids = _entity_ids(list(range(n)), [f"uid{i}" for i in range(n)])

    result = build_entity_graph(df, entity_ids, max_degree=5)

    assert result.graph.number_of_edges() == 0
    assert len(result.excluded_hubs) == 1
    row = result.excluded_hubs.iloc[0]
    assert row["rule"] == "device_info"
    assert row["value"] == "Windows"
    assert row["uid_count"] == n


def test_hub_guard_does_not_exclude_values_at_or_below_threshold():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [1, 2, 3],
            "DeviceInfo": ["rare-device"] * 3,
        }
    )
    entity_ids = _entity_ids([1, 2, 3], ["uidA", "uidB", "uidC"])

    result = build_entity_graph(df, entity_ids, max_degree=3)

    assert result.excluded_hubs.empty
    assert result.graph.number_of_edges() == 3  # a full triangle


# --- get_connected_components -------------------------------------------------


def test_connected_components_includes_singletons():
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("a", "b")

    components = get_connected_components(graph)

    assert {"a", "b"} in components
    assert {"c"} in components
    assert len(components) == 2


# --- compute_cluster_features: hand-computed values --------------------------


def _feature_frame():
    # Cluster 1: uidA (2 txns) + uidB (1 txn), linked.
    # Cluster 2: uidC, a singleton (no edges).
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [0, 1000, 50, 10],
            "TransactionAmt": [100.0, 200.0, 150.0, 300.0],
            "P_emaildomain": ["a.com", "a.com", "b.com", "c.com"],
            "isFraud": [0, 0, 1, 0],
        }
    )
    entity_ids = _entity_ids([1, 2, 3, 4], ["uidA", "uidA", "uidB", "uidC"])
    graph = nx.Graph()
    graph.add_nodes_from(["uidA", "uidB", "uidC"])
    graph.add_edge("uidA", "uidB")
    return df, entity_ids, graph


def test_compute_cluster_features_hand_computed():
    df, entity_ids, graph = _feature_frame()

    result = compute_cluster_features(df, entity_ids, graph)

    a = result.loc["uidA"]
    b = result.loc["uidB"]
    c = result.loc["uidC"]

    # Cluster 1 (uidA, uidB): 3 transactions, 2 uids, 1 edge.
    assert a["cluster_size_uids"] == 2
    assert a["cluster_txn_count"] == 3
    assert a["cluster_edge_density"] == pytest.approx(1.0)  # 1 edge / 1 possible
    assert a["node_degree"] == 1
    assert a["cluster_velocity"] == pytest.approx(3.0)  # span < 1 day, clipped to 1
    # amounts [100, 200, 150]: mean 150, sample std 50 -> cv = 1/3
    assert a["cluster_amt_cv"] == pytest.approx(50.0 / 150.0)
    # bins (//180s): txn1->0, txn3->0, txn2->5 => 2-of-3 share the busiest bin
    assert a["cluster_burst_concentration"] == pytest.approx(2 / 3)
    # cluster emails {a.com, b.com} = 2, cluster uids = 2 -> ratio 1.0
    assert a["cluster_email_uid_ratio"] == pytest.approx(1.0)
    # uidB is fraud, uidA is not -> 1 of 2 members => 0.5
    assert a["cluster_prior_fraud_share"] == pytest.approx(0.5)
    assert b["cluster_prior_fraud_share"] == pytest.approx(0.5)

    # Per-uid features differ between uidA and uidB.
    assert a["uid_email_domain_count"] == 1  # uidA: only a.com
    assert b["uid_email_domain_count"] == 1  # uidB: only b.com
    assert b["node_degree"] == 1

    # Cluster 2 (uidC): a singleton -- no possible edges, no variance.
    assert c["cluster_size_uids"] == 1
    assert c["cluster_txn_count"] == 1
    assert pd.isna(c["cluster_edge_density"])  # 0 possible edges
    assert pd.isna(c["cluster_amt_cv"])  # a single observation has no spread
    assert c["node_degree"] == 0
    assert c["cluster_prior_fraud_share"] == pytest.approx(0.0)


def test_compute_cluster_features_no_rows_returns_empty_frame():
    df, entity_ids, graph = _feature_frame()

    result = compute_cluster_features(df, entity_ids, graph, as_of=0)  # nothing qualifies

    assert result.empty


# --- topology features: k_core_number, star_ratio (include_topology=True) ---


def _topology_frame():
    """Three clusters chosen to need BOTH new features to tell apart:
    a star (hub uidH + 3 leaves, a tree -- no cycle) and a same-size clique
    (uidP/Q/R/S, K4) end up with the identical star_ratio (max degree /
    cluster size), by construction -- that's the whole reason star_ratio
    alone doesn't distinguish shape and has to be read alongside
    cluster_edge_density (already existing) and k_core_number (new): a tree
    can never have a k-core above 1 (no cycles to sustain one), while K4's
    every node is in its own 3-core. uidZ is an isolated singleton, the
    existing degenerate case every other feature already handles.
    """
    uids = ["uidH", "uidL1", "uidL2", "uidL3", "uidP", "uidQ", "uidR", "uidS", "uidZ"]
    df = pd.DataFrame(
        {
            "TransactionID": [10, 11, 12, 13, 20, 21, 22, 23, 30],
            "TransactionDT": [100, 101, 102, 103, 200, 201, 202, 203, 300],
            "TransactionAmt": [100.0] * 9,
            "P_emaildomain": ["x.com"] * 9,
            "isFraud": [0] * 9,
        }
    )
    entity_ids = _entity_ids([10, 11, 12, 13, 20, 21, 22, 23, 30], uids)

    graph = nx.Graph()
    graph.add_nodes_from(uids)
    for leaf in ("uidL1", "uidL2", "uidL3"):
        graph.add_edge("uidH", leaf)
    for u1, u2 in itertools.combinations(("uidP", "uidQ", "uidR", "uidS"), 2):
        graph.add_edge(u1, u2)
    return df, entity_ids, graph


def test_topology_features_absent_unless_requested():
    """include_topology defaults to False -- every existing caller
    (run_pipeline.py included, unmodified) keeps getting exactly the
    original 10 columns, computed exactly as before.
    """
    df, entity_ids, graph = _topology_frame()

    result = compute_cluster_features(df, entity_ids, graph)

    assert "k_core_number" not in result.columns
    assert "star_ratio" not in result.columns
    assert list(result.columns) == [
        "cluster_size_uids",
        "cluster_txn_count",
        "cluster_edge_density",
        "node_degree",
        "cluster_velocity",
        "cluster_amt_cv",
        "cluster_burst_concentration",
        "uid_email_domain_count",
        "cluster_email_uid_ratio",
        "cluster_prior_fraud_share",
    ]


def test_topology_features_star_vs_clique_vs_singleton():
    df, entity_ids, graph = _topology_frame()

    result = compute_cluster_features(df, entity_ids, graph, include_topology=True)

    hub = result.loc["uidH"]
    leaf = result.loc["uidL1"]
    clique_member = result.loc["uidP"]
    lone = result.loc["uidZ"]

    # Star (hub degree 3, cluster size 4): star_ratio = 3/4. A tree has no
    # cycle, so k-core tops out at 1 for every connected node in it.
    assert hub["star_ratio"] == pytest.approx(0.75)
    assert leaf["star_ratio"] == pytest.approx(0.75)  # per-cluster, not per-uid
    assert hub["k_core_number"] == 1
    assert leaf["k_core_number"] == 1
    assert hub["cluster_edge_density"] == pytest.approx(3 / 6)  # 3 edges / C(4,2)

    # Clique (K4): every node degree 3, same cluster size 4 -> the SAME
    # star_ratio as the star above -- demonstrating why star_ratio alone
    # can't tell the two shapes apart. edge_density and k_core_number do:
    # K4's minimum degree is 3, so its whole 4-node graph is a 3-core.
    assert clique_member["star_ratio"] == pytest.approx(0.75)
    assert clique_member["cluster_edge_density"] == pytest.approx(1.0)  # 6/6
    assert clique_member["k_core_number"] == 3

    # Singleton: no edges at all.
    assert lone["star_ratio"] == pytest.approx(0.0)
    assert lone["k_core_number"] == 0
    assert pd.isna(lone["cluster_edge_density"])


def test_topology_features_ignore_transactions_at_or_after_as_of():
    """Explicit leakage coverage for both new features, extending the same
    test the task called for.

    k_core_number never touches `txns` at all -- it's a pure read of
    `graph`, exactly like node_degree -- so it can't leak by construction,
    asserted here rather than left implicit.

    star_ratio's active-membership logic is the one genuinely at risk:
    this adds a 5th star member (uidL4, graph-linked to the hub) whose
    ONLY transaction is dated at/after as_of. If the as_of filter were
    bypassed anywhere in star_ratio's computation, uidL4 would wrongly
    count toward cluster_uid_count and change the ratio; it must not.
    """
    df, entity_ids, graph = _topology_frame()
    as_of = 150  # after the star cluster's rows (DT 100-103), before the clique's

    graph.add_edge("uidH", "uidL4")  # graph-linked, but no transaction yet

    future_row = pd.DataFrame(
        {
            "TransactionID": [14],
            "TransactionDT": [5000],  # after as_of
            "TransactionAmt": [999999.0],
            "P_emaildomain": ["future-only-domain.com"],
            "isFraud": [1],
        }
    )
    future_entity = _entity_ids([14], ["uidL4"])

    df_with_future = pd.concat([df, future_row], ignore_index=True)
    entity_ids_with_future = pd.concat([entity_ids, future_entity])

    with_future = compute_cluster_features(
        df_with_future, entity_ids_with_future, graph, as_of=as_of, include_topology=True
    )
    without_future = compute_cluster_features(
        df, entity_ids, graph, as_of=as_of, include_topology=True
    )

    star_uids = ["uidH", "uidL1", "uidL2", "uidL3"]
    pd.testing.assert_frame_equal(
        with_future.loc[star_uids].sort_index(),
        without_future.loc[star_uids].sort_index(),
    )

    # uidL4's only transaction is post-as_of -- it must not get a row at all,
    # in either run, regardless of its (real, permanent) graph edge to the hub.
    assert "uidL4" not in with_future.index
    assert "uidL4" not in without_future.index


# --- the explicit leakage test -----------------------------------------------


def test_compute_cluster_features_ignores_transactions_at_or_after_as_of():
    """A transaction dated at/after as_of must not change any feature value,
    even though it belongs to a uid that's already in the cluster and would
    visibly change the numbers if it were (wrongly) included:
    it's a huge amount, a fraud label, and a new email domain, timed
    long after the other rows -- exactly the ingredients that would move
    cluster_amt_cv, cluster_prior_fraud_share, cluster_velocity,
    uid_email_domain_count and cluster_txn_count if leaked in.
    """
    df, entity_ids, graph = _feature_frame()
    as_of = 2000

    future_row = pd.DataFrame(
        {
            "TransactionID": [99],
            "TransactionDT": [5000],  # after as_of
            "TransactionAmt": [999999.0],
            "P_emaildomain": ["future-only-domain.com"],
            "isFraud": [1],
        }
    )
    future_entity = _entity_ids([99], ["uidA"])

    df_with_future = pd.concat([df, future_row], ignore_index=True)
    entity_ids_with_future = pd.concat([entity_ids, future_entity])

    with_future = compute_cluster_features(
        df_with_future, entity_ids_with_future, graph, as_of=as_of
    )
    without_future = compute_cluster_features(df, entity_ids, graph, as_of=as_of)

    pd.testing.assert_frame_equal(
        with_future.loc[["uidA", "uidB", "uidC"]].sort_index(),
        without_future.loc[["uidA", "uidB", "uidC"]].sort_index(),
    )

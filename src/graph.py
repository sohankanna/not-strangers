"""Build an entity graph and derive cluster-level features, causally.

Nodes are uids (as produced by entities.resolve_entities). Two uids are
linked when they share a "strong identifier" -- see LINKAGE_RULES below for
the exact rules and the rationale for each; this list is meant to be
readable and defensible on its own, not just correct in code.

Hub guard: a value shared by an enormous number of uids (a default device
string like "Windows", a free-email domain like "gmail.com") is not evidence
of a relationship between any particular pair of uids -- it is a common
default. build_entity_graph excludes any value shared by more than
`max_degree` uids from linkage entirely, and reports what it excluded.

IMPORTANT, found empirically on the real data: `max_degree` needs to be well
below its own default (1000) or the graph collapses into one giant
supercluster. addr1 is a region-level code (a few hundred distinct values
across ~524k uid'd rows), and card3/card5 are almost constant (dominated by
one or two values, e.g. 150.0/226.0) -- so card_bank_addr is close to a proxy
for addr1 alone, not an independent signal. On the full dataset, sweeping
max_degree gave: 20 -> largest cluster 126 uids (0.06% of all uids); 30 ->
919 (0.46%); 35 -> 5,141 (2.58%); 1000 (the literal default) -> 127,708
(64%). There's a sharp phase transition between 30 and 35, not a gradual
one. run_pipeline.py calls build_entity_graph with max_degree=20 for this
reason -- the function's own default stays 1000 as specified, but that
default should not be used as-is for this dataset.

Causality: per CLAUDE.md, "Cluster features must be computed causally.
Graph structure and aggregates for a test transaction may only use
transactions with strictly earlier TransactionDT." compute_cluster_features
enforces this for every transaction-level aggregate (velocity, amount CV,
burst concentration, email heterogeneity, prior-fraud share) by filtering
`transactions` to strictly-before-`as_of` internally, regardless of what the
caller passes in.

For the *graph itself* (which uids are linked to which), this module does
not track a per-edge timestamp -- causal correctness for graph topology
relies on the caller building `graph` from transactions no later than
`as_of` in the first place. This is a deliberate scope decision, not an
oversight: in this project's actual pipeline (run_pipeline.py), the graph is
built once from train-period transactions only, and reused for both
train-period features (as_of unset, i.e. the whole train window) and
test-period features (as_of = the train/test boundary) -- both calls only
ever need edges established by train-period data, so a single train-only
graph is exactly right for both, with no per-edge timestamp bookkeeping
required. Passing a `graph` built from data at or after `as_of` would be a
caller error this module does not try to detect.

Topology features (k_core_number, star_ratio): additive-only. Existing
callers of compute_cluster_features are completely unaffected -- these two
are computed only when the caller explicitly passes include_topology=True,
so every existing call site (run_pipeline.py included) keeps getting
exactly the same 10 columns, computed exactly the same way, with the
default False. They exist to test a specific hypothesis: whether cluster
SHAPE (not just size/rate aggregates) predicts abuse -- a hub-and-spoke
device farm and a tight mutual clique of the same size look identical to
every existing feature here, but not to k-core depth or degree
concentration. Both are pure graph-topology reads (like node_degree
already is): computed from `graph` as given and `graph`'s degree/core
structure, not re-filtered by as_of internally, for the same reason
node_degree isn't -- causal correctness for graph topology is the
caller's responsibility (build `graph` from pre-as_of data), stated above.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400
BURST_WINDOW_SECONDS = 180  # 3 minutes


@dataclass(frozen=True)
class LinkageRule:
    """One rule for linking two uids, with the reasoning behind it."""

    name: str
    key_columns: tuple[str, ...]
    rationale: str


LINKAGE_RULES: tuple[LinkageRule, ...] = (
    LinkageRule(
        name="device_info",
        key_columns=("DeviceInfo",),
        rationale=(
            "The exact same device fingerprint appearing across multiple "
            "client identities is strong evidence of a shared physical "
            "device -- one operator running several card+address "
            "combinations, or a device farm."
        ),
    ),
    LinkageRule(
        name="addr1_email",
        key_columns=("addr1", "P_emaildomain"),
        rationale=(
            "A shared delivery/billing address code AND a shared purchaser "
            "email domain together are a stronger joint signal than either "
            "alone: many people share a common email provider, and many "
            "share a common region-coded addr1 value, but the co-occurrence "
            "of both narrows this considerably."
        ),
    ),
    LinkageRule(
        name="card_bank_addr",
        key_columns=("card3", "card5", "addr1"),
        rationale=(
            "card3/card5 are issuing-bank/network category codes that are "
            "essentially fixed once card1 is fixed (confirmed empirically "
            "in the D1 investigation -- they never varied across the 20 "
            "largest uids). Requiring them to match jointly WITH addr1 "
            "links different card1 values that share the same "
            "issuer/network profile and address -- catching a coordinated "
            "actor issuing multiple card numbers to the same address."
        ),
    ),
)


@dataclass
class EntityGraph:
    """The linkage graph plus a record of what the hub guard excluded.

    Attributes:
        graph: Undirected graph, nodes are uids, edges are pairs linked by
            at least one rule in LINKAGE_RULES. Each edge carries a "rules"
            attribute: the set of rule names that established it.
        excluded_hubs: One row per (rule, value) pair excluded by the hub
            guard, with columns "rule", "value", "uid_count".
    """

    graph: nx.Graph
    excluded_hubs: pd.DataFrame


def _prepare(transactions: pd.DataFrame, entity_ids: pd.Series) -> pd.DataFrame:
    """Align transactions to entity_ids and drop rows with no uid."""
    txns = transactions.copy()
    if "TransactionID" in txns.columns:
        txns = txns.set_index("TransactionID")
    txns["uid"] = entity_ids  # aligns by index label, not position
    return txns.loc[txns["uid"].notna()]


def build_entity_graph(
    transactions: pd.DataFrame,
    entity_ids: pd.Series,
    max_degree: int = 1000,
) -> EntityGraph:
    """Build a graph linking uids that share a strong identifier.

    Args:
        transactions: Raw transaction rows. DeviceInfo is used when present
            (e.g. transactions already left-joined with identity via
            src.data.load_transactions) but is not required -- the
            device_info rule is simply skipped if the column is absent.
        entity_ids: Series indexed by TransactionID mapping each transaction
            to its resolved uid (as produced by entities.resolve_entities).
        max_degree: A value shared by more than this many distinct uids is
            treated as a common default rather than evidence of a
            relationship, and excluded from linkage entirely.

    Returns:
        An EntityGraph: every uid with at least one row in `transactions` is
        a node (isolated if it shares no strong identifier with anyone),
        edges come from LINKAGE_RULES, and excluded_hubs records what the
        hub guard removed.
    """
    txns = _prepare(transactions, entity_ids)

    graph = nx.Graph()
    graph.add_nodes_from(txns["uid"].unique())

    excluded_hubs: list[dict] = []

    for rule in LINKAGE_RULES:
        cols = list(rule.key_columns)
        if not all(c in txns.columns for c in cols):
            continue

        sub = txns.dropna(subset=cols)
        if sub.empty:
            continue

        groups = sub.groupby(cols, observed=True)["uid"].unique()
        for key_value, uids in groups.items():
            n_uids = len(uids)
            if n_uids > max_degree:
                excluded_hubs.append(
                    {"rule": rule.name, "value": key_value, "uid_count": n_uids}
                )
                continue
            if n_uids < 2:
                continue
            for u1, u2 in itertools.combinations(uids, 2):
                if graph.has_edge(u1, u2):
                    graph[u1][u2]["rules"].add(rule.name)
                else:
                    graph.add_edge(u1, u2, rules={rule.name})

    excluded_df = pd.DataFrame(excluded_hubs, columns=["rule", "value", "uid_count"])
    return EntityGraph(graph=graph, excluded_hubs=excluded_df)


def get_connected_components(graph: nx.Graph) -> list[set[str]]:
    """Return the connected components of an entity graph as candidate rings.

    Args:
        graph: An entity graph, as produced by build_entity_graph(...).graph.

    Returns:
        A list of node-id sets, one per connected component (including
        singletons -- a uid linked to no one is its own component).
    """
    return list(nx.connected_components(graph))


def compute_cluster_features(
    transactions: pd.DataFrame,
    entity_ids: pd.Series,
    graph: nx.Graph,
    as_of: float | None = None,
    include_topology: bool = False,
) -> pd.DataFrame:
    """Compute cluster-level features for each entity, causally.

    Every returned value is computed only from transactions with
    TransactionDT strictly earlier than `as_of` (when given) -- this is
    enforced here, inside this function, not left to the caller to have
    already filtered.

    Args:
        transactions: Raw transaction rows. Must include TransactionID,
            TransactionDT, TransactionAmt, P_emaildomain and isFraud.
        entity_ids: Series indexed by TransactionID mapping each transaction
            to its resolved uid.
        graph: An entity graph (see build_entity_graph). Must have been
            built from transactions no later than `as_of` -- see this
            module's docstring for why that's the caller's responsibility.
        as_of: If given, only transactions with TransactionDT strictly
            earlier than this value contribute to any returned value. Set
            to the train/test split boundary when computing features for
            test-period transactions.
        include_topology: If True, also compute k_core_number and
            star_ratio (see Returns below). Defaults to False so every
            existing caller -- including run_pipeline.py, unmodified --
            keeps getting exactly the 10 columns it always has, computed
            exactly the same way. Additive-only: new code opts in
            explicitly rather than every caller getting new columns for
            free.

    Returns:
        A DataFrame indexed by entity_id (uid) with columns:
          - cluster_size_uids: distinct uids in the entity's cluster.
          - cluster_txn_count: transactions across the whole cluster.
          - cluster_edge_density: 2E / (V(V-1)) among currently-active
            cluster members; NaN for a singleton (no possible edges).
          - node_degree: this uid's own degree in `graph`.
          - cluster_velocity: cluster_txn_count / max(cluster's active
            day-span, 1) -- transactions per day.
          - cluster_amt_cv: std/mean of TransactionAmt across the cluster;
            NaN for a singleton (sample std undefined).
          - cluster_burst_concentration: the largest share of the cluster's
            transactions falling in any single 3-minute window.
          - uid_email_domain_count: distinct P_emaildomain values seen on
            this uid's own transactions.
          - cluster_email_uid_ratio: distinct P_emaildomain values across
            the whole cluster / distinct uids in the cluster -- operationalises
            "several people behind one card fingerprint": a ratio near 1
            means each member brings a different email domain (heterogeneous,
            many distinct people), near 0 means the cluster shares very few
            email domains despite having many members (homogeneous).
          - cluster_prior_fraud_share: share of the cluster's member uids
            with at least one isFraud=1 transaction, itself subject to the
            same as_of cutoff as everything else here.
          - k_core_number (only when include_topology=True): this uid's
            k-core index in `graph` (networkx's core_number) -- how deep the
            densest subgraph containing this node goes. A node in a 4-core
            has at least 4 neighbors that are themselves at least
            that embedded; a leaf's core number is at most 1. Like
            node_degree, this reads `graph` as given (already pre-as_of by
            construction) and defaults to 0 for a uid absent from the graph.
          - star_ratio (only when include_topology=True): the highest
            `node_degree` among the cluster's currently-active members,
            divided by cluster_size_uids. Close to 1 for a hub-and-spoke
            shape (one high-degree hub, everyone else degree ~1); a
            same-size mutual clique also trends toward 1 by this formula
            alone, so pair it with cluster_edge_density to tell the two
            apart (a clique has both a high star_ratio AND a high edge
            density; a star has a high star_ratio but a LOW edge density).

        Only uids with at least one qualifying (pre-as_of) transaction get a
        row -- a uid with no history yet as of the cutoff gets no row here,
        which callers should treat as "no cluster signal available" (left-join
        NaN), not zero.
    """
    txns = _prepare(transactions, entity_ids)
    if as_of is not None:
        txns = txns.loc[txns["TransactionDT"] < as_of]

    result_columns = [
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
    if include_topology:
        result_columns = result_columns + ["k_core_number", "star_ratio"]
    empty_result = pd.DataFrame(columns=result_columns)
    empty_result.index.name = "entity_id"
    if txns.empty:
        return empty_result

    # --- uid -> cluster_id, including uids active here but absent from graph ---
    components = get_connected_components(graph)
    cluster_of: dict[str, int] = {}
    for i, comp in enumerate(components):
        for u in comp:
            cluster_of[u] = i
    next_id = len(components)
    for u in txns["uid"].unique():
        if u not in cluster_of:
            cluster_of[u] = next_id
            next_id += 1
    txns = txns.copy()
    txns["cluster_id"] = txns["uid"].map(cluster_of)

    # --- per-cluster transaction-level aggregates ---
    by_cluster = txns.groupby("cluster_id")
    cluster_txn_count = by_cluster.size()
    cluster_uid_count = by_cluster["uid"].nunique()

    span_days = (
        by_cluster["TransactionDT"].max() - by_cluster["TransactionDT"].min()
    ) / SECONDS_PER_DAY
    cluster_velocity = cluster_txn_count / span_days.clip(lower=1)

    cluster_amt_cv = by_cluster["TransactionAmt"].apply(
        lambda s: s.std() / s.mean() if s.mean() not in (0, None) and len(s) > 1 else np.nan
    )

    bin_id = txns["TransactionDT"] // BURST_WINDOW_SECONDS
    bin_counts = txns.groupby(["cluster_id", bin_id]).size()
    max_bin_count = bin_counts.groupby(level=0).max()
    cluster_burst_concentration = max_bin_count / cluster_txn_count

    cluster_email_domains = by_cluster["P_emaildomain"].nunique(dropna=True)
    cluster_email_uid_ratio = cluster_email_domains / cluster_uid_count

    # --- per-uid transaction-level aggregates ---
    by_uid = txns.groupby("uid")
    uid_email_domain_count = by_uid["P_emaildomain"].nunique(dropna=True)
    uid_ever_fraud = by_uid["isFraud"].max()

    uid_to_cluster = txns.drop_duplicates("uid").set_index("uid")["cluster_id"]
    cluster_prior_fraud_share = uid_ever_fraud.groupby(uid_to_cluster).mean()

    # --- graph-topology features ---
    node_degree_full = dict(graph.degree())
    active_uids = set(txns["uid"].unique())
    active_edges = [
        (u, v)
        for u, v in graph.edges()
        if u in active_uids and v in active_uids
    ]
    if active_edges:
        edge_cluster_ids = [cluster_of[u] for u, _ in active_edges]
        edge_count_by_cluster = pd.Series(edge_cluster_ids).value_counts()
    else:
        edge_count_by_cluster = pd.Series(dtype="int64")
    possible_edges = cluster_uid_count * (cluster_uid_count - 1) / 2
    cluster_edge_density = (
        edge_count_by_cluster.reindex(cluster_uid_count.index).fillna(0)
        / possible_edges.replace(0, np.nan)
    )

    # --- topology features (only when requested -- see include_topology) ---
    per_cluster_extra: dict[str, pd.Series] = {}
    if include_topology:
        # star_ratio's numerator: each active member's own (whole-graph)
        # node_degree, maxed within its cluster -- the same node_degree_full
        # dict node_degree itself uses, same default-to-0 convention for a
        # uid absent from the graph.
        active_node_degree = txns["uid"].map(lambda u: node_degree_full.get(u, 0))
        max_node_degree_by_cluster = active_node_degree.groupby(txns["cluster_id"]).max()
        per_cluster_extra["star_ratio"] = max_node_degree_by_cluster / cluster_uid_count

    # --- assemble per-cluster table, then broadcast onto each uid ---
    per_cluster = pd.DataFrame(
        {
            "cluster_size_uids": cluster_uid_count,
            "cluster_txn_count": cluster_txn_count,
            "cluster_edge_density": cluster_edge_density,
            "cluster_velocity": cluster_velocity,
            "cluster_amt_cv": cluster_amt_cv,
            "cluster_burst_concentration": cluster_burst_concentration,
            "cluster_email_uid_ratio": cluster_email_uid_ratio,
            "cluster_prior_fraud_share": cluster_prior_fraud_share,
            **per_cluster_extra,
        }
    )

    result = per_cluster.loc[uid_to_cluster.to_numpy()].copy()
    result.index = uid_to_cluster.index
    result["node_degree"] = result.index.map(lambda u: node_degree_full.get(u, 0))
    result["uid_email_domain_count"] = uid_email_domain_count
    result.index.name = "entity_id"

    if include_topology:
        core_number_full = nx.core_number(graph)
        result["k_core_number"] = result.index.map(lambda u: core_number_full.get(u, 0))

    return result[list(empty_result.columns)]

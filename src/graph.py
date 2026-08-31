"""Build an entity graph and derive cluster-level features, causally.

Graph structure and aggregates for a test transaction may only use
transactions with TransactionDT strictly earlier than that transaction's
own TransactionDT -- see the "causal cluster features" rule in CLAUDE.md.
Computing features over the whole graph (including future transactions)
would leak test-period information into features used to predict that same
test period.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_entity_graph(transactions: pd.DataFrame, entity_ids: pd.Series) -> nx.Graph:
    """Build a graph connecting entities that share transactions or linkage keys.

    Args:
        transactions: Raw transaction rows.
        entity_ids: Series indexed by TransactionID mapping each transaction
            to its resolved entity_id (as produced by
            entities.resolve_entities).

    Returns:
        An undirected graph whose nodes are entity_ids.
    """
    raise NotImplementedError


def get_connected_components(graph: nx.Graph) -> list[set[str]]:
    """Return the connected components of an entity graph as candidate rings.

    Args:
        graph: An entity graph, as produced by build_entity_graph.

    Returns:
        A list of node-id sets, one per connected component.
    """
    raise NotImplementedError


def compute_cluster_features(
    transactions: pd.DataFrame,
    entity_ids: pd.Series,
    graph: nx.Graph,
    as_of: float | None = None,
) -> pd.DataFrame:
    """Compute cluster-level features for each entity, causally.

    Combines graph-topology features (cluster size, density, degree) with
    transaction-row features (velocity, amount variance, burst concentration)
    -- the latter is why `transactions` is a required argument rather than
    deriving everything from `graph` alone.

    Args:
        transactions: Raw transaction rows.
        entity_ids: Series indexed by TransactionID mapping each transaction
            to its resolved entity_id.
        graph: An entity graph, as produced by build_entity_graph.
        as_of: If given, only transactions with TransactionDT strictly
            earlier than this value may contribute to the returned features.
            Must be set to the start of the evaluation window when computing
            features for test-period transactions, so no feature for a test
            transaction is derived from transactions in that same test
            period.

    Returns:
        A DataFrame indexed by entity_id with one column per cluster feature.
    """
    raise NotImplementedError

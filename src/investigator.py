"""LLM layer: explains and prioritizes flagged clusters. Never decides.

This module may only produce human-readable explanations and a relative
ordering of clusters for an investigation queue. It must never determine an
allow/step_up/review action -- that is policy.py's sole responsibility. See
CLAUDE.md: "The LLM layer explains and prioritizes. policy.py decides. Never
merge them."
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ClusterExplanation:
    """A human-readable explanation of why a cluster was flagged.

    Attributes:
        cluster_id: The entity/cluster this explanation is about.
        entity_ids: The entities that make up the cluster.
        narrative: A natural-language summary of the suspicious pattern.
        evidence: The specific features/facts the narrative is grounded in.
        priority_score: A relative score for ranking clusters in the
            investigation queue. Not a policy decision.
    """

    cluster_id: str
    entity_ids: list[str]
    narrative: str
    evidence: dict[str, float]
    priority_score: float


def explain_cluster(
    cluster_id: str,
    cluster_features: pd.DataFrame,
    transactions: pd.DataFrame,
) -> ClusterExplanation:
    """Generate a natural-language explanation for a flagged cluster.

    Args:
        cluster_id: The entity/cluster to explain.
        cluster_features: Per-entity cluster features, as produced by
            graph.compute_cluster_features.
        transactions: Raw transaction rows belonging to the cluster.

    Returns:
        A ClusterExplanation for the given cluster.
    """
    raise NotImplementedError


def prioritize_clusters(
    explanations: list[ClusterExplanation],
) -> list[ClusterExplanation]:
    """Order cluster explanations for an investigation queue.

    Args:
        explanations: Explanations to rank, as produced by explain_cluster.

    Returns:
        The same explanations, ordered from highest to lowest priority.
    """
    raise NotImplementedError

"""Deterministic decision layer. Decides; never explains.

policy.py is the sole place decisions get made, driven by model scores and
cluster features against fixed thresholds. It must never be merged with
investigator.py -- see CLAUDE.md: "The LLM layer explains and prioritizes.
policy.py decides. Never merge them."

Actions are allow / step_up / review (no outright block): step_up is the
realistic payments action for a suspected but unconfirmed abuse ring, and is
cheaper on false positives than a block, which is what the cost_per_10k
analysis in evaluate.py is measuring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class PolicyDecision:
    """A decision made about a single entity.

    Attributes:
        entity_id: The entity this decision applies to.
        action: One of "allow", "step_up", "review".
        reason: A short machine-generated justification (score/threshold
            based, not an LLM narrative).
    """

    entity_id: str
    action: Literal["allow", "step_up", "review"]
    reason: str


def decide(
    entity_id: str,
    model_score: float,
    cluster_features: pd.DataFrame,
    thresholds: dict[str, float],
) -> PolicyDecision:
    """Decide an action for a single entity.

    Args:
        entity_id: The entity to decide on.
        model_score: The model's abuse score for this entity.
        cluster_features: Per-entity cluster features, as produced by
            graph.compute_cluster_features.
        thresholds: Score thresholds keyed by action name (e.g.
            "step_up", "review").

    Returns:
        The PolicyDecision for this entity.
    """
    raise NotImplementedError


def apply_policy(
    scores: pd.DataFrame, thresholds: dict[str, float]
) -> pd.DataFrame:
    """Apply the policy to a batch of scored entities.

    Args:
        scores: A DataFrame indexed by entity_id with a model score column
            and any cluster feature columns needed by decide().
        thresholds: Score thresholds keyed by action name.

    Returns:
        A DataFrame indexed by entity_id with "action" and "reason" columns.
    """
    raise NotImplementedError

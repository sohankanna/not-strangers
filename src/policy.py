"""Deterministic decision layer. Decides; never explains.

policy.py is the sole place decisions get made, driven by model scores
against fixed thresholds. It must never be merged with investigator.py --
see CLAUDE.md: "The LLM layer explains and prioritizes. policy.py decides.
Never merge them." This module does not import investigator (see
tests/test_policy.py for both a static-import check and a behavioral one:
decisions are identical whether or not investigator/an API key is
available).

Actions are allow / step_up / review (no outright block): step_up is the
realistic payments action for a suspected but unconfirmed abuse ring, and
is cheaper on false positives than a block, which is what the cost_per_10k
analysis in evaluate.py is measuring.

Thresholds are read from real cost-curve sweeps
(scripts/derive_policy_thresholds.py), not hand-picked:
  - STEP_UP_THRESHOLD = 0.0103: the cost-minimizing threshold for
    (cost_fn=500, cost_fp=5) on the cluster model's test scores. This is
    the exact same sweep already reported in results/ablation.md's
    "Threshold sweep and cost curve" section (the "cluster" row) --
    step_up is a light, automated friction cost.
  - REVIEW_THRESHOLD = 0.1843: the cost-minimizing threshold for
    (cost_fn=500, cost_fp=50). A full manual review is assumed 10x
    costlier per false positive than a step-up challenge (analyst time, a
    held transaction, worse customer experience). The 10x multiplier is
    illustrative, not a Razorpay figure -- same caveat as every other cost
    assumption in this project.

Every decision can be turned into an audit record via build_audit_record --
see scripts/write_audit_sample.py for the script that actually produces
results/audit_sample.jsonl from a real batch of test-period transactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src import model as _model

STEP_UP_THRESHOLD = 0.0103
REVIEW_THRESHOLD = 0.1843

DEFAULT_THRESHOLDS: dict[str, float] = {
    "step_up": STEP_UP_THRESHOLD,
    "review": REVIEW_THRESHOLD,
}

# Derived from model.py's own frozen hyperparameters (read, not modified)
# so the version string can't silently drift from what actually trained.
MODEL_VERSION = f"cluster_seed{_model.SEED}_boost{_model.NUM_BOOST_ROUND}"


@dataclass
class PolicyDecision:
    """A decision made about a single entity.

    Attributes:
        entity_id: The entity this decision applies to.
        action: One of "allow", "step_up", "review".
        reason: A short machine-generated justification (score/threshold
            based, not an LLM narrative).
        threshold_applied: The specific threshold value that determined
            this action (STEP_UP_THRESHOLD or REVIEW_THRESHOLD, from
            `thresholds`).
    """

    entity_id: str
    action: Literal["allow", "step_up", "review"]
    reason: str
    threshold_applied: float


def decide(
    entity_id: str,
    model_score: float,
    cluster_features: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> PolicyDecision:
    """Decide an action for a single entity.

    The decision itself is purely score-vs-threshold -- cluster_features is
    not used to adjust the threshold (thresholds are fixed, cost-derived
    constants, not conditioned on cluster state). It's accepted here so
    callers building an audit record have it in hand alongside the
    decision; see build_audit_record.

    Args:
        entity_id: The entity to decide on.
        model_score: The model's abuse score for this entity.
        cluster_features: Per-entity cluster features, as produced by
            graph.compute_cluster_features. Unused in the comparison
            itself.
        thresholds: Score thresholds keyed by "step_up"/"review"; defaults
            to DEFAULT_THRESHOLDS (see module docstring for provenance).

    Returns:
        The PolicyDecision for this entity.
    """
    del cluster_features  # not used in the decision itself; see docstring
    t = thresholds or DEFAULT_THRESHOLDS
    step_up_t, review_t = t["step_up"], t["review"]

    if model_score >= review_t:
        return PolicyDecision(
            entity_id=entity_id,
            action="review",
            reason=f"score {model_score:.4f} >= review threshold {review_t:.4f}",
            threshold_applied=review_t,
        )
    if model_score >= step_up_t:
        return PolicyDecision(
            entity_id=entity_id,
            action="step_up",
            reason=(
                f"score {model_score:.4f} >= step_up threshold {step_up_t:.4f} "
                f"(< review threshold {review_t:.4f})"
            ),
            threshold_applied=step_up_t,
        )
    return PolicyDecision(
        entity_id=entity_id,
        action="allow",
        reason=f"score {model_score:.4f} < step_up threshold {step_up_t:.4f}",
        threshold_applied=step_up_t,
    )


def apply_policy(
    scores: pd.DataFrame, thresholds: dict[str, float] | None = None
) -> pd.DataFrame:
    """Apply the policy to a batch of scored entities.

    Vectorized batch path -- logically equivalent to calling decide() for
    every row (see tests/test_policy.py for the equivalence test), kept
    separate for performance on the full ~199k-entity population.

    Args:
        scores: A DataFrame indexed by entity_id with a "score" column.
        thresholds: Score thresholds keyed by "step_up"/"review"; defaults
            to DEFAULT_THRESHOLDS.

    Returns:
        A DataFrame indexed by entity_id with "action", "reason", and
        "threshold_applied" columns.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    step_up_t, review_t = t["step_up"], t["review"]
    score = scores["score"]

    action = pd.Series("allow", index=scores.index, dtype=object)
    action[score >= step_up_t] = "step_up"
    action[score >= review_t] = "review"

    threshold_applied = pd.Series(step_up_t, index=scores.index, dtype=float)
    threshold_applied[score >= review_t] = review_t

    def _reason(s: float) -> str:
        if s >= review_t:
            return f"score {s:.4f} >= review threshold {review_t:.4f}"
        if s >= step_up_t:
            return (
                f"score {s:.4f} >= step_up threshold {step_up_t:.4f} "
                f"(< review threshold {review_t:.4f})"
            )
        return f"score {s:.4f} < step_up threshold {step_up_t:.4f}"

    reason = score.map(_reason)

    return pd.DataFrame(
        {"action": action, "reason": reason, "threshold_applied": threshold_applied},
        index=scores.index,
    )


def build_audit_record(
    decision: PolicyDecision,
    transaction_id,
    uid: str,
    score: float,
    feature_values: dict,
    timestamp: str,
    model_version: str = MODEL_VERSION,
) -> dict:
    """Assemble one JSON-serializable audit-trail record for a decision.

    Pure data assembly, no I/O -- see scripts/write_audit_sample.py for the
    script that writes a real batch to results/audit_sample.jsonl.

    Args:
        decision: The PolicyDecision this record documents.
        transaction_id: The specific transaction this record is for (a
            decision is made per-entity/uid, but every transaction that
            uid touches gets its own audit record for traceability).
        uid: The entity/uid the decision was made for.
        score: The model score used for this decision.
        feature_values: The cluster feature values in effect at decision
            time (for investigative context; not what drove the
            threshold comparison itself, which is score-only).
        timestamp: ISO-8601 timestamp of when this record was produced.
        model_version: Defaults to this module's MODEL_VERSION.

    Returns:
        A flat, JSON-serializable dict.
    """
    return {
        "transaction_id": transaction_id,
        "uid": uid,
        "model_version": model_version,
        "score": score,
        "threshold_applied": decision.threshold_applied,
        "feature_values": feature_values,
        "action": decision.action,
        "reason": decision.reason,
        "timestamp": timestamp,
    }

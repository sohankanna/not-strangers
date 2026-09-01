"""LLM layer: explains and prioritizes flagged clusters. Never decides.

This module may only produce human-readable explanations and a relative
ordering of clusters for an investigation queue. It must never determine an
allow/step_up/review action -- that is policy.py's sole responsibility. See
CLAUDE.md: "The LLM layer explains and prioritizes. policy.py decides. Never
merge them." (policy.py has no import of this module -- see its own
docstring and tests/test_policy.py.)

Uses the Anthropic API (model claude-sonnet-4-6, ANTHROPIC_API_KEY read from
the environment) to write a narrative for a cluster. The evidence dict is
passed to the model as structured JSON (not prose), and the system prompt
explicitly forbids the model from stating any number not already present in
that JSON -- this is checked programmatically, not just requested, in
scripts/eval_investigator.py.

If ANTHROPIC_API_KEY is an identity-linked (workspace) key, the API also
requires an `anthropic-workspace-id` header -- set ANTHROPIC_WORKSPACE_ID
in the environment and it's passed as a default header on every request.
It's omitted entirely when unset, so a plain (non-identity-linked) key
keeps working unchanged. See README.md's setup section.

Graceful degradation: if ANTHROPIC_API_KEY is unset, or the API call fails
for any reason (network, rate limit, malformed response, a missing
workspace header, an empty balance), explain_cluster falls back to a
deterministic template narrative built by directly formatting the evidence
dict's own values, and marks the result with source="ungrounded-fallback".
This module must never raise for lack of a key or a flaky network call --
`make results` has to complete for a reviewer who has never set
ANTHROPIC_API_KEY.

The fallback path staying silent is exactly what let a whole session run
with 30/30 explanations silently falling back while a report claimed "at
least one explanation used the real LLM path" -- see DEVLOG.md's entry on
this. Graceful degradation is still correct (a bad key must not crash
`make results`), but the failure itself must never be invisible: every
fallback caused by an exception logs the exception's type and message to
stderr, and records it on the ClusterExplanation (`error`) so any report
built from these can state plainly what actually happened, derived from
the real outcome rather than from whether a key happened to be present.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import pandas as pd

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 400

_SYSTEM_PROMPT = """\
You are a fraud-risk analyst writing a short briefing note about a cluster \
of linked transaction identities (a "uid" is a persistent client identity; \
a cluster is a group of uids linked by a shared device, address+email, or \
card/bank/address combination).

You will be given a JSON object of evidence about ONE cluster. Write a \
concise 3-5 sentence narrative explaining what the evidence suggests about \
this cluster -- whether it looks like coordinated abuse, looks benign, or \
is ambiguous. Be specific about which evidence fields support your read.

HARD RULE, more important than anything else in this prompt: you must \
NEVER state a number, percentage, count, or statistic that is not present \
verbatim (or trivially rounded, e.g. 0.78 from 0.7797) in the evidence \
JSON below. Do not compute derived numbers (no new ratios, no rounding to \
a different unit, no estimates, no "approximately X"). If you want to \
convey magnitude without a number from the evidence, use a qualitative \
word ("high", "several", "concentrated") instead of inventing a figure. \
This is checked programmatically after you respond; an invented number is \
a failure of this task, not a minor style issue.

Return ONLY the narrative text, no preamble, no headers, no markdown.
"""


@dataclass
class ClusterExplanation:
    """A human-readable explanation of why a cluster was flagged.

    Attributes:
        cluster_id: The entity/cluster this explanation is about.
        entity_ids: The entities (uids) that make up the cluster.
        narrative: A natural-language summary of the suspicious pattern.
        evidence: The specific features/facts the narrative is grounded in
            -- every number in `narrative` must trace back to a value here.
        priority_score: A relative score for ranking clusters in the
            investigation queue. Not a policy decision.
        source: "llm" if `narrative` came from the Anthropic API,
            "ungrounded-fallback" if it came from the deterministic
            template because no API key was available or the call failed.
        error: None on success. Otherwise the reason the fallback path was
            taken -- "ANTHROPIC_API_KEY not set", or "{ExceptionType}:
            {message}" for a failed API call. Always set when
            source="ungrounded-fallback", always None when source="llm".
    """

    cluster_id: str
    entity_ids: list[str]
    narrative: str
    evidence: dict
    priority_score: float
    source: str = "llm"
    error: str | None = None


def _round_evidence_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        return round(value, 4)
    return value


def build_evidence(cluster_features: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    """Assemble the JSON-serializable evidence dict for a cluster.

    Args:
        cluster_features: The rows of graph.compute_cluster_features's
            output belonging to this cluster's member uids (one row per
            member; cluster-level columns repeat, per-uid columns vary).
        transactions: Raw transaction rows belonging to this cluster's
            member uids.

    Returns:
        A flat dict of JSON-serializable numbers -- every value here is a
        candidate for the LLM (or the fallback template) to cite.
    """
    first = cluster_features.iloc[0]
    evidence = {
        "cluster_size_uids": int(first["cluster_size_uids"]),
        "cluster_txn_count": int(first["cluster_txn_count"]),
        "cluster_edge_density": _round_evidence_value(float(first["cluster_edge_density"]))
        if pd.notna(first["cluster_edge_density"])
        else None,
        "cluster_velocity": _round_evidence_value(float(first["cluster_velocity"])),
        "cluster_amt_cv": _round_evidence_value(float(first["cluster_amt_cv"]))
        if pd.notna(first["cluster_amt_cv"])
        else None,
        "cluster_burst_concentration": _round_evidence_value(
            float(first["cluster_burst_concentration"])
        ),
        "cluster_email_uid_ratio": _round_evidence_value(
            float(first["cluster_email_uid_ratio"])
        ),
        "cluster_prior_fraud_share": _round_evidence_value(
            float(first["cluster_prior_fraud_share"])
        ),
        "total_transaction_amount": _round_evidence_value(
            float(transactions["TransactionAmt"].sum())
        ),
        "mean_transaction_amount": _round_evidence_value(
            float(transactions["TransactionAmt"].mean())
        ),
        "distinct_product_codes": int(transactions["ProductCD"].nunique())
        if "ProductCD" in transactions.columns
        else None,
        "distinct_email_domains": int(transactions["P_emaildomain"].nunique())
        if "P_emaildomain" in transactions.columns
        else None,
    }
    return {k: v for k, v in evidence.items() if v is not None}


def _priority_score(evidence: dict) -> float:
    """Heuristic ranking score for the investigation queue.

    Not a policy decision -- just an ordering for where an investigator
    should look first. Weights prior-fraud share heaviest (it's the
    strongest signal per results/ablation.md's sanity checks), with
    transaction volume and burst concentration as tie-breakers.
    """
    prior_fraud = evidence.get("cluster_prior_fraud_share") or 0.0
    txn_count = evidence.get("cluster_txn_count") or 0
    burst = evidence.get("cluster_burst_concentration") or 0.0
    return float(prior_fraud * 100 + burst * 10 + min(txn_count, 100) * 0.1)


def _fallback_narrative(evidence: dict) -> str:
    """A deterministic narrative built by directly formatting evidence
    values -- grounded by construction, since it contains no number that
    isn't a literal evidence value.
    """
    parts = [
        f"{key.replace('_', ' ')}={value}" for key, value in sorted(evidence.items())
    ]
    return (
        "[template fallback -- no LLM call was made or it failed] "
        "Cluster summary: " + "; ".join(parts) + "."
    )


def _call_anthropic(evidence: dict) -> str:
    import anthropic

    api_key = os.environ["ANTHROPIC_API_KEY"]
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client_kwargs: dict = {"api_key": api_key}
    if workspace_id:
        client_kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}

    client = anthropic.Anthropic(**client_kwargs)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(evidence, indent=2)}],
    )
    return response.content[0].text.strip()


def explain_cluster(
    cluster_id: str,
    cluster_features: pd.DataFrame,
    transactions: pd.DataFrame,
) -> ClusterExplanation:
    """Generate a natural-language explanation for a flagged cluster.

    Args:
        cluster_id: A label identifying the cluster (e.g. a representative
            member uid, or an integer cluster index as a string).
        cluster_features: The rows of graph.compute_cluster_features's
            output belonging to this cluster's member uids.
        transactions: Raw transaction rows belonging to this cluster's
            member uids.

    Returns:
        A ClusterExplanation. `source` is "llm" on success, or
        "ungrounded-fallback" if ANTHROPIC_API_KEY was unset or the API
        call failed -- this function never raises for either reason, but
        every fallback caused by a failed call logs the exception to
        stderr and records it in `.error` (see the module docstring for
        why this matters: a silent fallback here previously hid an entire
        session's worth of unmeasured LLM calls).
    """
    evidence = build_evidence(cluster_features, transactions)
    entity_ids = cluster_features.index.tolist()
    priority_score = _priority_score(evidence)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        error_message = "ANTHROPIC_API_KEY not set"
        print(
            f"[investigator] {cluster_id}: {error_message}; using fallback narrative",
            file=sys.stderr,
        )
        return ClusterExplanation(
            cluster_id=cluster_id,
            entity_ids=entity_ids,
            narrative=_fallback_narrative(evidence),
            evidence=evidence,
            priority_score=priority_score,
            source="ungrounded-fallback",
            error=error_message,
        )

    try:
        narrative = _call_anthropic(evidence)
        return ClusterExplanation(
            cluster_id=cluster_id,
            entity_ids=entity_ids,
            narrative=narrative,
            evidence=evidence,
            priority_score=priority_score,
            source="llm",
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(
            f"[investigator] {cluster_id}: Anthropic API call failed "
            f"({error_message}); using fallback narrative",
            file=sys.stderr,
        )
        return ClusterExplanation(
            cluster_id=cluster_id,
            entity_ids=entity_ids,
            narrative=_fallback_narrative(evidence),
            evidence=evidence,
            priority_score=priority_score,
            source="ungrounded-fallback",
            error=error_message,
        )


def prioritize_clusters(
    explanations: list[ClusterExplanation],
) -> list[ClusterExplanation]:
    """Order cluster explanations for an investigation queue.

    Args:
        explanations: Explanations to rank, as produced by explain_cluster.

    Returns:
        The same explanations, ordered from highest to lowest priority_score.
    """
    return sorted(explanations, key=lambda e: e.priority_score, reverse=True)

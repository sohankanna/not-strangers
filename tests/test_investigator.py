"""Tests for src/investigator.py: graceful degradation, groundedness of the
fallback path, and priority ordering. No live API calls -- see
scripts/eval_investigator.py for that (and its honest caveat about whether
a key was actually available when it last ran).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.investigator import (
    ClusterExplanation,
    build_evidence,
    explain_cluster,
    prioritize_clusters,
)


def _cluster_features(n_members: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_size_uids": [n_members] * n_members,
            "cluster_txn_count": [10] * n_members,
            "cluster_edge_density": [1.0] * n_members,
            "cluster_velocity": [5.0] * n_members,
            "cluster_amt_cv": [0.3] * n_members,
            "cluster_burst_concentration": [0.6] * n_members,
            "cluster_email_uid_ratio": [0.5] * n_members,
            "cluster_prior_fraud_share": [0.25] * n_members,
            "node_degree": list(range(1, n_members + 1)),
            "uid_email_domain_count": [1] * n_members,
        },
        index=pd.Index([f"uid{i}" for i in range(n_members)], name="entity_id"),
    )


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionAmt": [100.0, 200.0, 150.0],
            "ProductCD": ["W", "W", "C"],
            "P_emaildomain": ["a.com", "b.com", "a.com"],
        }
    )


def test_explain_cluster_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = explain_cluster("cluster-1", _cluster_features(), _transactions())

    assert result.source == "ungrounded-fallback"
    assert result.entity_ids == ["uid0", "uid1"]
    assert isinstance(result.narrative, str) and result.narrative


def test_explain_cluster_falls_back_when_api_call_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    import src.investigator as investigator_module

    def _boom(evidence):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(investigator_module, "_call_anthropic", _boom)

    result = explain_cluster("cluster-1", _cluster_features(), _transactions())

    assert result.source == "ungrounded-fallback"


def test_explain_cluster_uses_llm_path_when_call_succeeds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    import src.investigator as investigator_module

    monkeypatch.setattr(
        investigator_module, "_call_anthropic", lambda evidence: "a clean narrative"
    )

    result = explain_cluster("cluster-1", _cluster_features(), _transactions())

    assert result.source == "llm"
    assert result.narrative == "a clean narrative"


def test_fallback_narrative_is_grounded_by_construction(monkeypatch):
    evidence = build_evidence(_cluster_features(), _transactions())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    explanation = explain_cluster("cluster-1", _cluster_features(), _transactions())

    # Every numeric evidence value must appear as text in the fallback
    # narrative -- it's built by directly formatting the evidence dict.
    for value in evidence.values():
        assert str(value) in explanation.narrative


def test_build_evidence_contains_only_json_serializable_numbers():
    evidence = build_evidence(_cluster_features(), _transactions())

    assert evidence["cluster_size_uids"] == 2
    assert evidence["cluster_txn_count"] == 10
    for key, value in evidence.items():
        assert isinstance(value, (int, float)), f"{key} is not numeric: {value!r}"


def test_prioritize_clusters_orders_by_priority_score_descending():
    low = ClusterExplanation(
        cluster_id="low", entity_ids=["u1"], narrative="", evidence={}, priority_score=1.0
    )
    high = ClusterExplanation(
        cluster_id="high", entity_ids=["u2"], narrative="", evidence={}, priority_score=9.0
    )
    mid = ClusterExplanation(
        cluster_id="mid", entity_ids=["u3"], narrative="", evidence={}, priority_score=5.0
    )

    ordered = prioritize_clusters([low, high, mid])

    assert [e.cluster_id for e in ordered] == ["high", "mid", "low"]

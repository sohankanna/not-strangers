"""Tests for src/policy.py: decision logic, the vectorized/scalar
equivalence, the audit record shape, and the architectural separation from
investigator.py (both statically and behaviorally).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import policy


def test_decide_allow_below_step_up_threshold():
    d = policy.decide("u1", 0.001, pd.DataFrame())
    assert d.action == "allow"
    assert d.threshold_applied == policy.STEP_UP_THRESHOLD


def test_decide_step_up_between_thresholds():
    mid = (policy.STEP_UP_THRESHOLD + policy.REVIEW_THRESHOLD) / 2
    d = policy.decide("u1", mid, pd.DataFrame())
    assert d.action == "step_up"
    assert d.threshold_applied == policy.STEP_UP_THRESHOLD


def test_decide_review_at_or_above_review_threshold():
    d = policy.decide("u1", policy.REVIEW_THRESHOLD, pd.DataFrame())
    assert d.action == "review"
    assert d.threshold_applied == policy.REVIEW_THRESHOLD


def test_decide_boundaries_are_inclusive_on_the_low_side():
    # score exactly at STEP_UP_THRESHOLD should already be step_up, not allow
    d = policy.decide("u1", policy.STEP_UP_THRESHOLD, pd.DataFrame())
    assert d.action == "step_up"


def test_apply_policy_matches_decide_row_by_row():
    scores = pd.DataFrame(
        {"score": [0.0, 0.005, policy.STEP_UP_THRESHOLD, 0.05, policy.REVIEW_THRESHOLD, 0.9]},
        index=["u1", "u2", "u3", "u4", "u5", "u6"],
    )

    batch = policy.apply_policy(scores)

    for entity_id, row in scores.iterrows():
        single = policy.decide(entity_id, row["score"], pd.DataFrame())
        assert batch.loc[entity_id, "action"] == single.action
        assert batch.loc[entity_id, "threshold_applied"] == single.threshold_applied
        assert batch.loc[entity_id, "reason"] == single.reason


def test_build_audit_record_shape():
    decision = policy.decide("u1", 0.5, pd.DataFrame())
    record = policy.build_audit_record(
        decision,
        transaction_id=12345,
        uid="u1",
        score=0.5,
        feature_values={"cluster_size_uids": 3},
        timestamp="2026-08-31T00:00:00Z",
    )

    assert record["transaction_id"] == 12345
    assert record["uid"] == "u1"
    assert record["action"] == decision.action
    assert record["threshold_applied"] == decision.threshold_applied
    assert record["model_version"] == policy.MODEL_VERSION
    assert record["feature_values"] == {"cluster_size_uids": 3}


def test_policy_module_does_not_import_investigator():
    source = Path(policy.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("investigator" in name for name in imported)


def test_decisions_identical_with_investigator_disabled(monkeypatch):
    scores = pd.DataFrame(
        {"score": [0.001, 0.02, 0.5]}, index=["u1", "u2", "u3"]
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    result_enabled = policy.apply_policy(scores)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "src.investigator", None)  # simulate unavailable
    result_disabled = policy.apply_policy(scores)

    pd.testing.assert_frame_equal(result_enabled, result_disabled)

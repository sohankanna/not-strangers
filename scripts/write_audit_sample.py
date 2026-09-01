"""Task 3: write a real audit trail sample from the actual test-period
scores, using policy.py's decisions.

Scores every test-period transaction with the cluster model, applies
policy.apply_policy at the uid level, then writes one audit record per
TRANSACTION (not per uid) to results/audit_sample.jsonl -- a uid's decision
is shared across all of its transactions, but each transaction gets its own
traceable record. Writes a sample (SAMPLE_SIZE), not the full ~118k-row
test set, since this is a demonstration artifact, not a production log.

Usage:
    python scripts/write_audit_sample.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src import policy, run_pipeline

RESULTS_DIR = REPO_ROOT / "results"
SAMPLE_SIZE = 200
FEATURE_COLUMNS = [
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


def _clean_feature_values(row: pd.Series) -> dict:
    values = {}
    for col in FEATURE_COLUMNS:
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            values[col] = None
        elif isinstance(v, float):
            values[col] = round(float(v), 4)
        else:
            values[col] = v
    return values


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    y_score = trained.cluster_model.predict(trained.X_test_cluster)
    test_uid = pipeline_data.entity_ids.reindex(trained.y_test.index)

    per_transaction_scores = pd.DataFrame(
        {"score": y_score, "uid": test_uid.to_numpy()}, index=trained.y_test.index
    )

    # Decide once per uid (policy operates on entities, not transactions),
    # using each uid's own score. A uid with multiple test transactions
    # gets one score per transaction from the model; policy.apply_policy
    # is applied per-transaction here since that's the granularity the
    # audit trail needs, and decide()'s output only depends on the score.
    decisions = policy.apply_policy(per_transaction_scores[["score"]])

    cluster_features_by_txn = run_pipeline.broadcast_cluster_features(
        pipeline_data.entity_ids, pipeline_data.cluster_features
    ).reindex(trained.y_test.index)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        per_transaction_scores.index.to_numpy(), size=SAMPLE_SIZE, replace=False
    )
    sample_idx = pd.Index(sample_idx).sort_values()

    timestamp = datetime.now(timezone.utc).isoformat()

    records = []
    for transaction_id in sample_idx:
        score = float(per_transaction_scores.loc[transaction_id, "score"])
        uid = per_transaction_scores.loc[transaction_id, "uid"]
        decision_row = decisions.loc[transaction_id]
        decision = policy.PolicyDecision(
            entity_id=uid if pd.notna(uid) else str(transaction_id),
            action=decision_row["action"],
            reason=decision_row["reason"],
            threshold_applied=float(decision_row["threshold_applied"]),
        )
        feature_values = (
            _clean_feature_values(cluster_features_by_txn.loc[transaction_id])
            if transaction_id in cluster_features_by_txn.index
            else {col: None for col in FEATURE_COLUMNS}
        )
        record = policy.build_audit_record(
            decision,
            transaction_id=int(transaction_id),
            uid=uid if pd.notna(uid) else None,
            score=round(score, 6),
            feature_values=feature_values,
            timestamp=timestamp,
        )
        records.append(record)

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "audit_sample.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    action_counts = pd.Series([r["action"] for r in records]).value_counts().to_dict()
    print(f"Wrote {len(records)} audit records to results/audit_sample.jsonl")
    print(f"Action counts in sample: {action_counts}")


if __name__ == "__main__":
    main()

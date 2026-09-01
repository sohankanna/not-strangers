"""Task 6: qualitative case studies of the 3 highest-priority clusters.

There is no ground truth for "this is a real coordinated ring" anywhere in
this dataset (see README.md's Limitations section) -- qualitative
inspection of what's actually in a cluster is the only validation
available here, not a substitute for one. This script assembles the raw
material (uid count, transaction count, shared identifiers, time
distribution, amounts, fraud labels) for the top 3 clusters by
investigator.py's priority_score; the actual write-up of what it means,
including calling out anything that looks like a false positive, is done
by hand in results/case_studies.md after reading this script's output --
not templated, since "does this look like a real ring" is a judgment call
this script cannot make for you.

Usage:
    python scripts/case_studies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src import investigator, run_pipeline
from src.graph import get_connected_components

RESULTS_DIR = REPO_ROOT / "results"
TOP_N = 3

SHARED_ID_COLUMNS = ["DeviceInfo", "addr1", "P_emaildomain", "card3", "card5"]


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()

    full = pipeline_data.df.set_index("TransactionID")
    full = pd.concat([full, pipeline_data.entity_ids.rename("uid")], axis=1)

    components = get_connected_components(pipeline_data.entity_graph.graph)
    multi = [c for c in components if len(c) >= 2]

    explanations = []
    for i, members in enumerate(multi):
        members_sorted = sorted(members)
        cf = pipeline_data.cluster_features.loc[
            [m for m in members_sorted if m in pipeline_data.cluster_features.index]
        ]
        if cf.empty:
            continue
        txns = full[full["uid"].isin(members_sorted)]
        explanations.append(
            investigator.explain_cluster(f"cluster-{i}", cf, txns)
        )

    ranked = investigator.prioritize_clusters(explanations)
    top = ranked[:TOP_N]

    for rank, explanation in enumerate(top, start=1):
        members = sorted(explanation.entity_ids)
        txns = full[full["uid"].isin(members)].copy()
        print(f"\n{'=' * 80}\nRANK {rank}: {explanation.cluster_id}\n{'=' * 80}")
        print(f"priority_score: {explanation.priority_score:.4f}")
        print(f"n_members: {len(members)}, n_transactions: {len(txns)}")
        print(f"members: {members}")
        print(f"evidence: {explanation.evidence}")
        print(f"narrative (source={explanation.source}): {explanation.narrative}")

        print("\n-- per-uid breakdown --")
        per_uid = txns.groupby("uid").agg(
            n_txns=("TransactionAmt", "size"),
            total_amt=("TransactionAmt", "sum"),
            n_fraud=("isFraud", "sum"),
            min_dt=("TransactionDT", "min"),
            max_dt=("TransactionDT", "max"),
        )
        print(per_uid.to_string())

        print("\n-- shared identifier columns across the cluster --")
        for col in SHARED_ID_COLUMNS:
            if col not in txns.columns:
                continue
            counts = txns[col].value_counts(dropna=True)
            print(f"{col}: {dict(counts.head(5))}")

        print("\n-- amount distribution --")
        print(txns["TransactionAmt"].describe().to_string())

        print("\n-- time span --")
        day_span = (txns["TransactionDT"].max() - txns["TransactionDT"].min()) / 86400
        print(f"min_dt={txns['TransactionDT'].min():,.0f} max_dt={txns['TransactionDT'].max():,.0f} span_days={day_span:.2f}")
        print(f"as_of={pipeline_data.as_of:,.0f} (train/test boundary)")

        print("\n-- fraud labels --")
        print(f"n_fraud_txns={int(txns['isFraud'].sum())} / {len(txns)}  fraud_rate={txns['isFraud'].mean():.4f}")

        print("\n-- ProductCD --")
        if "ProductCD" in txns.columns:
            print(dict(txns["ProductCD"].value_counts()))


if __name__ == "__main__":
    main()

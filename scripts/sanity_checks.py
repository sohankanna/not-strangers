"""Task 3: adversarial sanity checks on the ablation, before believing any
lift.

  1. Correlation of every cluster feature with isFraud -- anything above
     0.5 is flagged.
  2. An explicit, concrete trace of cluster_prior_fraud_share on real data,
     to check it only ever uses pre-as_of labels (the most likely leak in
     the pipeline).
  3. The ablation re-run with cluster_prior_fraud_share removed entirely,
     as a second row appended to results/ablation.md's results table.
  4. A concrete check that a test-period transaction's cluster assignment
     never depends on test-period edges.

Appends a "## Sanity checks" section to results/ablation.md (not a separate
file) since #3 explicitly asks for a second row in that same table.

Usage:
    python scripts/sanity_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src import evaluate, model, run_pipeline
from src.graph import get_connected_components

RESULTS_DIR = REPO_ROOT / "results"

CLUSTER_FEATURE_COLUMNS = [
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


def _correlation_section(trained: run_pipeline.TrainedModels) -> list[str]:
    lines = ["### 1. Correlation of cluster features with isFraud (train set)", ""]

    rows = []
    for col in CLUSTER_FEATURE_COLUMNS:
        series = trained.X_train_cluster[col]
        n_valid = series.notna().sum()
        c = series.corr(trained.y_train)
        rows.append((col, c, n_valid))
    rows.sort(key=lambda r: (abs(r[1]) if pd.notna(r[1]) else -1), reverse=True)

    lines.append(
        f"Computed over the {trained.X_train_cluster.shape[0]:,} train rows; "
        "each feature's correlation uses only the rows where that feature "
        "is non-null (pandas' default pairwise behavior) -- n_valid shows "
        "how many that was per feature."
    )
    lines.append("")
    lines.append("| feature | correlation with isFraud | n_valid |")
    lines.append("|---|---:|---:|")
    flagged = []
    for col, c, n_valid in rows:
        flag = ""
        if pd.notna(c) and abs(c) > 0.5:
            flag = " **(>0.5)**"
            flagged.append((col, c))
        c_str = f"{c:.4f}" if pd.notna(c) else "NaN"
        lines.append(f"| {col} | {c_str}{flag} | {n_valid:,} |")
    lines.append("")

    if flagged:
        joined = ", ".join(f"{c} ({v:.3f})" for c, v in flagged)
        lines.append(
            f"**{len(flagged)} feature(s) exceed the 0.5 red-flag "
            f"threshold: {joined}.** Investigated in section 2."
        )
    else:
        lines.append("No feature exceeds the 0.5 red-flag threshold.")
    lines.append("")
    return lines


def _trace_prior_fraud_leak(pipeline_data: run_pipeline.PipelineData) -> list[str]:
    lines = [
        "### 2. Tracing cluster_prior_fraud_share for a leak",
        "",
        "cluster_prior_fraud_share is computed inside "
        "graph.compute_cluster_features as: for every transaction with "
        "`TransactionDT < as_of`, take the per-uid max of isFraud over "
        "*that uid's own* qualifying transactions "
        "(`by_uid[\"isFraud\"].max()`), then average that per-uid flag "
        "across the uids in each cluster. The `as_of` filter is applied to "
        "`transactions` before any of this runs -- there is no code path in "
        "compute_cluster_features that reads isFraud from a row that didn't "
        "pass the `TransactionDT < as_of` filter. That's the argument from "
        "reading the code (and it's what tests/test_graph.py's explicit "
        "leakage test already checks structurally). Below is the argument "
        "from the actual data instead.",
        "",
    ]

    df = pipeline_data.df
    entity_ids = pipeline_data.entity_ids
    as_of = pipeline_data.as_of
    full = df.set_index("TransactionID")
    full = full.assign(uid=entity_ids)
    valid = full.loc[full["uid"].notna()].copy()

    components = get_connected_components(pipeline_data.entity_graph.graph)
    cluster_of = {}
    for i, comp in enumerate(components):
        for u in comp:
            cluster_of[u] = i

    # Only rows whose uid is an actual graph node matter for this check.
    valid["cluster_id"] = valid["uid"].map(cluster_of)
    valid = valid.loc[valid["cluster_id"].notna()]

    uid_to_cluster = valid.drop_duplicates("uid").set_index("uid")["cluster_id"]

    # "expected": recomputed independently, straight from raw rows
    # restricted to TransactionDT < as_of, bypassing graph.py entirely --
    # this is what compute_cluster_features's own logic should produce.
    pre = valid[valid["TransactionDT"] < as_of]
    ever_fraud_pre = pre.groupby("uid")["isFraud"].max()
    expected_share = ever_fraud_pre.groupby(uid_to_cluster.reindex(ever_fraud_pre.index)).mean()

    # "leaked": what it would be if test-period rows were (wrongly) included.
    ever_fraud_all = valid.groupby("uid")["isFraud"].max()
    leaked_share = ever_fraud_all.groupby(uid_to_cluster.reindex(ever_fraud_all.index)).mean()

    # Global check across every cluster the pipeline actually produced a
    # value for, not just one hand-picked example.
    reported_by_uid = pipeline_data.cluster_features["cluster_prior_fraud_share"]
    reported_by_cluster = reported_by_uid.groupby(uid_to_cluster.reindex(reported_by_uid.index)).first()
    common_idx = reported_by_cluster.index.intersection(expected_share.index)
    mismatches = (
        (reported_by_cluster.loc[common_idx] - expected_share.loc[common_idx]).abs() > 1e-9
    )
    n_checked = len(common_idx)
    n_mismatched = int(mismatches.sum())

    lines.append(
        f"Checked all {n_checked:,} clusters that have both a reported "
        "cluster_prior_fraud_share and an independently-recomputable "
        "pre-as_of value: comparing the pipeline's reported value against "
        "one computed straight from raw rows with `TransactionDT < as_of`, "
        f"bypassing graph.py entirely. Mismatches: **{n_mismatched}**."
    )
    lines.append("")

    # Concrete, non-vacuous example: the cluster where leaking test-period
    # data would have changed the number the most -- if even this cluster's
    # reported value matches the pre-as_of-only recomputation, the filter
    # is doing real work, not passing by coincidence.
    gap = (leaked_share - expected_share).abs().dropna()
    gap = gap[gap > 1e-9]
    if gap.empty:
        lines.append(
            "No cluster's leaked-vs-expected prior-fraud share actually "
            "differed on this run, so no single example could demonstrate "
            "the filter doing real work -- reported honestly rather than "
            "picking a less meaningful example. The all-cluster mismatch "
            "count above is the check that matters here."
        )
        lines.append("")
        return lines

    best_cluster_id = gap.idxmax()
    members = sorted(components[int(best_cluster_id)])
    member_rows = valid[valid["cluster_id"] == best_cluster_id]

    reported = float(reported_by_cluster.loc[best_cluster_id])
    expected_value = float(expected_share.loc[best_cluster_id])
    leaked_value = float(leaked_share.loc[best_cluster_id])

    lines.append(
        f"Concrete example -- the cluster where leaking would change this "
        f"feature the most: cluster #{int(best_cluster_id)}, "
        f"{len(members)} member uids."
    )
    lines.append("")
    fraud_rows = member_rows[member_rows["isFraud"] == 1].sort_values(
        ["uid", "TransactionDT"]
    )
    lines.append(
        f"Its {len(fraud_rows)} fraud-labeled transaction(s) (all other "
        "member rows are isFraud=0, omitted for brevity):"
    )
    lines.append("")
    lines.append("| uid | TransactionDT | period | isFraud |")
    lines.append("|---|---:|---|---:|")
    for _, row in fraud_rows.iterrows():
        period = "test" if row["TransactionDT"] >= as_of else "train"
        lines.append(
            f"| {row['uid']} | {row['TransactionDT']:,.0f} | {period} | 1 |"
        )
    lines.append("")

    lines.append(f"- Reported by the pipeline (as_of-filtered): **{reported:.4f}**")
    lines.append(
        f"- Independently recomputed from raw rows with "
        f"`TransactionDT < as_of`, bypassing graph.py entirely: "
        f"**{expected_value:.4f}**"
    )
    lines.append(
        f"- What it would be if test-period rows leaked in (all-time max "
        f"isFraud per member, no as_of filter): **{leaked_value:.4f}**"
    )
    lines.append("")

    if abs(reported - expected_value) < 1e-9:
        lines.append(
            "**Reported value matches the independently-recomputed "
            "pre-as_of-only value, and differs from the would-leak value "
            "for this cluster -- confirmed on real data, not just in the "
            "unit test: no test-period fraud label influenced this "
            "cluster's prior-fraud feature.**"
        )
    else:
        lines.append(
            "**MISMATCH: the reported value does not match the "
            "independently-recomputed pre-as_of value. This is a leak and "
            "is being reported plainly, not fixed quietly.**"
        )
    lines.append("")
    return lines


def _reablation_without_prior_fraud(
    pipeline_data: run_pipeline.PipelineData,
    trained: run_pipeline.TrainedModels,
) -> tuple[list[str], dict[str, float]]:
    lines = [
        "### 3. Ablation re-run with cluster_prior_fraud_share removed",
        "",
    ]

    drop_col = "cluster_prior_fraud_share"
    X_train = trained.X_train_cluster.drop(columns=[drop_col])
    X_test = trained.X_test_cluster.drop(columns=[drop_col])

    trimmed_model = model.train_cluster_model(X_train, trained.y_train)
    trimmed_metrics = evaluate.evaluate_model(
        trimmed_model,
        X_test,
        trained.y_test,
        threshold=run_pipeline.DEFAULT_THRESHOLD,
        cost_fn=run_pipeline.COST_FN,
        cost_fp=run_pipeline.COST_FP,
    )

    lines.append(
        "Same training path (train_cluster_model, identical hyperparameters "
        "and seed), same cluster feature set minus this one column."
    )
    lines.append("")
    lines.append("| model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| cluster (no cluster_prior_fraud_share) | "
        f"{trimmed_metrics['pr_auc']:.4f} | "
        f"{trimmed_metrics['recall_at_1pct_fpr']:.4f} | "
        f"{trimmed_metrics['cost_per_10k']:.2f} |"
    )
    lines.append("")
    return lines, trimmed_metrics


def _cluster_assignment_independence(pipeline_data: run_pipeline.PipelineData) -> list[str]:
    lines = [
        "### 4. Cluster assignment for a test-period transaction never "
        "depends on test-period edges",
        "",
        "Structurally: `graph.build_entity_graph` is called in "
        "run_pipeline.load_and_prepare with `train_df` only -- `test_df` is "
        "never passed to it, so no test-period transaction can ever "
        "contribute an edge or a node. Verified concretely below rather "
        "than just re-reading the call site.",
        "",
    ]

    df = pipeline_data.df
    entity_ids = pipeline_data.entity_ids
    as_of = pipeline_data.as_of
    full = df.set_index("TransactionID")
    full = full.assign(uid=entity_ids)
    valid = full.loc[full["uid"].notna()]

    train_uids = set(valid.loc[valid["TransactionDT"] < as_of, "uid"].unique())
    test_only_uids = set(
        valid.loc[valid["TransactionDT"] >= as_of, "uid"].unique()
    ) - train_uids

    graph_nodes = set(pipeline_data.entity_graph.graph.nodes())
    leaked_nodes = graph_nodes - train_uids
    lines.append(
        f"- Every node in the graph corresponds to a uid with at least one "
        f"train-period transaction: {len(leaked_nodes)} nodes found in the "
        "graph with zero train-period transactions (should be 0)."
    )

    if test_only_uids:
        example_uid = sorted(test_only_uids)[0]
        in_graph = example_uid in graph_nodes
        broadcast = run_pipeline.broadcast_cluster_features(
            entity_ids, pipeline_data.cluster_features
        )
        example_txn_ids = entity_ids.index[entity_ids == example_uid]
        example_features = broadcast.loc[broadcast.index.isin(example_txn_ids)]
        all_null = bool(example_features.isna().all().all())
        node_status = "absent from" if not in_graph else "PRESENT IN"
        null_status = "all null" if all_null else "NOT all null"
        if not in_graph and all_null:
            verdict = "as expected -- no train-period history means no cluster signal, not a fabricated one"
        else:
            verdict = "UNEXPECTED"
        lines.append(
            f"- Concrete example: uid `{example_uid}` appears ONLY in the "
            f"test period (no train-period transactions). It is "
            f"{node_status} the graph's node set, and its broadcast "
            f"cluster features are {null_status} ({verdict})."
        )
    else:
        lines.append(
            "- No test-period-only uid found in this run to use as a "
            "concrete example (every uid observed in the test period also "
            "had train-period history)."
        )
    lines.append("")
    return lines


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    lines = ["## Sanity checks", ""]
    lines += _correlation_section(trained)
    lines += _trace_prior_fraud_leak(pipeline_data)
    reablation_lines, _ = _reablation_without_prior_fraud(pipeline_data, trained)
    lines += reablation_lines
    lines += _cluster_assignment_independence(pipeline_data)

    ablation_path = RESULTS_DIR / "ablation.md"
    existing = ablation_path.read_text(encoding="utf-8")
    ablation_path.write_text(
        existing.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Appended 'Sanity checks' section to results/ablation.md")


if __name__ == "__main__":
    main()

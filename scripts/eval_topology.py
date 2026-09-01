"""Task 2: does cluster SHAPE (not just size/rate aggregates) add anything
to the ablation, once the dominant confound is removed?

Motivation, from the task that produced this script: results/stability.md
found that once `cluster_prior_fraud_share` is removed, the remaining
graph-structure aggregates (size, density, velocity, amount CV, burst
concentration) show no reliable lift across rolling temporal splits (mean
+0.0060, spread 0.0230, sign flips). That is this project's central
weakness -- the graph-structure hypothesis is currently unsupported. This
re-tests it once, with two richer topology features
(graph.compute_cluster_features's new `k_core_number` and `star_ratio`,
see Task 1 / src/graph.py) instead of only aggregates, on the SAME single
80% split as results/ablation.md.

Four rows, in results/ablation_topology.md (a NEW file -- results/ablation.md
is not touched or overwritten):
  1. baseline (txn features only)
  2. + original cluster features
  3. + original cluster features, minus cluster_prior_fraud_share
  4. + original + topology features, minus cluster_prior_fraud_share

Row 4 is the one that matters: it isolates whether the two topology
features add anything the aggregates didn't, with the dominant, partly-
circular confound (cluster_prior_fraud_share) removed from both rows 3 and
4 equally. Rows 1-2 are read straight off run_pipeline.load_and_prepare()/
train_both_models() -- the identical, already-verified pipeline call that
produces results/ablation.md's own numbers, not retrained here. Rows 3-4
train fresh models via model.train_cluster_model (frozen, unmodified) on
modified feature sets; every reported metric comes from evaluate.py's
frozen pr_auc/recall_at_fpr/cost_per_10k, exactly as results/ablation.md's
own numbers do.

No frozen files modified; graph.compute_cluster_features's new
include_topology=True parameter (additive-only, see Task 1) is the only
new capability this script relies on.

Usage:
    python scripts/eval_topology.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import evaluate, graph, model, run_pipeline

RESULTS_DIR = REPO_ROOT / "results"
DROP_COL = "cluster_prior_fraud_share"
COST_FN = run_pipeline.COST_FN
COST_FP = run_pipeline.COST_FP
THRESHOLD = run_pipeline.DEFAULT_THRESHOLD

TOPOLOGY_COLUMNS = ("k_core_number", "star_ratio")


def _feature_importance_table(booster: lgb.Booster) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)


def main() -> None:
    print("[eval_topology] loading pipeline data (rows 1-2 reuse the existing, verified pipeline)...")
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    rows: list[dict] = []

    baseline_metrics = evaluate.evaluate_model(
        trained.baseline_model, trained.X_test_baseline, trained.y_test,
        threshold=THRESHOLD, cost_fn=COST_FN, cost_fp=COST_FP,
    )
    rows.append({"row": 1, "label": "baseline (txn features only)", **baseline_metrics})
    print(f"[eval_topology] row 1 (baseline): {baseline_metrics}")

    cluster_metrics = evaluate.evaluate_model(
        trained.cluster_model, trained.X_test_cluster, trained.y_test,
        threshold=THRESHOLD, cost_fn=COST_FN, cost_fp=COST_FP,
    )
    rows.append({"row": 2, "label": "+ original cluster features", **cluster_metrics})
    print(f"[eval_topology] row 2 (+ original cluster features): {cluster_metrics}")

    print("[eval_topology] row 3: training on original cluster features minus cluster_prior_fraud_share...")
    X_train_trimmed = trained.X_train_cluster.drop(columns=[DROP_COL])
    X_test_trimmed = trained.X_test_cluster.drop(columns=[DROP_COL])
    trimmed_model = model.train_cluster_model(X_train_trimmed, trained.y_train)
    trimmed_metrics = evaluate.evaluate_model(
        trimmed_model, X_test_trimmed, trained.y_test,
        threshold=THRESHOLD, cost_fn=COST_FN, cost_fp=COST_FP,
    )
    rows.append(
        {"row": 3, "label": "+ original cluster features, minus cluster_prior_fraud_share", **trimmed_metrics}
    )
    print(f"[eval_topology] row 3: {trimmed_metrics}")

    print("[eval_topology] computing topology features (include_topology=True, additive-only)...")
    cluster_features_topology = graph.compute_cluster_features(
        pipeline_data.train_df,
        pipeline_data.entity_ids,
        pipeline_data.entity_graph.graph,
        as_of=pipeline_data.as_of,
        include_topology=True,
    )
    for col in TOPOLOGY_COLUMNS:
        assert col in cluster_features_topology.columns, f"{col} missing from topology output"

    cluster_by_txn_topology = run_pipeline.broadcast_cluster_features(
        pipeline_data.entity_ids, cluster_features_topology
    )
    cluster_train_topology = cluster_by_txn_topology.reindex(trained.y_train.index)
    cluster_test_topology = cluster_by_txn_topology.reindex(trained.y_test.index)

    X_train_topology = model.build_feature_matrix(pipeline_data.train_df, cluster_train_topology)
    X_test_topology = model.build_feature_matrix(pipeline_data.test_df, cluster_test_topology)
    X_train_topology_trimmed = X_train_topology.drop(columns=[DROP_COL])
    X_test_topology_trimmed = X_test_topology.drop(columns=[DROP_COL])

    print("[eval_topology] row 4: training on original + topology features, minus cluster_prior_fraud_share...")
    topology_model = model.train_cluster_model(X_train_topology_trimmed, trained.y_train)
    topology_metrics = evaluate.evaluate_model(
        topology_model, X_test_topology_trimmed, trained.y_test,
        threshold=THRESHOLD, cost_fn=COST_FN, cost_fp=COST_FP,
    )
    rows.append(
        {
            "row": 4,
            "label": "+ original + topology features, minus cluster_prior_fraud_share",
            **topology_metrics,
        }
    )
    print(f"[eval_topology] row 4 (the row that matters): {topology_metrics}")

    importance_table = _feature_importance_table(topology_model)
    write_report(rows, importance_table, X_train_topology_trimmed.shape[1])
    print("[eval_topology] wrote results/ablation_topology.md")


def write_report(rows: list[dict], importance_table: pd.DataFrame, n_features_row4: int) -> None:
    lines: list[str] = []
    lines.append("# Topology features ablation: does cluster SHAPE add anything?")
    lines.append("")
    lines.append(
        "results/stability.md found that once `cluster_prior_fraud_share` is "
        "removed, the remaining graph-structure aggregates (cluster size, "
        "edge density, velocity, amount CV, burst concentration) show no "
        "reliable lift across rolling temporal splits: mean +0.0060, spread "
        "0.0230, sign flips between splits. That is this project's central "
        "weakness -- the graph-structure hypothesis is currently "
        "unsupported. This tests it once more, on the same single 80% "
        "split as results/ablation.md, with two richer topology features "
        "instead of only aggregates: `k_core_number` (this uid's k-core "
        "index -- how deeply embedded it is in a dense subgraph) and "
        "`star_ratio` (max node degree in the cluster / cluster size -- "
        "the hub-and-spoke device-farm signature, as distinct from a "
        "mutually-connected clique). Both are additive-only extensions to "
        "graph.compute_cluster_features (see Task 1); nothing about the "
        "existing 10 features or their computation changed."
    )
    lines.append("")
    lines.append(
        "Rows 1-2 are read directly from `run_pipeline.load_and_prepare()` / "
        "`train_both_models()` -- the identical call that produces "
        "results/ablation.md's own numbers, not retrained here. Rows 3-4 "
        "train fresh models via `model.train_cluster_model` (frozen, "
        "unmodified) on modified feature sets; every metric below comes "
        "from `evaluate.py`'s frozen `pr_auc`/`recall_at_fpr`/`cost_per_10k`, "
        "exactly as results/ablation.md's own numbers do."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| row | model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |")
    lines.append("|---:|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['row']} | {r['label']} | {r['pr_auc']:.4f} | "
            f"{r['recall_at_1pct_fpr']:.4f} | {r['cost_per_10k']:.2f} |"
        )
    lines.append("")

    row3 = next(r for r in rows if r["row"] == 3)
    row4 = next(r for r in rows if r["row"] == 4)
    pr_auc_delta = row4["pr_auc"] - row3["pr_auc"]
    recall_delta = row4["recall_at_1pct_fpr"] - row3["recall_at_1pct_fpr"]
    cost_delta = row4["cost_per_10k"] - row3["cost_per_10k"]
    lines.append(
        f"**Row 4 vs. row 3 (topology added, dominant confound removed from "
        f"both): PR-AUC {pr_auc_delta:+.4f}, recall@1%FPR {recall_delta:+.4f}, "
        f"cost per 10k {cost_delta:+.2f}** (negative is better for cost). "
        "This single-split delta is the direct answer to \"does topology add "
        "anything the aggregates didn't\" -- reported as-is; see "
        "results/stability_topology.md (Task 3) for whether this delta, "
        "positive or negative, holds up across rolling splits the way "
        "results/stability.md found the aggregate-only version did not."
    )
    lines.append("")

    lines.append("## Feature importances, row 4's model (top 20 by gain)")
    lines.append("")
    lines.append("| feature | gain |")
    lines.append("|---|---:|")
    for _, r in importance_table.head(20).iterrows():
        lines.append(f"| {r['feature']} | {r['gain']:,.1f} |")
    lines.append("")

    lines.append("## Where the two new topology features land, specifically")
    lines.append("")
    lines.append(
        f"Row 4's model has {n_features_row4} total features. Reported "
        "explicitly regardless of top-20 rank, since a low-importance "
        "feature would otherwise be invisible above:"
    )
    lines.append("")
    for col in TOPOLOGY_COLUMNS:
        rank = int(importance_table.index[importance_table["feature"] == col][0]) + 1
        gain = float(importance_table.loc[importance_table["feature"] == col, "gain"].iloc[0])
        lines.append(
            f"- `{col}`: gain **{gain:,.1f}**, rank **{rank}** of "
            f"{len(importance_table)} total features."
        )
    lines.append("")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ablation_topology.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

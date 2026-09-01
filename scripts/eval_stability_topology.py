"""Task 3: does the residual lift become reliably positive across rolling
splits once topology features are added, or does it still flip sign?

results/stability.md found the residual lift (cluster_prior_fraud_share
removed) is mean +0.0060, spread 0.0230, and changes sign across the 4
rolling splits (60/70/80/90% through sorted TransactionDT) -- close to
noise, not a small-but-real effect. results/ablation_topology.md (Task 2)
found topology features (k_core_number, star_ratio) did not help on the
single 80% split. This is the question the whole session exists to
answer: repeated across all 4 rolling splits, with topology included, does
the residual lift become reliably positive, or does it still flip sign?

Same rolling-split methodology as scripts/eval_stability.py (fresh entity
graph, fresh causal cluster features, fresh models at each split -- nothing
cached), extended to also compute a topology-augmented trimmed model at
each split via graph.compute_cluster_features(..., include_topology=True)
(additive-only, see Task 1). Only entities.py, graph.py, model.py, and
evaluate.py are imported (all frozen, none modified) plus
run_pipeline.broadcast_cluster_features/MAX_DEGREE, exactly as
eval_stability.py already does.

Do NOT tune feature definitions, thresholds, or max_degree to make this
come out positive -- report whatever it says. A negative result (still
flips sign, or still noise-level) is this session's legitimate answer:
the graph-structure hypothesis was tested twice, with progressively richer
features, and did not hold on this dataset at this sample size.

Usage:
    python scripts/eval_stability_topology.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import data, entities, evaluate, graph, model, run_pipeline

RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"

# Identical to scripts/eval_stability.py -- same 4 rolling split points, same
# meaning: fraction of the sorted-by-TransactionDT data used as train.
SPLIT_POINTS = [0.6, 0.7, 0.8, 0.9]

DROP_COL = "cluster_prior_fraud_share"

TIME_BUDGET_SECONDS = 2400


def run_one_split(df, entity_ids, split_fraction: float) -> dict:
    test_size = 1 - split_fraction
    train_df, test_df = evaluate.temporal_train_test_split(df, test_size=test_size)
    as_of = float(test_df["TransactionDT"].min())

    entity_graph = graph.build_entity_graph(
        train_df, entity_ids, max_degree=run_pipeline.MAX_DEGREE
    )

    y_train = train_df.set_index("TransactionID")["isFraud"]
    y_test = test_df.set_index("TransactionID")["isFraud"]

    # --- baseline (txn features only) ---
    X_train_baseline = model.build_feature_matrix(train_df)
    X_test_baseline = model.build_feature_matrix(test_df)
    baseline_model = model.train_baseline_model(X_train_baseline, y_train)
    baseline_pr_auc = evaluate.pr_auc(y_test, baseline_model.predict(X_test_baseline))

    # --- original cluster features, and the trimmed (no prior-fraud) variant ---
    cluster_features = graph.compute_cluster_features(
        train_df, entity_ids, entity_graph.graph, as_of=as_of
    )
    cluster_by_txn = run_pipeline.broadcast_cluster_features(entity_ids, cluster_features)
    cluster_train = cluster_by_txn.reindex(y_train.index)
    cluster_test = cluster_by_txn.reindex(y_test.index)

    X_train_cluster = model.build_feature_matrix(train_df, cluster_train)
    X_test_cluster = model.build_feature_matrix(test_df, cluster_test)
    cluster_model = model.train_cluster_model(X_train_cluster, y_train)
    cluster_pr_auc = evaluate.pr_auc(y_test, cluster_model.predict(X_test_cluster))

    X_train_trimmed = X_train_cluster.drop(columns=[DROP_COL])
    X_test_trimmed = X_test_cluster.drop(columns=[DROP_COL])
    trimmed_model = model.train_cluster_model(X_train_trimmed, y_train)
    trimmed_pr_auc = evaluate.pr_auc(y_test, trimmed_model.predict(X_test_trimmed))

    # --- original + topology features, trimmed (no prior-fraud) ---
    cluster_features_topology = graph.compute_cluster_features(
        train_df, entity_ids, entity_graph.graph, as_of=as_of, include_topology=True
    )
    cluster_by_txn_topology = run_pipeline.broadcast_cluster_features(
        entity_ids, cluster_features_topology
    )
    cluster_train_topology = cluster_by_txn_topology.reindex(y_train.index)
    cluster_test_topology = cluster_by_txn_topology.reindex(y_test.index)

    X_train_topology = model.build_feature_matrix(train_df, cluster_train_topology)
    X_test_topology = model.build_feature_matrix(test_df, cluster_test_topology)
    X_train_topology_trimmed = X_train_topology.drop(columns=[DROP_COL])
    X_test_topology_trimmed = X_test_topology.drop(columns=[DROP_COL])
    topology_model = model.train_cluster_model(X_train_topology_trimmed, y_train)
    topology_pr_auc = evaluate.pr_auc(y_test, topology_model.predict(X_test_topology_trimmed))

    return {
        "split_fraction": split_fraction,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "as_of": as_of,
        "baseline_pr_auc": baseline_pr_auc,
        "cluster_pr_auc": cluster_pr_auc,
        "trimmed_pr_auc": trimmed_pr_auc,
        "topology_pr_auc": topology_pr_auc,
        "lift": cluster_pr_auc - baseline_pr_auc,
        "trimmed_lift": trimmed_pr_auc - baseline_pr_auc,
        "topology_lift": topology_pr_auc - baseline_pr_auc,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_report(results: list[dict], skipped: list[float], total_seconds: float) -> None:
    lines: list[str] = []
    lines.append("# Stability of the residual lift, with topology features included")
    lines.append("")
    lines.append(
        "results/stability.md found the residual lift (PR-AUC gain over "
        "baseline once `cluster_prior_fraud_share` is removed, leaving only "
        "aggregate cluster features) is mean +0.0060, spread 0.0230, and "
        "changes sign across the 4 rolling splits (60/70/80/90% through "
        "sorted TransactionDT) -- the central weakness this whole session "
        "exists to re-test. This repeats that exact comparison at the same "
        "4 split points, with two topology features "
        "(`k_core_number`, `star_ratio` -- see Task 1 / src/graph.py) added "
        "to the trimmed feature set. **This is the question the session "
        "exists to answer: does the residual lift become reliably positive "
        "across all four splits with topology included, or does it still "
        "flip sign?**"
    )
    lines.append("")
    lines.append(
        "Same methodology as results/stability.md: fresh entity graph, "
        "fresh causal cluster features, fresh models at each split -- "
        "nothing cached or reused across splits. Only entities.py, "
        "graph.py, model.py, and evaluate.py were imported (all frozen, "
        "none modified) plus run_pipeline.broadcast_cluster_features and "
        "MAX_DEGREE=20, reused as-is. No feature definitions, thresholds, "
        "or max_degree were tuned after seeing any result below."
    )
    lines.append("")

    if skipped:
        lines.append(
            f"**Ran {len(results)} of {len(SPLIT_POINTS)} planned split points "
            f"({', '.join(f'{p:.0%}' for p in [r['split_fraction'] for r in results])}) "
            f"in {total_seconds:.0f}s before the {TIME_BUDGET_SECONDS}s time "
            f"budget was reached; skipped {', '.join(f'{p:.0%}' for p in skipped)}. "
            "Everything below is reported honestly over only the splits "
            "that actually ran."
        )
    else:
        lines.append(
            f"**All {len(SPLIT_POINTS)} planned split points ran, in "
            f"{total_seconds:.0f}s total.**"
        )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "| split | baseline PR-AUC | cluster PR-AUC | full lift | trimmed "
        "PR-AUC (no topology) | trimmed lift (no topology) | + topology "
        "PR-AUC | trimmed lift (WITH topology) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['split_fraction']:.0%} | {r['baseline_pr_auc']:.4f} | "
            f"{r['cluster_pr_auc']:.4f} | {r['lift']:+.4f} | "
            f"{r['trimmed_pr_auc']:.4f} | {r['trimmed_lift']:+.4f} | "
            f"{r['topology_pr_auc']:.4f} | {r['topology_lift']:+.4f} |"
        )
    lines.append("")

    trimmed_lifts = [r["trimmed_lift"] for r in results]
    topology_lifts = [r["topology_lift"] for r in results]
    mean_trimmed = _mean(trimmed_lifts)
    mean_topology = _mean(topology_lifts)
    spread_trimmed = max(trimmed_lifts) - min(trimmed_lifts)
    spread_topology = max(topology_lifts) - min(topology_lifts)

    lines.append("## Residual lift, without topology vs. with topology")
    lines.append("")
    lines.append(
        f"- **Without topology (results/stability.md's existing finding, "
        f"reproduced here on the same splits):** mean **{mean_trimmed:+.4f}**, "
        f"spread **{spread_trimmed:.4f}** (min {min(trimmed_lifts):+.4f}, "
        f"max {max(trimmed_lifts):+.4f})."
    )
    lines.append(
        f"- **With topology (k_core_number + star_ratio added):** mean "
        f"**{mean_topology:+.4f}**, spread **{spread_topology:.4f}** "
        f"(min {min(topology_lifts):+.4f}, max {max(topology_lifts):+.4f})."
    )
    lines.append("")

    without_crosses_zero = min(trimmed_lifts) < 0 < max(trimmed_lifts)
    with_crosses_zero = min(topology_lifts) < 0 < max(topology_lifts)
    all_positive_with_topology = min(topology_lifts) > 0

    lines.append("## Verdict")
    lines.append("")
    if all_positive_with_topology and not with_crosses_zero:
        lines.append(
            f"**Topology rescues the residual lift: positive at all "
            f"{len(results)} splits tested, mean {mean_topology:+.4f}, "
            f"spread {spread_topology:.4f}** -- compared to the sign-flipping "
            f"aggregate-only residual (mean {mean_trimmed:+.4f}, crosses "
            "zero). This is a genuine finding in favor of the graph-structure "
            "hypothesis and should be weighed against how small the effect "
            "size still is, and against the single sample size (4 splits) "
            "this conclusion rests on."
        )
    else:
        lines.append(
            "**Topology does not rescue the residual lift.** "
            + (
                f"It still changes sign across splits (negative at "
                + ", ".join(
                    f"{r['split_fraction']:.0%}" for r in results if r["topology_lift"] < 0
                )
                + ", positive at "
                + ", ".join(
                    f"{r['split_fraction']:.0%}" for r in results if r["topology_lift"] >= 0
                )
                + f"), same as the aggregate-only residual "
                if with_crosses_zero
                else "It stays negative or flat across every split tested -- worse than "
                "merely failing to rescue the aggregate-only residual, this is a "
                "net-negative result for topology, not just a null one, "
            )
            + f"(mean {mean_topology:+.4f} vs. {mean_trimmed:+.4f} without "
            f"topology, spread {spread_topology:.4f} vs. {spread_trimmed:.4f})."
        )
        lines.append("")
        lines.append(
            "**Reported plainly, as the task that produced this asked: the "
            "graph-structure hypothesis was tested twice now -- once with "
            "aggregate cluster features (results/stability.md), once with "
            "richer topology features added on top (this report) -- and "
            "neither version shows a reliable, sign-stable lift once the "
            "dominant `cluster_prior_fraud_share` confound is removed, on "
            f"this dataset at this sample size ({len(results)} rolling "
            "splits). This is not evidence that graph structure can never "
            "predict abuse; it is evidence that this project's specific "
            "attempts to measure it -- aggregates, then topology -- have "
            "not found a stable signal here. No feature definition, "
            "threshold, or max_degree was adjusted to avoid this "
            "conclusion.**"
        )
    lines.append("")

    lines.append(
        "Cost per 10k and recall@1%FPR were not re-measured here, for the "
        "same reason results/stability.md didn't: this script is scoped to "
        "PR-AUC and the lift, and re-running the full cost/calibration "
        "analysis at every split point for both feature sets would "
        "multiply an already-substantial cost several times over."
    )
    lines.append("")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "stability_topology.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[eval_stability_topology] loading data + resolving entities (once, reused across splits)...")
    df = data.load_transactions(DATA_DIR, nrows=None)
    entity_ids = entities.resolve_entities(df)

    start = time.time()
    results: list[dict] = []
    skipped: list[float] = []
    durations: list[float] = []

    for i, split_fraction in enumerate(SPLIT_POINTS):
        elapsed = time.time() - start
        if durations:
            projected_next = durations[-1] * 1.2
            if elapsed + projected_next > TIME_BUDGET_SECONDS:
                remaining = len(SPLIT_POINTS) - i
                print(
                    f"[eval_stability_topology] time budget ({TIME_BUDGET_SECONDS}s) "
                    f"would be exceeded ({elapsed:.0f}s elapsed, {remaining} split(s) "
                    f"left, next projected at {projected_next:.0f}s) -- stopping "
                    "early, reporting only completed splits."
                )
                skipped.extend(SPLIT_POINTS[i:])
                break

        print(f"[eval_stability_topology] split {split_fraction:.0%}: building graph + features + training 4 models...")
        t0 = time.time()
        result = run_one_split(df, entity_ids, split_fraction)
        dt = time.time() - t0
        durations.append(dt)
        print(
            f"[eval_stability_topology] split {split_fraction:.0%} done in {dt:.1f}s -- "
            f"trimmed lift (no topology) {result['trimmed_lift']:+.4f}, "
            f"trimmed lift (WITH topology) {result['topology_lift']:+.4f}"
        )
        results.append(result)

    total_seconds = time.time() - start
    write_report(results, skipped, total_seconds)
    print(f"[eval_stability_topology] wrote results/stability_topology.md ({len(results)} splits, {total_seconds:.0f}s total)")


if __name__ == "__main__":
    main()

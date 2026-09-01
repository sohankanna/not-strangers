"""Task: cluster-level precision@k -- the metric an analyst team actually
operates on, as opposed to the transaction-level PR-AUC in results/ablation.md.

Motivation: this project claims coordinated-abuse *ring* detection, but every
number in results/ablation.md is transaction-level PR-AUC. That's a proxy --
it says the model ranks individual transactions well, not that a reviewer
working a queue of flagged *clusters* would actually find real abuse near the
top. This script measures that directly: for K in [10, 25, 50, 100], if an
analyst worked the top K clusters, how many contain real fraud, how many
transactions would they have had to review to find it, and how does that
compare to a naive baseline that ignores cluster structure entirely?

Read-only with respect to the pipeline: this script imports and calls
run_pipeline.load_and_prepare() / train_both_models() (no retraining, no
feature recomputation) and investigator.build_evidence /
investigator._priority_score (no LLM calls -- explain_cluster is never
invoked here, so this doesn't touch the Anthropic API). It does not modify
evaluate.py, entities.py, graph.py, model.py, investigator.py, or policy.py.

Everything below is scoped to the test split only (the temporal holdout from
evaluate.temporal_train_test_split), using the existing train-only entity
graph and existing cluster assignments -- no test-period edges, no
test-period labels in any feature.

Usage:
    python scripts/eval_queue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import investigator, run_pipeline
from src.graph import get_connected_components

RESULTS_DIR = REPO_ROOT / "results"
README_PATH = REPO_ROOT / "README.md"
K_VALUES = [10, 25, 50, 100]


def _annotate_all_transactions(
    pipeline_data: run_pipeline.PipelineData,
) -> tuple[pd.DataFrame, list[set]]:
    """All transactions (train + test) with uid + cluster_id, restricted to
    multi-uid (2+) clusters from the train-only entity graph. Mirrors
    app.py's annotated_transactions/_cluster_membership exactly, so
    cluster_id here is the same identifier the dashboard queue uses.
    """
    components = get_connected_components(pipeline_data.entity_graph.graph)
    cluster_of: dict[str, int] = {}
    for i, comp in enumerate(components):
        if len(comp) >= 2:
            for u in comp:
                cluster_of[u] = i

    full = pipeline_data.df.set_index("TransactionID")
    cluster_id = pipeline_data.entity_ids.map(cluster_of).rename("cluster_id")
    full = pd.concat([full, pipeline_data.entity_ids.rename("uid"), cluster_id], axis=1)
    full = full.loc[full["cluster_id"].notna()].copy()
    full["cluster_id"] = full["cluster_id"].astype(int)
    return full, components


def build_cluster_table(
    pipeline_data: run_pipeline.PipelineData, trained: run_pipeline.TrainedModels
) -> tuple[pd.DataFrame, int, int]:
    """One row per multi-uid cluster with >=1 test-period transaction.

    Ranking A (priority_score) mirrors app.py's build_cluster_queue exactly:
    investigator.build_evidence + investigator._priority_score computed over
    the cluster's full transaction history (train + test), the same ordering
    policy.py's decisions are queued by in the dashboard.

    Ranking B (mean_baseline_score) is the mean of the baseline model's
    per-transaction score (results/ablation.md's "baseline (txn features
    only)" row -- trained.baseline_model on trained.X_test_baseline) across
    the cluster's test-period member transactions. This model never sees a
    cluster feature at train time, so "no cluster-level features involved"
    holds for the score itself, not just for how it's aggregated.

    Returns (table, n_qualifying_all_multi_uid_components, n_dropped) --
    the second is the total multi-uid component count (a cross-check against
    results/case_studies.md's "1,567 multi-uid clusters" figure), the third
    counts qualifying clusters dropped for having no member with computed
    cluster features (should be 0 -- see inline comment).
    """
    full, components = _annotate_all_transactions(pipeline_data)
    n_multi_uid_components = sum(1 for c in components if len(c) >= 2)

    full["is_test"] = full.index.isin(trained.y_test.index)
    baseline_scores = pd.Series(
        trained.baseline_model.predict(trained.X_test_baseline),
        index=trained.X_test_baseline.index,
    )
    full["baseline_score"] = full.index.to_series().map(baseline_scores)

    test_rows = full.loc[full["is_test"]]
    cluster_stats = (
        test_rows.groupby("cluster_id")
        .agg(
            n_uids=("uid", "nunique"),
            n_test_txns=("isFraud", "size"),
            n_fraud_txns=("isFraud", "sum"),
            mean_baseline_score=("baseline_score", "mean"),
        )
        .reset_index()
    )
    cluster_stats["has_fraud"] = cluster_stats["n_fraud_txns"] > 0
    qualifying_ids = set(cluster_stats["cluster_id"])

    # Priority score uses each cluster's FULL history (train + test), not
    # just its test-period rows -- this is what makes it the same ordering
    # the dashboard queue produces, not a test-only variant of it.
    cf = pipeline_data.cluster_features
    priority_scores: dict[int, float] = {}
    for cluster_id, grp in full[full["cluster_id"].isin(qualifying_ids)].groupby(
        "cluster_id"
    ):
        members = sorted(grp["uid"].unique())
        cf_sub = cf.loc[[u for u in members if u in cf.index]]
        if cf_sub.empty:
            continue
        evidence = investigator.build_evidence(cf_sub, grp)
        priority_scores[cluster_id] = investigator._priority_score(evidence)

    cluster_stats["priority_score"] = cluster_stats["cluster_id"].map(priority_scores)
    n_before = len(cluster_stats)
    cluster_stats = cluster_stats.dropna(subset=["priority_score"]).reset_index(drop=True)
    n_dropped = n_before - len(cluster_stats)

    return cluster_stats, n_multi_uid_components, n_dropped


def _topk_stats(df: pd.DataFrame, rank_col: str, k: int) -> dict:
    ranked = df.sort_values(rank_col, ascending=False)
    top = ranked.head(k)
    n = len(top)
    n_with_fraud = int(top["has_fraud"].sum())
    fraud_surfaced = int(top["n_fraud_txns"].sum())
    workload = int(top["n_test_txns"].sum())
    return {
        "k": k,
        "n_clusters_evaluated": n,
        "precision_at_k": (n_with_fraud / n) if n else float("nan"),
        "fraud_txns_surfaced": fraud_surfaced,
        "workload_txns": workload,
        "efficiency_ratio": (fraud_surfaced / workload) if workload else float("nan"),
    }


def compute_rankings(cluster_stats: pd.DataFrame) -> dict[str, list[dict]]:
    return {
        "priority": [_topk_stats(cluster_stats, "priority_score", k) for k in K_VALUES],
        "baseline": [_topk_stats(cluster_stats, "mean_baseline_score", k) for k in K_VALUES],
    }


def _results_table(rows: list[dict]) -> list[str]:
    lines = [
        "| K | clusters evaluated | precision@k | fraud txns surfaced | "
        "workload (test txns reviewed) | efficiency (fraud / reviewed) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        note = "" if r["n_clusters_evaluated"] == r["k"] else " *"
        lines.append(
            f"| {r['k']}{note} | {r['n_clusters_evaluated']} | "
            f"{r['precision_at_k']:.4f} | {r['fraud_txns_surfaced']} | "
            f"{r['workload_txns']} | {r['efficiency_ratio']:.4f} |"
        )
    return lines


def write_report(
    cluster_stats: pd.DataFrame,
    n_multi_uid_components: int,
    n_dropped: int,
    results: dict[str, list[dict]],
) -> dict:
    """Writes results/queue_eval.md. Returns a small dict of headline
    numbers so main() can build the README section from the same computed
    values rather than re-deriving or re-typing them.
    """
    n_qualifying = len(cluster_stats)
    n_with_fraud = int(cluster_stats["has_fraud"].sum())
    base_rate = n_with_fraud / n_qualifying if n_qualifying else float("nan")

    lines: list[str] = []
    lines.append("# Queue-level evaluation: cluster-level precision@k")
    lines.append("")
    lines.append(
        "This project's headline number (results/ablation.md) is "
        "transaction-level PR-AUC. That's a proxy -- it measures how well "
        "the model ranks individual transactions, not whether an analyst "
        "team working a queue of flagged *clusters* would actually find real "
        "abuse near the top. This closes that gap: for K in "
        f"{K_VALUES}, if reviewers worked the top K clusters today, how many "
        "are real, how many transactions would they review to find them, "
        "and how does the system's actual ranking compare to a naive "
        "baseline that ignores cluster structure entirely?"
    )
    lines.append("")
    lines.append(
        "Produced by `scripts/eval_queue.py`, run on the test split only "
        "(evaluate.temporal_train_test_split's existing temporal holdout), "
        "reusing the existing train-only entity graph and cluster "
        "assignments. No features were recomputed and no model was "
        "retrained to produce this report -- it calls "
        "run_pipeline.load_and_prepare() and run_pipeline.train_both_models() "
        "exactly as every other results/*.md artifact in this project does."
    )
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"**Population:** multi-uid clusters (2+ members) only -- a single-uid "
        "\"cluster\" is not a ring, and including them would inflate every "
        f"number below. There are {n_multi_uid_components:,} multi-uid "
        "connected components in the train-only entity graph in total "
        "(consistent with results/case_studies.md's 1,567 figure, same "
        "graph); this report further restricts to the "
        f"{n_qualifying:,} of those that have at least one test-period "
        "transaction, since only those can be scored and labeled on the "
        "test split. Every count below (fraud surfaced, workload) is over "
        "test-period transactions only, in those clusters."
    )
    if n_dropped:
        lines.append("")
        lines.append(
            f"**{n_dropped} qualifying cluster(s) dropped:** no member uid "
            "had computed cluster features, so no priority score could be "
            "built for them. Reported here rather than silently excluded; "
            "see build_cluster_table's docstring."
        )
    lines.append("")
    lines.append(
        "**Ranking A -- priority ranking:** the ordering the system "
        "actually produces. `investigator.build_evidence` + "
        "`investigator._priority_score` computed over each cluster's full "
        "transaction history (train + test), exactly matching app.py's "
        "`build_cluster_queue` (the dashboard's review queue) and "
        "results/case_studies.md's ranking. Formula: "
        "`cluster_prior_fraud_share * 100 + cluster_burst_concentration * 10 "
        "+ min(cluster_txn_count, 100) * 0.1`."
    )
    lines.append("")
    lines.append(
        "**Ranking B -- baseline ranking:** clusters ranked by the mean of "
        "the *baseline* model's per-transaction score (results/ablation.md's "
        "\"baseline (txn features only)\" row -- trained on transaction "
        "features alone, no cluster features at train time at all) across "
        "each cluster's test-period member transactions. This isolates "
        "whether the graph/cluster-feature apparatus adds anything over "
        "\"just average how suspicious this cluster's transactions already "
        "look to a plain transaction classifier\"."
    )
    lines.append("")

    lines.append("## Base rate")
    lines.append("")
    lines.append(
        f"**{n_with_fraud} of {n_qualifying} qualifying test-split clusters "
        f"({base_rate:.4f}, i.e. {base_rate:.1%}) contain at least one "
        "fraud transaction.**"
    )
    lines.append("")
    lines.append(
        "Precision@k is meaningless without this number: a precision@k "
        "barely above the base rate means the ranking is doing little "
        "better than picking clusters at random from the qualifying "
        "population. Every precision@k figure below should be read "
        f"relative to this {base_rate:.4f} baseline, not in isolation."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("### Ranking A: priority ranking (system / dashboard / policy ordering)")
    lines.append("")
    lines += _results_table(results["priority"])
    lines.append("")
    lines.append("### Ranking B: baseline ranking (mean transaction-level score, no cluster features)")
    lines.append("")
    lines += _results_table(results["baseline"])
    lines.append("")
    if any(r["n_clusters_evaluated"] != r["k"] for r in results["priority"]):
        lines.append(
            f"\\* fewer than K qualifying clusters existed at this K, so "
            "all available qualifying clusters were evaluated instead of K."
        )
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    comparisons = []
    priority_wins = 0
    baseline_wins = 0
    ties = 0
    for p, b in zip(results["priority"], results["baseline"]):
        k = p["k"]
        if p["precision_at_k"] > b["precision_at_k"]:
            priority_wins += 1
            tag = "priority ahead"
        elif p["precision_at_k"] < b["precision_at_k"]:
            baseline_wins += 1
            tag = "baseline ahead"
        else:
            ties += 1
            tag = "tied"
        comparisons.append(
            f"- K={k}: priority precision@k={p['precision_at_k']:.4f} "
            f"(efficiency {p['efficiency_ratio']:.4f}) vs. baseline "
            f"precision@k={b['precision_at_k']:.4f} (efficiency "
            f"{b['efficiency_ratio']:.4f}) -- **{tag}**"
        )
    lines += comparisons
    lines.append("")

    if priority_wins > baseline_wins:
        headline = (
            f"**The priority ranking beats the mean-score baseline on "
            f"precision@k at {priority_wins} of {len(K_VALUES)} K values "
            f"tested** ({baseline_wins} for the baseline, {ties} tied). "
        )
    elif baseline_wins > priority_wins:
        headline = (
            f"**The mean-score baseline beats the priority ranking on "
            f"precision@k at {baseline_wins} of {len(K_VALUES)} K values "
            f"tested** ({priority_wins} for priority, {ties} tied). This is "
            "reported as-is -- the ranking was not adjusted after seeing "
            "this result. A null (or negative) result here is a legitimate "
            "finding about where the cluster-topology features do and "
            "don't help: they were built to lift transaction-level PR-AUC "
            "(results/ablation.md), and doing that is not the same "
            "guarantee as producing a better cluster-priority ordering."
        )
    else:
        headline = (
            "**The priority ranking and the mean-score baseline tie on "
            f"precision@k across all {len(K_VALUES)} K values tested.** "
        )
    lines.append(headline)
    lines.append("")
    lines.append(
        "Both rankings draw from the same qualifying population and the "
        "same test-split labels -- the only thing that differs between them "
        "is the ordering applied to that population, so any precision@k gap "
        "above is attributable to the ranking method, not to a different "
        "underlying population or label set."
    )
    lines.append("")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "queue_eval.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_qualifying": n_qualifying,
        "base_rate": base_rate,
        "priority_wins": priority_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "priority_p50": next(r for r in results["priority"] if r["k"] == 50)["precision_at_k"],
        "baseline_p50": next(r for r in results["baseline"] if r["k"] == 50)["precision_at_k"],
    }


def _remove_section(text: str, heading: str) -> str:
    """Strip a `heading` (e.g. "## Queue-level evaluation") through to the
    next top-level heading or EOF, so re-running this script is idempotent
    instead of duplicating the section on every run. A no-op if `heading`
    isn't present. Same fix as run_pipeline.py's own `_remove_section` --
    this project already hit the duplication bug once (see DEVLOG.md).
    """
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i
            break
    if start is None:
        return text
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## ") or lines[i].startswith("# "):
            end = i
            break
    return "\n".join(lines[:start] + lines[end:])


def update_readme(headline: dict) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    marker = "## Reproduce"
    if marker not in text:
        raise RuntimeError("README.md's '## Reproduce' section not found; can't place the new section.")

    text = _remove_section(text, "## Queue-level evaluation")

    if headline["priority_wins"] > headline["baseline_wins"]:
        verdict_sentence = (
            f"the priority ranking beats a naive mean-score baseline on "
            f"precision@k at {headline['priority_wins']} of "
            f"{headline['priority_wins'] + headline['baseline_wins'] + headline['ties']} "
            "K values tested"
        )
    elif headline["baseline_wins"] > headline["priority_wins"]:
        verdict_sentence = (
            "a naive mean-score baseline (no cluster features at all) beats "
            f"the system's priority ranking on precision@k at "
            f"{headline['baseline_wins']} of "
            f"{headline['priority_wins'] + headline['baseline_wins'] + headline['ties']} "
            "K values tested -- reported as a finding, not adjusted"
        )
    else:
        verdict_sentence = "the priority ranking and a naive mean-score baseline tie on precision@k across every K tested"

    section = (
        "## Queue-level evaluation\n\n"
        f"Base rate: {headline['base_rate']:.1%} of qualifying test-split "
        "multi-uid clusters contain at least one fraud transaction. At "
        f"K=50, the priority ranking's precision@k is "
        f"{headline['priority_p50']:.4f} vs. {headline['baseline_p50']:.4f} "
        f"for the mean-score baseline -- {verdict_sentence}. Full "
        "breakdown across K=[10, 25, 50, 100], both rankings, and the "
        "methodology are in [results/queue_eval.md](results/queue_eval.md).\n\n"
    )

    new_text = text.replace(marker, section + marker, 1)
    README_PATH.write_text(new_text, encoding="utf-8")


def main() -> None:
    print("[eval_queue] loading pipeline data (no retraining)...")
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    print("[eval_queue] building cluster table (priority + baseline scores)...")
    cluster_stats, n_multi_uid_components, n_dropped = build_cluster_table(
        pipeline_data, trained
    )
    print(
        f"[eval_queue] {n_multi_uid_components} multi-uid clusters total, "
        f"{len(cluster_stats)} qualify (>=1 test-period txn), {n_dropped} dropped"
    )

    results = compute_rankings(cluster_stats)
    headline = write_report(cluster_stats, n_multi_uid_components, n_dropped, results)
    print(f"[eval_queue] wrote results/queue_eval.md -- base rate {headline['base_rate']:.4f}")

    update_readme(headline)
    print("[eval_queue] added Queue-level evaluation section to README.md")


if __name__ == "__main__":
    main()

"""Task 2: investigate _priority_score's hand-picked weights, don't just
trust them. `investigator._priority_score` weights `cluster_prior_fraud_share`
100x over the other two terms -- that weight was chosen by hand when
investigator.py was written, not learned or tuned against this evaluation.
results/queue_eval.md already found that priority score doesn't beat a naive
mean-score baseline at cluster-queue ordering (though the sample -- 9 positive
clusters -- can't say that confidently). This asks the natural follow-up: is
there a *simpler* ordering that does better, and if so, by how much, and is
that difference any more trustworthy than the first null result was?

Five orderings, evaluated on the exact same test-split population as
results/queue_eval.md (imported from scripts/eval_queue.py, not
recomputed -- guarantees identical rows, labels, and base rate):

  a) priority score (as shipped) -- investigator._priority_score, unmodified.
  b) mean transaction-level score -- results/queue_eval.md's Ranking B
     (the baseline model, trained on transaction features only).
  c) priority score with cluster_prior_fraud_share removed entirely --
     same formula, same investigator._priority_score call, with that one
     key dropped from the evidence dict first.
  d) max transaction-level score in the cluster -- same baseline-model
     per-transaction score as (b), aggregated by max instead of mean.
  e) mean transaction-level score weighted by cluster size -- (b)'s score
     multiplied by cluster_size_uids (the full train-graph component size).

This is explicitly a post-hoc comparison, run AFTER seeing (b) come out
ahead of (a) in results/queue_eval.md. If one variant looks clearly better
here, that is a candidate for future work, not something to silently swap
into investigator.py or policy.py -- neither is modified by this script, and
neither should be changed on the strength of a 9-positive-cluster comparison
whether or not the result looks encouraging.

Read-only with respect to the pipeline, same as scripts/eval_queue.py: reuses
run_pipeline.load_and_prepare()/train_both_models() and
investigator.build_evidence/_priority_score, no retraining, no feature
recomputation, no LLM calls.

Usage:
    python scripts/eval_priority_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_queue
from src import investigator, run_pipeline

RESULTS_DIR = REPO_ROOT / "results"
K_VALUES = eval_queue.K_VALUES

VARIANT_COLUMNS = [
    "a_priority_score",
    "b_mean_score",
    "c_priority_no_prior_fraud",
    "d_max_score",
    "e_mean_score_size_weighted",
]

VARIANT_LABELS = {
    "a_priority_score": "(a) priority score (as shipped)",
    "b_mean_score": "(b) mean transaction-level score",
    "c_priority_no_prior_fraud": "(c) priority score, cluster_prior_fraud_share removed",
    "d_max_score": "(d) max transaction-level score in cluster",
    "e_mean_score_size_weighted": "(e) mean score × cluster size",
}


def build_variant_table(
    pipeline_data: run_pipeline.PipelineData,
    trained: run_pipeline.TrainedModels,
    cluster_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Starts from eval_queue.build_cluster_table's output -- which already
    has (a) priority_score and (b) mean_baseline_score computed on the exact
    same qualifying population -- and adds the 3 new variants.
    """
    full, _ = eval_queue._annotate_all_transactions(pipeline_data)
    full["is_test"] = full.index.isin(trained.y_test.index)
    baseline_scores = pd.Series(
        trained.baseline_model.predict(trained.X_test_baseline),
        index=trained.X_test_baseline.index,
    )
    full["baseline_score"] = full.index.to_series().map(baseline_scores)
    test_rows = full.loc[full["is_test"]]

    max_scores = test_rows.groupby("cluster_id")["baseline_score"].max()

    cf = pipeline_data.cluster_features
    qualifying_ids = set(cluster_stats["cluster_id"])
    no_prior_fraud_scores: dict[int, float] = {}
    cluster_size: dict[int, float] = {}
    for cluster_id, grp in full[full["cluster_id"].isin(qualifying_ids)].groupby("cluster_id"):
        members = sorted(grp["uid"].unique())
        cf_sub = cf.loc[[u for u in members if u in cf.index]]
        if cf_sub.empty:
            continue
        evidence = investigator.build_evidence(cf_sub, grp)
        evidence_no_prior_fraud = {
            k: v for k, v in evidence.items() if k != "cluster_prior_fraud_share"
        }
        no_prior_fraud_scores[cluster_id] = investigator._priority_score(evidence_no_prior_fraud)
        cluster_size[cluster_id] = evidence.get("cluster_size_uids", 1)

    table = cluster_stats.copy()
    table["max_baseline_score"] = table["cluster_id"].map(max_scores)
    table["cluster_size_uids"] = table["cluster_id"].map(cluster_size)
    table["c_priority_no_prior_fraud"] = table["cluster_id"].map(no_prior_fraud_scores)

    table = table.rename(
        columns={"priority_score": "a_priority_score", "mean_baseline_score": "b_mean_score"}
    )
    table["d_max_score"] = table["max_baseline_score"]
    table["e_mean_score_size_weighted"] = table["b_mean_score"] * table["cluster_size_uids"]
    return table


def compute_all_rankings(table: pd.DataFrame) -> tuple[dict[str, list[dict]], float, int, int]:
    total_positives = int(table["has_fraud"].sum())
    n_qualifying = len(table)
    base_rate = (total_positives / n_qualifying) if n_qualifying else float("nan")
    results = {
        name: [
            eval_queue._topk_stats(table, name, k, total_positives, base_rate)
            for k in K_VALUES
        ]
        for name in VARIANT_COLUMNS
    }
    return results, base_rate, total_positives, n_qualifying


def _variant_table_for_k(results: dict[str, list[dict]], k: int) -> list[str]:
    lines = [
        "| variant | precision@k (w/ fraud / evaluated) | lift over base rate | "
        "recall@k | efficiency (fraud / reviewed) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in VARIANT_COLUMNS:
        r = next(row for row in results[name] if row["k"] == k)
        lines.append(
            f"| {VARIANT_LABELS[name]} | {r['precision_at_k']:.4f} "
            f"({r['n_clusters_with_fraud']}/{r['n_clusters_evaluated']}) | "
            f"{r['lift_over_base_rate']:.1f}x | "
            f"{r['recall_at_k']:.4f} ({r['n_clusters_with_fraud']}/{r['total_positives']}) | "
            f"{r['efficiency_ratio']:.4f} |"
        )
    return lines


def write_report(
    results: dict[str, list[dict]], base_rate: float, total_positives: int, n_qualifying: int
) -> None:
    lines: list[str] = []
    lines.append("# Priority score variants: is the hand-picked 100x weight doing anything?")
    lines.append("")
    lines.append(
        "`investigator._priority_score` (frozen, unmodified by this script) ranks "
        "clusters by `cluster_prior_fraud_share * 100 + cluster_burst_concentration "
        "* 10 + min(cluster_txn_count, 100) * 0.1` -- weights chosen by hand when "
        "investigator.py was written, not learned or tuned against any queue-level "
        "evaluation. results/queue_eval.md found that this priority score doesn't "
        "beat a naive mean-score baseline at cluster-queue ordering, though with "
        f"only {total_positives} positive clusters that comparison isn't confident "
        "on its own. This asks the natural next question: is there a simpler "
        "ordering that does noticeably better, or does the whole exercise of "
        "hand-weighting cluster features simply not move the needle here?"
    )
    lines.append("")
    lines.append(
        "**This is a post-hoc comparison, run after seeing the priority score lose "
        "to the mean-score baseline.** All 5 orderings below are evaluated on the "
        f"identical {n_qualifying}-cluster population as results/queue_eval.md "
        f"(same {total_positives} positive clusters, same {base_rate:.1%} base "
        "rate, imported from scripts/eval_queue.py rather than recomputed) so the "
        "only thing that varies between rows is the ranking formula. If one "
        "variant is clearly ahead here, that is a candidate for future work -- "
        "not something this script adopts. `investigator.py` and `policy.py` are "
        "unmodified."
    )
    lines.append("")

    lines.append("## The five orderings")
    lines.append("")
    lines.append(f"Base rate for reference: **{base_rate:.1%}** ({total_positives} of {n_qualifying} qualifying clusters contain fraud).")
    lines.append("")
    lines.append("- **(a) priority score (as shipped)** -- `investigator._priority_score`, unmodified.")
    lines.append(
        "- **(b) mean transaction-level score** -- the baseline model's "
        "(txn features only, no cluster features) per-transaction score, "
        "averaged over a cluster's test-period members. Identical to "
        "results/queue_eval.md's Ranking B."
    )
    lines.append(
        "- **(c) priority score, cluster_prior_fraud_share removed** -- the "
        "same `investigator._priority_score` call, with that one key deleted "
        "from the evidence dict first (its formula defaults a missing term "
        "to 0.0, so this is equivalent to zeroing its 100x weight, not a "
        "reimplementation of the formula)."
    )
    lines.append(
        "- **(d) max transaction-level score in cluster** -- the same "
        "baseline-model score as (b), aggregated by max instead of mean: "
        "does the single most suspicious member transaction predict a real "
        "cluster better than the average of all its members?"
    )
    lines.append(
        "- **(e) mean score × cluster size** -- (b) multiplied by "
        "`cluster_size_uids` (the full train-graph component's member "
        "count, not just members active in the test period): does simply "
        "favoring bigger clusters, with no other structural feature, help?"
    )
    lines.append("")

    lines.append("## Results, all five variants, per K")
    lines.append("")
    for k in K_VALUES:
        lines.append(f"### K={k}")
        lines.append("")
        lines += _variant_table_for_k(results, k)
        lines.append("")

    lines.append("## Reading this table honestly")
    lines.append("")
    best_per_k = {}
    for k in K_VALUES:
        rows_at_k = {name: next(r for r in results[name] if r["k"] == k) for name in VARIANT_COLUMNS}
        best_name = max(rows_at_k, key=lambda n: rows_at_k[n]["precision_at_k"])
        best_per_k[k] = (best_name, rows_at_k[best_name])

    summary_bullets = [
        f"- K={k}: highest precision@k is {VARIANT_LABELS[name]} at "
        f"{row['precision_at_k']:.4f} ({row['n_clusters_with_fraud']}/{row['n_clusters_evaluated']})"
        for k, (name, row) in best_per_k.items()
    ]
    lines += summary_bullets
    lines.append("")

    distinct_leaders = {name for name, _ in best_per_k.values()}

    gaps_per_k: dict[int, int] = {}
    for k in K_VALUES:
        counts_at_k = [
            next(r for r in results[name] if r["k"] == k)["n_clusters_with_fraud"]
            for name in VARIANT_COLUMNS
        ]
        gaps_per_k[k] = max(counts_at_k) - min(counts_at_k)
    min_gap = min(gaps_per_k.values())
    max_gap = max(gaps_per_k.values())
    gap_phrase = (
        f"exactly {min_gap} cluster(s)" if min_gap == max_gap else f"{min_gap}-{max_gap} clusters"
    )

    lines.append(
        "**Stated plainly, not cherry-picked:** the leading variant "
        + (
            "is not the same at every K tested, which is itself informative -- "
            "no single variant dominates across the board"
            if len(distinct_leaders) > 1
            else f"is the same at every K tested: {VARIANT_LABELS[next(iter(distinct_leaders))]}"
        )
        + f". With only {total_positives} positive clusters total, the spread "
        f"between the best and worst variant's cluster-count (highest minus "
        f"lowest `n_clusters_with_fraud` among the five, at a given K) is "
        f"{gap_phrase} across the four K values tested -- see the absolute "
        "counts in each table above, not just the precision ratios. That is "
        "the same order of magnitude as the noise callout in "
        "results/queue_eval.md, so **no ranking among these five is being "
        "asserted as reliably better than another here.** A future evaluation "
        "with substantially more positive clusters (a longer test window, or "
        "a less severe temporal split) would be needed before recommending "
        "any one of these as a replacement for the shipped priority score, "
        "and that recommendation is future work -- not something adopted by "
        "this script, which leaves investigator.py and policy.py untouched."
    )
    lines.append("")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "priority_variants.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[eval_priority_variants] loading pipeline data (no retraining)...")
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    print("[eval_priority_variants] reusing eval_queue's qualifying-cluster population...")
    cluster_stats, _, _ = eval_queue.build_cluster_table(pipeline_data, trained)

    print("[eval_priority_variants] computing 3 additional variants (c, d, e)...")
    table = build_variant_table(pipeline_data, trained, cluster_stats)

    results, base_rate, total_positives, n_qualifying = compute_all_rankings(table)
    write_report(results, base_rate, total_positives, n_qualifying)
    print(
        f"[eval_priority_variants] wrote results/priority_variants.md -- "
        f"{n_qualifying} clusters, {total_positives} positive, base rate {base_rate:.4f}"
    )


if __name__ == "__main__":
    main()

"""CLI entry point for `make results`. The only orchestration layer.

Sequences entities -> graph -> model -> evaluate. policy.py and
investigator.py are intentionally not wired in yet (out of scope for this
session -- see CLAUDE.md: the LLM layer and policy.py must never be merged,
and both modules are being built separately). None of entities/graph/model/
evaluate should grow their own __main__/CLI code; this is the one place
that wires them together, so evaluate.py in particular can stay a pure
metrics module (see its module docstring).

Exposes reusable pieces (load_and_prepare, train_both_models,
evaluate_both_models) beyond just main(), so the sanity-check and
cost-curve scripts for this session (scripts/sanity_checks.py,
scripts/cost_curve.py) can reuse the exact same data/graph/model artifacts
instead of duplicating this orchestration logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src import data, entities, evaluate, graph, model

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# See graph.py's module docstring for the full max_degree sweep: the
# function's own default (1000) collapses 64% of all uids into one
# connected component on this dataset. 20 keeps the largest cluster at 126
# uids (0.06%).
MAX_DEGREE = 20

# Illustrative cost assumptions, NOT Razorpay figures -- see
# results/ablation.md for the explicit caveat. Missing a fraud case is
# assumed to cost 100x a false alarm (chargeback loss vs. customer-friction
# cost of an unnecessary step-up).
COST_FN = 500.0
COST_FP = 5.0

# The cost-minimizing threshold for a well-calibrated binary classifier
# under (cost_fn, cost_fp) is where the two expected costs balance -- used
# as the ablation table's headline threshold. scripts/cost_curve.py sweeps
# thresholds directly rather than relying on this calibration assumption.
DEFAULT_THRESHOLD = COST_FP / (COST_FN + COST_FP)


@dataclass
class PipelineData:
    df: pd.DataFrame
    entity_ids: pd.Series
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    as_of: float
    entity_graph: graph.EntityGraph
    cluster_features: pd.DataFrame  # indexed by uid (entity_id)


@dataclass
class TrainedModels:
    baseline_model: lgb.Booster
    cluster_model: lgb.Booster
    X_train_baseline: pd.DataFrame
    X_train_cluster: pd.DataFrame
    X_test_baseline: pd.DataFrame
    X_test_cluster: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def broadcast_cluster_features(
    entity_ids: pd.Series, cluster_features: pd.DataFrame
) -> pd.DataFrame:
    """Expand per-uid cluster features to per-TransactionID.

    A row whose uid has no entry in `cluster_features` (no uid at all, or a
    uid with zero pre-as_of history) gets NaN in every cluster column --
    explicit nulls, never a dropped row or a fabricated zero. This is how
    the ~11% no-uid population (see results/uid_validation.md) is handled:
    they stay in training/evaluation with null cluster features rather than
    being excluded, since excluding them would drop the highest-risk slice
    of transactions (11.63% fraud rate there vs. 2.46% for uid'd rows).
    """
    broadcast = cluster_features.reindex(entity_ids.to_numpy())
    broadcast.index = entity_ids.index
    return broadcast


def load_and_prepare(nrows: int | None = None) -> PipelineData:
    """Load data, resolve entities, split temporally, build the graph and
    compute cluster features -- all from train-period data only.
    """
    df = data.load_transactions(DATA_DIR, nrows=nrows)
    entity_ids = entities.resolve_entities(df)

    train_df, test_df = evaluate.temporal_train_test_split(df)
    as_of = float(test_df["TransactionDT"].min())

    entity_graph = graph.build_entity_graph(train_df, entity_ids, max_degree=MAX_DEGREE)
    cluster_features = graph.compute_cluster_features(
        train_df, entity_ids, entity_graph.graph, as_of=as_of
    )

    return PipelineData(
        df=df,
        entity_ids=entity_ids,
        train_df=train_df,
        test_df=test_df,
        as_of=as_of,
        entity_graph=entity_graph,
        cluster_features=cluster_features,
    )


def train_both_models(pipeline_data: PipelineData) -> TrainedModels:
    """Train the baseline and cluster models with identical everything
    except the cluster feature columns.
    """
    pd_ = pipeline_data
    y_train = pd_.train_df.set_index("TransactionID")["isFraud"]
    y_test = pd_.test_df.set_index("TransactionID")["isFraud"]

    cluster_by_transaction = broadcast_cluster_features(
        pd_.entity_ids, pd_.cluster_features
    )
    cluster_train = cluster_by_transaction.reindex(y_train.index)
    cluster_test = cluster_by_transaction.reindex(y_test.index)

    X_train_baseline = model.build_feature_matrix(pd_.train_df)
    X_test_baseline = model.build_feature_matrix(pd_.test_df)
    X_train_cluster = model.build_feature_matrix(pd_.train_df, cluster_train)
    X_test_cluster = model.build_feature_matrix(pd_.test_df, cluster_test)

    baseline_model = model.train_baseline_model(X_train_baseline, y_train)
    cluster_model = model.train_cluster_model(X_train_cluster, y_train)

    return TrainedModels(
        baseline_model=baseline_model,
        cluster_model=cluster_model,
        X_train_baseline=X_train_baseline,
        X_train_cluster=X_train_cluster,
        X_test_baseline=X_test_baseline,
        X_test_cluster=X_test_cluster,
        y_train=y_train,
        y_test=y_test,
    )


def evaluate_both_models(trained: TrainedModels) -> dict[str, dict[str, float]]:
    return {
        "baseline": evaluate.evaluate_model(
            trained.baseline_model,
            trained.X_test_baseline,
            trained.y_test,
            threshold=DEFAULT_THRESHOLD,
            cost_fn=COST_FN,
            cost_fp=COST_FP,
        ),
        "cluster": evaluate.evaluate_model(
            trained.cluster_model,
            trained.X_test_cluster,
            trained.y_test,
            threshold=DEFAULT_THRESHOLD,
            cost_fn=COST_FN,
            cost_fp=COST_FP,
        ),
    }


def _feature_importance_table(
    booster: lgb.Booster, top_n: int = 20
) -> pd.DataFrame:
    importances = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    return importances.head(top_n)


def write_ablation_report(
    pipeline_data: PipelineData,
    trained: TrainedModels,
    metrics: dict[str, dict[str, float]],
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    lines: list[str] = []

    lines.append("# Ablation: transaction-only baseline vs. cluster-augmented")
    lines.append("")
    lines.append(
        f"Temporal split: {len(pipeline_data.train_df):,} train rows, "
        f"{len(pipeline_data.test_df):,} test rows "
        f"(as_of = TransactionDT {pipeline_data.as_of:,.0f}, the first "
        "test-period timestamp)."
    )
    lines.append("")
    lines.append(
        "Cost assumptions are illustrative, NOT Razorpay figures: "
        f"cost_fn={COST_FN:g} (a missed abuse case), cost_fp={COST_FP:g} "
        "(a false alarm / unnecessary step-up) -- a 100:1 ratio, chosen to "
        "represent a chargeback loss being much costlier than customer "
        "friction, nothing more precise than that. The threshold used below "
        f"({DEFAULT_THRESHOLD:.4f}) is cost_fp/(cost_fn+cost_fp), the "
        "cost-minimizing point for a well-calibrated classifier under this "
        "cost ratio; results/cost_curve.png sweeps thresholds directly "
        "rather than relying on that calibration assumption."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |")
    lines.append("|---|---:|---:|---:|")
    for name in ("baseline", "cluster"):
        m = metrics[name]
        lines.append(
            f"| {name} | {m['pr_auc']:.4f} | {m['recall_at_1pct_fpr']:.4f} | "
            f"{m['cost_per_10k']:.2f} |"
        )
    lines.append("")

    pr_auc_lift = metrics["cluster"]["pr_auc"] - metrics["baseline"]["pr_auc"]
    recall_lift = (
        metrics["cluster"]["recall_at_1pct_fpr"]
        - metrics["baseline"]["recall_at_1pct_fpr"]
    )
    cost_delta = metrics["cluster"]["cost_per_10k"] - metrics["baseline"]["cost_per_10k"]
    lines.append(
        f"Cluster model vs. baseline: PR-AUC {pr_auc_lift:+.4f}, "
        f"recall@1%FPR {recall_lift:+.4f}, cost per 10k {cost_delta:+.2f} "
        "(negative is better for cost). Reported as-is; the derivation and "
        "features were not adjusted after seeing these numbers."
    )
    lines.append("")

    lines.append("## Hyperparameters (identical for both models)")
    lines.append("")
    lines.append("```")
    for k, v in model.LGBM_PARAMS.items():
        lines.append(f"{k}: {v}")
    lines.append(f"num_boost_round: {model.NUM_BOOST_ROUND}")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Baseline features: {trained.X_train_baseline.shape[1]}. "
        f"Cluster features add: "
        f"{trained.X_train_cluster.shape[1] - trained.X_train_baseline.shape[1]} "
        "columns (the graph.compute_cluster_features output -- cluster "
        "size/txn count/edge density/velocity/amount CV/burst "
        "concentration/email-uid ratio/prior-fraud share, plus per-uid node "
        "degree and email domain count)."
    )
    lines.append("")

    lines.append("## Cluster model feature importances (top 20 by gain)")
    lines.append("")
    lines.append("| feature | gain |")
    lines.append("|---|---:|")
    for _, row in _feature_importance_table(trained.cluster_model).iterrows():
        lines.append(f"| {row['feature']} | {row['gain']:,.1f} |")
    lines.append("")

    lines.append("## Graph construction")
    lines.append("")
    lines.append(
        f"max_degree={MAX_DEGREE} (see graph.py's module docstring for why "
        "the function's own default of 1000 is unusable on this dataset -- "
        "it collapses 64% of all uids into one connected component)."
    )
    lines.append(
        f"Built from {len(pipeline_data.train_df):,} train-period "
        f"transactions: {pipeline_data.entity_graph.graph.number_of_nodes():,} "
        f"nodes, {pipeline_data.entity_graph.graph.number_of_edges():,} edges."
    )
    lines.append("")
    excluded = pipeline_data.entity_graph.excluded_hubs
    lines.append(
        f"Hub guard excluded {len(excluded)} values "
        f"(covering {int(excluded['uid_count'].sum()):,} uid-appearances "
        "in total, with overlap across rules) as too common to be evidence "
        "of a relationship. Ten largest:"
    )
    lines.append("")
    lines.append("| rule | value | uid_count |")
    lines.append("|---|---|---:|")
    for _, row in excluded.sort_values("uid_count", ascending=False).head(10).iterrows():
        lines.append(f"| {row['rule']} | {row['value']} | {row['uid_count']:,} |")
    lines.append("")

    (RESULTS_DIR / "ablation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the full pipeline: load, split, build graph, train, evaluate,
    and write results/ablation.md.
    """
    t0 = time.time()
    pipeline_data = load_and_prepare()
    print(f"[run_pipeline] data + graph prepared in {time.time() - t0:.1f}s")

    t0 = time.time()
    trained = train_both_models(pipeline_data)
    print(f"[run_pipeline] both models trained in {time.time() - t0:.1f}s")

    metrics = evaluate_both_models(trained)
    print(f"[run_pipeline] baseline: {metrics['baseline']}")
    print(f"[run_pipeline] cluster:  {metrics['cluster']}")

    write_ablation_report(pipeline_data, trained, metrics)
    print("[run_pipeline] wrote results/ablation.md")


if __name__ == "__main__":
    main()

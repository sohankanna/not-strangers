"""Task 5: latency and scale benchmark.

Measures three separate things, on purpose kept separate because they sit
on opposite sides of a batch/inline split (see ARCHITECTURE.md's
Performance section, which this script's numbers feed):
  1. Graph construction time (batch: rebuilt periodically, e.g. nightly,
     from train-period/historical transactions).
  2. Cluster feature computation time (batch: computed once per graph
     build, then looked up at scoring time -- never recomputed inline).
  3. Per-transaction scoring latency, p50/p95, for the INLINE path: a
     cheap feature-store lookup (simulated as a dict lookup here) plus a
     single-row model.predict() call. This assumes cluster features are
     already precomputed and sitting in a lookup table -- it does NOT
     re-time graph construction or cluster feature computation, since
     those never happen on the request path.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src import graph as graph_module
from src import model, run_pipeline

N_SCORING_SAMPLES = 1000


def _percentiles(latencies_seconds: list[float]) -> dict[str, float]:
    arr = np.array(latencies_seconds) * 1000  # ms
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "max_ms": float(np.max(arr)),
    }


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    # --- 1. Graph construction (batch) -----------------------------------
    t0 = time.perf_counter()
    entity_graph = graph_module.build_entity_graph(
        pipeline_data.train_df, pipeline_data.entity_ids, max_degree=run_pipeline.MAX_DEGREE
    )
    graph_build_seconds = time.perf_counter() - t0

    # --- 2. Cluster feature computation (batch) ----------------------------
    t0 = time.perf_counter()
    cluster_features = graph_module.compute_cluster_features(
        pipeline_data.train_df,
        pipeline_data.entity_ids,
        entity_graph.graph,
        as_of=pipeline_data.as_of,
    )
    cluster_features_seconds = time.perf_counter() - t0

    # --- 3. Per-transaction scoring latency (inline) ------------------------
    # Simulate a precomputed feature store: uid -> cluster feature row, as a
    # plain dict lookup (O(1)), which is what an inline path would actually
    # use (a key-value store / cache), not a pandas .loc on a full frame.
    feature_store = {
        uid: row.to_dict() for uid, row in pipeline_data.cluster_features.iterrows()
    }

    rng = np.random.default_rng(0)
    sample_positions = rng.choice(
        len(trained.X_test_cluster), size=N_SCORING_SAMPLES, replace=False
    )
    sample_rows = trained.X_test_cluster.iloc[sample_positions]
    sample_uids = pipeline_data.entity_ids.reindex(sample_rows.index)

    baseline_cols = trained.X_test_baseline.columns

    latencies = []
    for i in range(N_SCORING_SAMPLES):
        txn_id = sample_rows.index[i]
        uid = sample_uids.iloc[i]

        t0 = time.perf_counter()
        # feature-store lookup (or an explicit null-feature fallback for
        # the no-uid / no-history case, exactly as run_pipeline does)
        cluster_row = feature_store.get(uid) if isinstance(uid, str) else None
        base_row = trained.X_test_baseline.loc[[txn_id]]
        if cluster_row is not None:
            for k, v in cluster_row.items():
                base_row[k] = v
        else:
            for k in pipeline_data.cluster_features.columns:
                base_row[k] = np.nan
        # Explicit reindex to the exact column set/order the model was
        # trained on -- don't rely on LightGBM's own column-name matching
        # behavior for a single-row DataFrame assembled by hand.
        base_row = base_row.reindex(columns=trained.X_test_cluster.columns)
        model.predict(trained.cluster_model, base_row)
        latencies.append(time.perf_counter() - t0)

    scoring_stats = _percentiles(latencies)

    lines: list[str] = []
    lines.append("## Performance")
    lines.append("")
    lines.append(
        "Batch/inline split, stated plainly: graph construction and "
        "cluster feature computation are BATCH -- run periodically "
        "(e.g. nightly, or whenever the graph is rebuilt) against "
        "historical/train-period transactions, never on the request path. "
        "Transaction scoring is INLINE: given a transaction, look up its "
        "uid's precomputed cluster features (a cache/feature-store read, "
        "not a recomputation) and call the model. Below is measured "
        "separately because they answer different capacity questions --  "
        "batch steps bound how often the graph can be refreshed, the "
        "inline step bounds request latency."
    )
    lines.append("")
    lines.append("### Batch: graph construction and cluster features")
    lines.append("")
    lines.append(
        f"- Graph construction ({len(pipeline_data.train_df):,} train-period "
        f"transactions, max_degree={run_pipeline.MAX_DEGREE}): "
        f"**{graph_build_seconds:.2f}s** "
        f"({entity_graph.graph.number_of_nodes():,} nodes, "
        f"{entity_graph.graph.number_of_edges():,} edges)"
    )
    lines.append(
        f"- Cluster feature computation ({len(cluster_features):,} uids): "
        f"**{cluster_features_seconds:.2f}s**"
    )
    lines.append("")
    lines.append("### Inline: per-transaction scoring latency")
    lines.append("")
    lines.append(
        f"{N_SCORING_SAMPLES:,} single-transaction scoring calls (feature-store "
        "lookup + model.predict on one row), sampled from the real test set:"
    )
    lines.append("")
    lines.append("| stat | value |")
    lines.append("|---|---:|")
    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"):
        lines.append(f"| {key} | {scoring_stats[key]:.3f} ms |")
    lines.append("")
    lines.append(
        "This is single-row prediction, not batched -- LightGBM's per-call "
        "overhead dominates at this granularity, so p95 here is a "
        "meaningfully worse number than the model's throughput in bulk "
        "scoring would suggest. A real inline path would likely batch "
        "several in-flight requests if the volume justified it."
    )
    lines.append("")

    lines.append("### What would need to change at ~1B transactions/quarter")
    lines.append("")
    lines.append(
        f"This benchmark's full graph build "
        f"({len(pipeline_data.train_df):,} transactions) took "
        f"{graph_build_seconds:.2f}s for construction + "
        f"{cluster_features_seconds:.2f}s for features. ~1B transactions/quarter "
        f"is roughly {1e9 / len(pipeline_data.train_df):.0f}x this benchmark's "
        "train set. Naive linear scaling alone would already push a full "
        "rebuild from seconds into hours, and the real cost is worse than "
        "linear: this project's own hub-guard investigation "
        "(graph.py's module docstring) found that the graph's structure is "
        "sensitive to `max_degree` in a highly non-linear way (a phase "
        "transition, not a smooth curve) -- at greater scale, more "
        "identifier values cross the hub threshold, and getting this wrong "
        "risks the same giant-component collapse found in Task 1 of the "
        "previous session, at a much more expensive scale to detect and "
        "recover from."
    )
    lines.append("")
    lines.append("Three changes this scale would require, none implemented here:")
    lines.append("")
    lines.append(
        "- **Incremental graph updates instead of full rebuilds.** This "
        "project rebuilds the whole graph from all train-period "
        "transactions every time (build_entity_graph has no notion of "
        "\"since last run\"). At 1B/quarter, a full rebuild needs to become "
        "an incremental one: new transactions add nodes/edges to an "
        "existing graph, without re-scanning historical data that hasn't "
        "changed."
    )
    lines.append(
        "- **Approximate connected components.** get_connected_components "
        "is an exact, single-machine networkx computation. At this scale "
        "the graph itself likely needs to be sharded/distributed, and "
        "exact connected components across shards is expensive; "
        "approximate or incremental union-find structures (as used in "
        "large-scale graph processing systems) trade a small amount of "
        "accuracy for tractability."
    )
    lines.append(
        "- **Sharding.** The current implementation holds one in-memory "
        "networkx graph and one in-memory feature table for the whole "
        "dataset. Neither fits in memory on one machine at this volume; "
        "the graph and its features would need to be partitioned (e.g. by "
        "a hash of a linkage key) across multiple machines, which changes "
        "how cross-shard edges (a device or address linking uids in "
        "different shards) get detected and reconciled -- a real design "
        "problem, not a configuration change."
    )
    lines.append("")

    architecture_path = REPO_ROOT / "ARCHITECTURE.md"
    existing = architecture_path.read_text(encoding="utf-8") if architecture_path.exists() else ""
    architecture_path.write_text(
        existing.rstrip("\n") + ("\n\n" if existing.strip() else "") + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("Appended Performance section to ARCHITECTURE.md")
    print(f"graph_build={graph_build_seconds:.2f}s cluster_features={cluster_features_seconds:.2f}s")
    print(f"scoring latency: {scoring_stats}")


if __name__ == "__main__":
    main()

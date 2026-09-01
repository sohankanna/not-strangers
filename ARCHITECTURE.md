## Performance

Batch/inline split, stated plainly: graph construction and cluster feature computation are BATCH -- run periodically (e.g. nightly, or whenever the graph is rebuilt) against historical/train-period transactions, never on the request path. Transaction scoring is INLINE: given a transaction, look up its uid's precomputed cluster features (a cache/feature-store read, not a recomputation) and call the model. Below is measured separately because they answer different capacity questions --  batch steps bound how often the graph can be refreshed, the inline step bounds request latency.

### Batch: graph construction and cluster features

- Graph construction (472,432 train-period transactions, max_degree=20): **2.53s** (167,111 nodes, 62,804 edges)
- Cluster feature computation (167,111 uids): **13.31s**

### Inline: per-transaction scoring latency

1,000 single-transaction scoring calls (feature-store lookup + model.predict on one row), sampled from the real test set:

| stat | value |
|---|---:|
| p50_ms | 31.786 ms |
| p95_ms | 33.649 ms |
| p99_ms | 34.432 ms |
| mean_ms | 31.029 ms |
| max_ms | 43.111 ms |

This is single-row prediction, not batched -- LightGBM's per-call overhead dominates at this granularity, so p95 here is a meaningfully worse number than the model's throughput in bulk scoring would suggest. A real inline path would likely batch several in-flight requests if the volume justified it.

### What would need to change at ~1B transactions/quarter

This benchmark's full graph build (472,432 transactions) took 2.53s for construction + 13.31s for features. ~1B transactions/quarter is roughly 2117x this benchmark's train set. Naive linear scaling alone would already push a full rebuild from seconds into hours, and the real cost is worse than linear: this project's own hub-guard investigation (graph.py's module docstring) found that the graph's structure is sensitive to `max_degree` in a highly non-linear way (a phase transition, not a smooth curve) -- at greater scale, more identifier values cross the hub threshold, and getting this wrong risks the same giant-component collapse found earlier this project, at a much more expensive scale to detect and recover from.

Three changes this scale would require, none implemented here:

- **Incremental graph updates instead of full rebuilds.** This project rebuilds the whole graph from all train-period transactions every time (build_entity_graph has no notion of "since last run"). At 1B/quarter, a full rebuild needs to become an incremental one: new transactions add nodes/edges to an existing graph, without re-scanning historical data that hasn't changed.
- **Approximate connected components.** get_connected_components is an exact, single-machine networkx computation. At this scale the graph itself likely needs to be sharded/distributed, and exact connected components across shards is expensive; approximate or incremental union-find structures (as used in large-scale graph processing systems) trade a small amount of accuracy for tractability.
- **Sharding.** The current implementation holds one in-memory networkx graph and one in-memory feature table for the whole dataset. Neither fits in memory on one machine at this volume; the graph and its features would need to be partitioned (e.g. by a hash of a linkage key) across multiple machines, which changes how cross-shard edges (a device or address linking uids in different shards) get detected and reconciled -- a real design problem, not a configuration change.


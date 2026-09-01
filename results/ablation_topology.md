# Topology features ablation: does cluster SHAPE add anything?

results/stability.md found that once `cluster_prior_fraud_share` is removed, the remaining graph-structure aggregates (cluster size, edge density, velocity, amount CV, burst concentration) show no reliable lift across rolling temporal splits: mean +0.0060, spread 0.0230, sign flips between splits. That is this project's central weakness -- the graph-structure hypothesis is currently unsupported. This tests it once more, on the same single 80% split as results/ablation.md, with two richer topology features instead of only aggregates: `k_core_number` (this uid's k-core index -- how deeply embedded it is in a dense subgraph) and `star_ratio` (max node degree in the cluster / cluster size -- the hub-and-spoke device-farm signature, as distinct from a mutually-connected clique). Both are additive-only extensions to graph.compute_cluster_features (see Task 1); nothing about the existing 10 features or their computation changed.

Rows 1-2 are read directly from `run_pipeline.load_and_prepare()` / `train_both_models()` -- the identical call that produces results/ablation.md's own numbers, not retrained here. Rows 3-4 train fresh models via `model.train_cluster_model` (frozen, unmodified) on modified feature sets; every metric below comes from `evaluate.py`'s frozen `pr_auc`/`recall_at_fpr`/`cost_per_10k`, exactly as results/ablation.md's own numbers do.

## Results

| row | model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |
|---:|---|---:|---:|---:|
| 1 | baseline (txn features only) | 0.5646 | 0.4791 | 30078.40 |
| 2 | + original cluster features | 0.6322 | 0.5576 | 26155.72 |
| 3 | + original cluster features, minus cluster_prior_fraud_share | 0.5756 | 0.4951 | 29317.66 |
| 4 | + original + topology features, minus cluster_prior_fraud_share | 0.5711 | 0.4936 | 30151.22 |

**Row 4 vs. row 3 (topology added, dominant confound removed from both): PR-AUC -0.0045, recall@1%FPR -0.0015, cost per 10k +833.56** (negative is better for cost). This single-split delta is the direct answer to "does topology add anything the aggregates didn't" -- reported as-is; see results/stability_topology.md (Task 3) for whether this delta, positive or negative, holds up across rolling splits the way results/stability.md found the aggregate-only version did not.

## Feature importances, row 4's model (top 20 by gain)

| feature | gain |
|---|---:|
| V258 | 96,191.7 |
| DeviceInfo | 40,874.2 |
| C1 | 35,042.7 |
| C14 | 34,457.3 |
| cluster_amt_cv | 28,458.9 |
| C13 | 27,498.7 |
| cluster_velocity | 27,438.4 |
| R_emaildomain | 26,173.0 |
| card2 | 22,894.0 |
| V294 | 20,399.9 |
| cluster_txn_count | 19,249.1 |
| TransactionAmt | 18,895.2 |
| C11 | 17,494.9 |
| TransactionDT | 17,384.0 |
| D2 | 15,781.9 |
| card1 | 15,731.4 |
| addr1 | 15,014.7 |
| P_emaildomain | 14,575.6 |
| cluster_burst_concentration | 13,997.4 |
| id_31 | 12,938.0 |

## Where the two new topology features land, specifically

Row 4's model has 443 total features. Reported explicitly regardless of top-20 rank, since a low-importance feature would otherwise be invisible above:

- `k_core_number`: gain **187.0**, rank **228** of 443 total features.
- `star_ratio`: gain **948.0**, rank **113** of 443 total features.

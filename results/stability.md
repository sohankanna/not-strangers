# Stability of the cluster-feature lift across rolling temporal splits

results/ablation.md's headline lift (+0.0676 PR-AUC, cluster model vs. baseline) comes from exactly one train/test boundary: the last 20% of TransactionDT held out. A lift measured at a single split point could be a property of the model and features, or it could be a property of that particular boundary -- e.g. a burst of coordinated activity that happens to fall right at the 80% mark. This re-runs the full ablation (fresh entity graph, fresh causal cluster features, fresh baseline/cluster/trimmed-cluster models -- nothing cached or reused from results/ablation.md's own run) at 4 rolling split points: 60%, 70%, 80%, 90% of the way through the sorted TransactionDT range, train on everything before that point, test on everything after.

Only entities.py, graph.py, model.py, and evaluate.py were imported to produce these numbers (all frozen, none modified this session); run_pipeline.broadcast_cluster_features and MAX_DEGREE=20 are reused as-is rather than re-derived, since run_pipeline.py is not one of the frozen modules and this is the same broadcast logic every other artifact in this project already relies on.

**All 4 planned split points ran, in 408s total.**

## Results

| split | train rows | test rows | as_of (TransactionDT) | baseline PR-AUC | cluster PR-AUC | lift | trimmed PR-AUC (no cluster_prior_fraud_share) | trimmed lift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60% | 354,324 | 236,216 | 8,745,798 | 0.5620 | 0.5903 | +0.0284 | 0.5576 | -0.0044 |
| 70% | 413,378 | 177,162 | 10,438,003 | 0.5570 | 0.6210 | +0.0641 | 0.5756 | +0.0186 |
| 80% | 472,432 | 118,108 | 12,192,900 | 0.5646 | 0.6322 | +0.0676 | 0.5756 | +0.0110 |
| 90% | 531,486 | 59,054 | 13,990,941 | 0.6243 | 0.6706 | +0.0463 | 0.6231 | -0.0013 |

**Consistency check:** the 80% split point above should closely reproduce results/ablation.md's own numbers (baseline 0.5646, cluster 0.6322, lift +0.0676), since it's the same train/test boundary computed the same way. This run got baseline 0.5646, cluster 0.6322, lift +0.0676 -- matching closely (small differences are expected: LightGBM's own internal nondeterminism across runs, not a bug).

## Mean and spread of the lift across splits

- Full cluster-feature lift: mean **+0.0516** PR-AUC across the 4 splits that ran, spread (max - min) **0.0393** (min +0.0284, max +0.0676).
- Lift with `cluster_prior_fraud_share` removed: mean **+0.0060**, spread **0.0230** (min -0.0044, max +0.0186).

**The lift swings substantially across split points -- a spread of 0.0393 against a mean of +0.0516 is a large fraction of the effect itself.** results/ablation.md's single-split headline number should be read as one sample from a noisy distribution, not a fixed property of the feature set. This is reported as the finding, not smoothed over -- see the per-split table above for exactly where the lift was strongest and weakest.

**The residual lift (after removing `cluster_prior_fraud_share`) changes sign across splits -- negative at 60% (-0.0044), 90% (-0.0013), positive at 70% (+0.0186), 80% (+0.0110).** README.md currently describes the structural/graph features (edge density, velocity, burst concentration, email heterogeneity, cluster size -- everything left once `cluster_prior_fraud_share` is removed) as contributing "a smaller but genuine residual lift on their own," based on the single 80% split (+0.0110, matching this run's 80% row). Across 4 splits, that residual lift is not reliably positive: mean +0.0060, spread 0.0230, sign flipping between splits. The honest read is that the non-prior-fraud structural features' contribution is close to noise-level here, not a small-but-real effect -- flagged plainly as a finding this script produced, left for README.md's existing wording to be revisited on its own rather than edited as a side effect of this report.

Cost per 10k and recall@1%FPR (results/ablation.md's other two headline metrics) were not re-measured here -- this script is scoped to PR-AUC and the lift, per the task that produced it; re-running the full cost/calibration analysis at every split point would multiply the already-substantial cost of this script several times over for metrics that move together with PR-AUC in every split already measured in results/ablation.md and results/cost_curve.png.

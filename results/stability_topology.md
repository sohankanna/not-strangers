# Stability of the residual lift, with topology features included

results/stability.md found the residual lift (PR-AUC gain over baseline once `cluster_prior_fraud_share` is removed, leaving only aggregate cluster features) is mean +0.0060, spread 0.0230, and changes sign across the 4 rolling splits (60/70/80/90% through sorted TransactionDT) -- the central weakness this whole session exists to re-test. This repeats that exact comparison at the same 4 split points, with two topology features (`k_core_number`, `star_ratio` -- see Task 1 / src/graph.py) added to the trimmed feature set. **This is the question the session exists to answer: does the residual lift become reliably positive across all four splits with topology included, or does it still flip sign?**

Same methodology as results/stability.md: fresh entity graph, fresh causal cluster features, fresh models at each split -- nothing cached or reused across splits. Only entities.py, graph.py, model.py, and evaluate.py were imported (all frozen, none modified) plus run_pipeline.broadcast_cluster_features and MAX_DEGREE=20, reused as-is. No feature definitions, thresholds, or max_degree were tuned after seeing any result below.

**All 4 planned split points ran, in 890s total.**

## Results

| split | baseline PR-AUC | cluster PR-AUC | full lift | trimmed PR-AUC (no topology) | trimmed lift (no topology) | + topology PR-AUC | trimmed lift (WITH topology) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60% | 0.5620 | 0.5903 | +0.0284 | 0.5576 | -0.0044 | 0.5600 | -0.0020 |
| 70% | 0.5570 | 0.6210 | +0.0641 | 0.5756 | +0.0186 | 0.5705 | +0.0135 |
| 80% | 0.5646 | 0.6322 | +0.0676 | 0.5756 | +0.0110 | 0.5711 | +0.0065 |
| 90% | 0.6243 | 0.6706 | +0.0463 | 0.6231 | -0.0013 | 0.6140 | -0.0104 |

## Residual lift, without topology vs. with topology

- **Without topology (results/stability.md's existing finding, reproduced here on the same splits):** mean **+0.0060**, spread **0.0230** (min -0.0044, max +0.0186).
- **With topology (k_core_number + star_ratio added):** mean **+0.0019**, spread **0.0239** (min -0.0104, max +0.0135).

## Verdict

**Topology does not rescue the residual lift.** It still changes sign across splits (negative at 60%, 90%, positive at 70%, 80%), same as the aggregate-only residual (mean +0.0019 vs. +0.0060 without topology, spread 0.0239 vs. 0.0230).

**Reported plainly, as the task that produced this asked: the graph-structure hypothesis was tested twice now -- once with aggregate cluster features (results/stability.md), once with richer topology features added on top (this report) -- and neither version shows a reliable, sign-stable lift once the dominant `cluster_prior_fraud_share` confound is removed, on this dataset at this sample size (4 rolling splits). This is not evidence that graph structure can never predict abuse; it is evidence that this project's specific attempts to measure it -- aggregates, then topology -- have not found a stable signal here. No feature definition, threshold, or max_degree was adjusted to avoid this conclusion.**

Cost per 10k and recall@1%FPR were not re-measured here, for the same reason results/stability.md didn't: this script is scoped to PR-AUC and the lift, and re-running the full cost/calibration analysis at every split point for both feature sets would multiply an already-substantial cost several times over.

# Queue-level evaluation: cluster-level precision@k

This project's headline number (results/ablation.md) is transaction-level PR-AUC. That's a proxy -- it measures how well the model ranks individual transactions, not whether an analyst team working a queue of flagged *clusters* would actually find real abuse near the top. This closes that gap: for K in [10, 25, 50, 100], if reviewers worked the top K clusters today, how many are real, how many transactions would they review to find them, and how does the system's actual ranking compare to a naive baseline that ignores cluster structure entirely?

Produced by `scripts/eval_queue.py`, run on the test split only (evaluate.temporal_train_test_split's existing temporal holdout), reusing the existing train-only entity graph and cluster assignments. No features were recomputed and no model was retrained to produce this report -- it calls run_pipeline.load_and_prepare() and run_pipeline.train_both_models() exactly as every other results/*.md artifact in this project does.

## Methodology

**Population:** multi-uid clusters (2+ members) only -- a single-uid "cluster" is not a ring, and including them would inflate every number below. There are 1,567 multi-uid connected components in the train-only entity graph in total (consistent with results/case_studies.md's 1,567 figure, same graph); this report further restricts to the 494 of those that have at least one test-period transaction, since only those can be scored and labeled on the test split. Every count below (fraud surfaced, workload) is over test-period transactions only, in those clusters.

**Ranking A -- priority ranking:** the ordering the system actually produces. `investigator.build_evidence` + `investigator._priority_score` computed over each cluster's full transaction history (train + test), exactly matching app.py's `build_cluster_queue` (the dashboard's review queue) and results/case_studies.md's ranking. Formula: `cluster_prior_fraud_share * 100 + cluster_burst_concentration * 10 + min(cluster_txn_count, 100) * 0.1`.

**Ranking B -- baseline ranking:** clusters ranked by the mean of the *baseline* model's per-transaction score (results/ablation.md's "baseline (txn features only)" row -- trained on transaction features alone, no cluster features at train time at all) across each cluster's test-period member transactions. This isolates whether the graph/cluster-feature apparatus adds anything over "just average how suspicious this cluster's transactions already look to a plain transaction classifier".

## Base rate

**9 of 494 qualifying test-split clusters (0.0182, i.e. 1.8%) contain at least one fraud transaction.**

Precision@k is meaningless without this number: a precision@k barely above the base rate means the ranking is doing little better than picking clusters at random from the qualifying population. Every precision@k figure below should be read relative to this 0.0182 baseline, not in isolation.

## Results

### Ranking A: priority ranking (system / dashboard / policy ordering)

| K | clusters evaluated | precision@k | fraud txns surfaced | workload (test txns reviewed) | efficiency (fraud / reviewed) |
|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 0.2000 | 8 | 38 | 0.2105 |
| 25 | 25 | 0.0800 | 8 | 122 | 0.0656 |
| 50 | 50 | 0.0600 | 9 | 312 | 0.0288 |
| 100 | 100 | 0.0600 | 55 | 764 | 0.0720 |

### Ranking B: baseline ranking (mean transaction-level score, no cluster features)

| K | clusters evaluated | precision@k | fraud txns surfaced | workload (test txns reviewed) | efficiency (fraud / reviewed) |
|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 0.4000 | 46 | 81 | 0.5679 |
| 25 | 25 | 0.2000 | 54 | 188 | 0.2872 |
| 50 | 50 | 0.1000 | 54 | 293 | 0.1843 |
| 100 | 100 | 0.0600 | 56 | 599 | 0.0935 |

## Verdict

- K=10: priority precision@k=0.2000 (efficiency 0.2105) vs. baseline precision@k=0.4000 (efficiency 0.5679) -- **baseline ahead**
- K=25: priority precision@k=0.0800 (efficiency 0.0656) vs. baseline precision@k=0.2000 (efficiency 0.2872) -- **baseline ahead**
- K=50: priority precision@k=0.0600 (efficiency 0.0288) vs. baseline precision@k=0.1000 (efficiency 0.1843) -- **baseline ahead**
- K=100: priority precision@k=0.0600 (efficiency 0.0720) vs. baseline precision@k=0.0600 (efficiency 0.0935) -- **tied**

**The mean-score baseline beats the priority ranking on precision@k at 3 of 4 K values tested** (0 for priority, 1 tied). This is reported as-is -- the ranking was not adjusted after seeing this result. A null (or negative) result here is a legitimate finding about where the cluster-topology features do and don't help: they were built to lift transaction-level PR-AUC (results/ablation.md), and doing that is not the same guarantee as producing a better cluster-priority ordering.

Both rankings draw from the same qualifying population and the same test-split labels -- the only thing that differs between them is the ordering applied to that population, so any precision@k gap above is attributable to the ranking method, not to a different underlying population or label set.

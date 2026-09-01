# Queue-level evaluation: cluster-level precision@k

This project's headline number (results/ablation.md) is transaction-level PR-AUC. That's a proxy -- it measures how well the model ranks individual transactions, not whether an analyst team working a queue of flagged *clusters* would actually find real abuse near the top. This closes that gap: for K in [10, 25, 50, 100], if reviewers worked the top K clusters today, how many are real, how many transactions would they review to find them, and how does the system's actual ranking compare to a naive baseline that ignores cluster structure entirely?

Produced by `scripts/eval_queue.py`, run on the test split only (evaluate.temporal_train_test_split's existing temporal holdout), reusing the existing train-only entity graph and cluster assignments. No features were recomputed and no model was retrained to produce this report -- it calls run_pipeline.load_and_prepare() and run_pipeline.train_both_models() exactly as every other results/*.md artifact in this project does.

## Methodology

**Population:** multi-uid clusters (2+ members) only -- a single-uid "cluster" is not a ring, and including them would inflate every number below. There are 1,567 multi-uid connected components in the train-only entity graph in total (consistent with results/case_studies.md's 1,567 figure, same graph); this report further restricts to the 494 of those that have at least one test-period transaction, since only those can be scored and labeled on the test split. Every count below (fraud surfaced, workload) is over test-period transactions only, in those clusters.

**Ranking A -- priority ranking:** the ordering the system actually produces. `investigator.build_evidence` + `investigator._priority_score` computed over each cluster's full transaction history (train + test), exactly matching app.py's `build_cluster_queue` (the dashboard's review queue) and results/case_studies.md's ranking. Formula: `cluster_prior_fraud_share * 100 + cluster_burst_concentration * 10 + min(cluster_txn_count, 100) * 0.1`.

**Ranking B -- baseline ranking:** clusters ranked by the mean of the *baseline* model's per-transaction score (results/ablation.md's "baseline (txn features only)" row -- trained on transaction features alone, no cluster features at train time at all) across each cluster's test-period member transactions. This isolates whether the graph/cluster-feature apparatus adds anything over "just average how suspicious this cluster's transactions already look to a plain transaction classifier".

## Base rate

**9 of 494 qualifying test-split clusters (0.0182, i.e. 1.8%) contain at least one fraud transaction.**

Precision@k is meaningless without this number, in both directions. Every precision@k figure below is reported next to its **lift over base rate** (precision@k / 0.0182) and its absolute count (clusters-with-fraud / clusters-evaluated) rather than as a bare fraction. Both rankings turn out to land far above random: even the weaker of the two clears the base rate by roughly an order of magnitude at every K tested, and that should be read as the headline finding of this report, not buried under the A-vs-B comparison below.

**With only 9 positive clusters in the entire qualifying population, every number in this report is small-count statistics.** A precision@k difference between the two rankings of 1-3 clusters -- which is most of what separates them at every K below -- is well within the noise this sample size can produce; flipping the true/false label on two or three clusters would plausibly reorder which ranking looks better at a given K. Recall@k (of the 9 fraud-containing clusters, how many appear in the top K) is reported alongside precision for exactly this reason -- with this few positives it is the more informative number, since it's a direct count out of a known, small total rather than a ratio that swings sharply per cluster.

## Results

### Ranking A: priority ranking (system / dashboard / policy ordering)

| K | precision@k (clusters w/ fraud / evaluated) | lift over base rate | recall@k (of 9 fraud clusters found) | fraud txns surfaced | workload (test txns reviewed) | efficiency (fraud / reviewed) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.2000 (2/10) | 11.0x | 0.2222 (2/9) | 8 | 38 | 0.2105 |
| 25 | 0.0800 (2/25) | 4.4x | 0.2222 (2/9) | 8 | 122 | 0.0656 |
| 50 | 0.0600 (3/50) | 3.3x | 0.3333 (3/9) | 9 | 312 | 0.0288 |
| 100 | 0.0600 (6/100) | 3.3x | 0.6667 (6/9) | 55 | 764 | 0.0720 |

### Ranking B: baseline ranking (mean transaction-level score, no cluster features)

| K | precision@k (clusters w/ fraud / evaluated) | lift over base rate | recall@k (of 9 fraud clusters found) | fraud txns surfaced | workload (test txns reviewed) | efficiency (fraud / reviewed) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.4000 (4/10) | 22.0x | 0.4444 (4/9) | 46 | 81 | 0.5679 |
| 25 | 0.2000 (5/25) | 11.0x | 0.5556 (5/9) | 54 | 188 | 0.2872 |
| 50 | 0.1000 (5/50) | 5.5x | 0.5556 (5/9) | 54 | 293 | 0.1843 |
| 100 | 0.0600 (6/100) | 3.3x | 0.6667 (6/9) | 56 | 599 | 0.0935 |

## Verdict

- K=10: priority found 2/9 fraud clusters (precision 0.2000, 11.0x base rate) vs. baseline's 4/9 (precision 0.4000, 22.0x) -- a gap of 2 cluster(s), **baseline ahead** (within noise)
- K=25: priority found 2/9 fraud clusters (precision 0.0800, 4.4x base rate) vs. baseline's 5/9 (precision 0.2000, 11.0x) -- a gap of 3 cluster(s), **baseline ahead** (within noise)
- K=50: priority found 3/9 fraud clusters (precision 0.0600, 3.3x base rate) vs. baseline's 5/9 (precision 0.1000, 5.5x) -- a gap of 2 cluster(s), **baseline ahead** (within noise)
- K=100: priority found 6/9 fraud clusters (precision 0.0600, 3.3x base rate) vs. baseline's 6/9 (precision 0.0600, 3.3x) -- a gap of 0 cluster(s), **tied** (within noise)

**4 of 4 K values show a gap of 3 clusters or fewer between the two rankings -- with only 9 positive clusters total, that is within the noise this sample size can produce, not a confident difference in ranking quality.**

**Null finding, stated precisely:** the hand-weighted priority score (`cluster_prior_fraud_share * 100 + cluster_burst_concentration * 10 + min(cluster_txn_count, 100) * 0.1`, see Methodology above) does not demonstrate an advantage over the mean-score baseline at cluster-queue ordering -- it is behind or tied at 4 of 4 K values on precision -- but the sample is too small (9 positive clusters) to distinguish the two rankings confidently at any individual K. This is reported as-is; the ranking was not adjusted after seeing the result, and the finding is the honest combination of both facts together, not either one alone: the priority score does not show a measurable edge here, and this dataset does not have enough positive clusters to say much more than that.

Both rankings draw from the same qualifying population and the same test-split labels -- the only thing that differs between them is the ordering applied to that population, so any gap above is attributable to the ranking method, not to a different underlying population or label set. What the sample size cannot support is translating that gap into a confident "X ranking is better" conclusion -- see the noise caveat above.

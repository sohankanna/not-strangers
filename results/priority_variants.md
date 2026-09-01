# Priority score variants: is the hand-picked 100x weight doing anything?

`investigator._priority_score` (frozen, unmodified by this script) ranks clusters by `cluster_prior_fraud_share * 100 + cluster_burst_concentration * 10 + min(cluster_txn_count, 100) * 0.1` -- weights chosen by hand when investigator.py was written, not learned or tuned against any queue-level evaluation. results/queue_eval.md found that this priority score doesn't beat a naive mean-score baseline at cluster-queue ordering, though with only 9 positive clusters that comparison isn't confident on its own. This asks the natural next question: is there a simpler ordering that does noticeably better, or does the whole exercise of hand-weighting cluster features simply not move the needle here?

**This is a post-hoc comparison, run after seeing the priority score lose to the mean-score baseline.** All 5 orderings below are evaluated on the identical 494-cluster population as results/queue_eval.md (same 9 positive clusters, same 1.8% base rate, imported from scripts/eval_queue.py rather than recomputed) so the only thing that varies between rows is the ranking formula. If one variant is clearly ahead here, that is a candidate for future work -- not something this script adopts. `investigator.py` and `policy.py` are unmodified.

## The five orderings

Base rate for reference: **1.8%** (9 of 494 qualifying clusters contain fraud).

- **(a) priority score (as shipped)** -- `investigator._priority_score`, unmodified.
- **(b) mean transaction-level score** -- the baseline model's (txn features only, no cluster features) per-transaction score, averaged over a cluster's test-period members. Identical to results/queue_eval.md's Ranking B.
- **(c) priority score, cluster_prior_fraud_share removed** -- the same `investigator._priority_score` call, with that one key deleted from the evidence dict first (its formula defaults a missing term to 0.0, so this is equivalent to zeroing its 100x weight, not a reimplementation of the formula).
- **(d) max transaction-level score in cluster** -- the same baseline-model score as (b), aggregated by max instead of mean: does the single most suspicious member transaction predict a real cluster better than the average of all its members?
- **(e) mean score × cluster size** -- (b) multiplied by `cluster_size_uids` (the full train-graph component's member count, not just members active in the test period): does simply favoring bigger clusters, with no other structural feature, help?

## Results, all five variants, per K

### K=10

| variant | precision@k (w/ fraud / evaluated) | lift over base rate | recall@k | efficiency (fraud / reviewed) |
|---|---:|---:|---:|---:|
| (a) priority score (as shipped) | 0.2000 (2/10) | 11.0x | 0.2222 (2/9) | 0.2105 |
| (b) mean transaction-level score | 0.4000 (4/10) | 22.0x | 0.4444 (4/9) | 0.5679 |
| (c) priority score, cluster_prior_fraud_share removed | 0.1000 (1/10) | 5.5x | 0.1111 (1/9) | 0.0156 |
| (d) max transaction-level score in cluster | 0.5000 (5/10) | 27.4x | 0.5556 (5/9) | 0.4821 |
| (e) mean score × cluster size | 0.4000 (4/10) | 22.0x | 0.4444 (4/9) | 0.3866 |

### K=25

| variant | precision@k (w/ fraud / evaluated) | lift over base rate | recall@k | efficiency (fraud / reviewed) |
|---|---:|---:|---:|---:|
| (a) priority score (as shipped) | 0.0800 (2/25) | 4.4x | 0.2222 (2/9) | 0.0656 |
| (b) mean transaction-level score | 0.2000 (5/25) | 11.0x | 0.5556 (5/9) | 0.2872 |
| (c) priority score, cluster_prior_fraud_share removed | 0.0800 (2/25) | 4.4x | 0.2222 (2/9) | 0.0110 |
| (d) max transaction-level score in cluster | 0.2000 (5/25) | 11.0x | 0.5556 (5/9) | 0.2160 |
| (e) mean score × cluster size | 0.2000 (5/25) | 11.0x | 0.5556 (5/9) | 0.1791 |

### K=50

| variant | precision@k (w/ fraud / evaluated) | lift over base rate | recall@k | efficiency (fraud / reviewed) |
|---|---:|---:|---:|---:|
| (a) priority score (as shipped) | 0.0600 (3/50) | 3.3x | 0.3333 (3/9) | 0.0288 |
| (b) mean transaction-level score | 0.1000 (5/50) | 5.5x | 0.5556 (5/9) | 0.1843 |
| (c) priority score, cluster_prior_fraud_share removed | 0.1000 (5/50) | 5.5x | 0.5556 (5/9) | 0.0799 |
| (d) max transaction-level score in cluster | 0.1200 (6/50) | 6.6x | 0.6667 (6/9) | 0.1264 |
| (e) mean score × cluster size | 0.1200 (6/50) | 6.6x | 0.6667 (6/9) | 0.1089 |

### K=100

| variant | precision@k (w/ fraud / evaluated) | lift over base rate | recall@k | efficiency (fraud / reviewed) |
|---|---:|---:|---:|---:|
| (a) priority score (as shipped) | 0.0600 (6/100) | 3.3x | 0.6667 (6/9) | 0.0720 |
| (b) mean transaction-level score | 0.0600 (6/100) | 3.3x | 0.6667 (6/9) | 0.0935 |
| (c) priority score, cluster_prior_fraud_share removed | 0.0500 (5/100) | 2.7x | 0.5556 (5/9) | 0.0527 |
| (d) max transaction-level score in cluster | 0.0600 (6/100) | 3.3x | 0.6667 (6/9) | 0.0685 |
| (e) mean score × cluster size | 0.0600 (6/100) | 3.3x | 0.6667 (6/9) | 0.0685 |

## Reading this table honestly

- K=10: highest precision@k is (d) max transaction-level score in cluster at 0.5000 (5/10)
- K=25: highest precision@k is (b) mean transaction-level score at 0.2000 (5/25)
- K=50: highest precision@k is (d) max transaction-level score in cluster at 0.1200 (6/50)
- K=100: highest precision@k is (a) priority score (as shipped) at 0.0600 (6/100)

**Stated plainly, not cherry-picked:** the leading variant is not the same at every K tested, which is itself informative -- no single variant dominates across the board. With only 9 positive clusters total, the spread between the best and worst variant's cluster-count (highest minus lowest `n_clusters_with_fraud` among the five, at a given K) is 1-4 clusters across the four K values tested -- see the absolute counts in each table above, not just the precision ratios. That is the same order of magnitude as the noise callout in results/queue_eval.md, so **no ranking among these five is being asserted as reliably better than another here.** A future evaluation with substantially more positive clusters (a longer test window, or a less severe temporal split) would be needed before recommending any one of these as a replacement for the shipped priority score, and that recommendation is future work -- not something adopted by this script, which leaves investigator.py and policy.py untouched.

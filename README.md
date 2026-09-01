# abuse-ring-sentinel

Detects coordinated payment abuse by resolving transactions into entities
and scoring the clusters they form, rather than scoring transactions in isolation.

## Result

| Model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |
|---|---|---|---|
| Baseline (txn features only) | 0.5646 | 0.4791 | 30,078.40 |
| + cluster features | 0.6322 | 0.5576 | 26,155.72 |
| + cluster features, minus cluster_prior_fraud_share | 0.5756 | 0.4951 | 29,317.66 |

Temporal holdout (472,432 train / 118,108 test rows). Reproduce:
`make results`. Costs (cost_fn=500, cost_fp=5) are illustrative, not
Razorpay figures. Full breakdown, hyperparameters, feature importances,
adversarial sanity checks and the threshold/cost curve are in
[results/ablation.md](results/ablation.md).

The third row matters: about 84% of the headline PR-AUC lift comes from
one feature (cluster_prior_fraud_share). It was traced end to end and
found not to leak test-period labels (verified across all 155,579 clusters,
not just spot-checked -- see results/ablation.md's Sanity checks section),
but its predictive power leans heavily on the label-propagation dynamic
described below, not on an independent abuse signal. The remaining
structural/graph features (edge density, velocity, burst concentration,
email heterogeneity, cluster size) contribute a smaller but genuine
residual lift on their own.

## Why clusters
## Data & labels
## Architecture
## Running it

## Limitations

- **Label noise.** Labels are chargeback-reported and, per how this
  dataset is constructed, propagate across a card once one transaction on
  it is reported -- a single confirmed chargeback can retroactively paint
  every other transaction on that card as fraud, whether or not each one
  actually was. `cluster_prior_fraud_share`, the single largest driver of
  the measured lift, is close to a direct measurement of this same
  propagation dynamic ("has this persistent card-identity already been
  caught"). That makes it a legitimate, non-leaking feature, not an
  independently-discovered abuse signal -- the model is partly learning to
  reproduce the label-generation process itself.
- **The uid over-merges.** `card1_addr1_origin_day` is a stable, highly
  label-pure identifier (98.53% of multi-transaction uids are label-pure,
  weighted 97.61% -- see results/uid_validation.md), but stability is not
  the same as correctness. The collision check in
  results/d1_investigation.md found P_emaildomain varying within 10 of the
  20 largest uids -- distinct people are demonstrably sharing a uid. A uid
  containing several distinct people who share a card fingerprint is
  treated in this project as signal (coordinated abuse), not an error to
  fix, but it means "cluster" here is not a verified single-person
  identity.
- **~11% of rows get no uid at all**, mostly from a missing addr1
  (66,794 of 590,540 rows). These are not dropped -- they get null cluster
  features and stay in training/evaluation -- because they are the
  *highest-risk* population: 11.63% fraud rate among NaN-uid rows vs. 2.46%
  among uid'd rows (results/uid_validation.md). Any cluster-based system
  that silently excludes unresolvable rows would be excluding
  disproportionately dangerous traffic, not a random slice.
- **Clusters have no ground truth.** There is no "this is a real
  coordinated ring" label anywhere in this dataset to validate cluster
  correctness against. Everything reported here is *feature lift* --
  whether adding cluster-derived features improves a fraud classifier's
  ranking and cost metrics on a temporal holdout -- not *ring-detection
  accuracy*. A cluster with a high fraud-history feature might be one
  coordinated ring, several unrelated people who happen to share a
  fingerprint, or a mix of both; this project cannot currently tell those
  apart, and doesn't claim to.
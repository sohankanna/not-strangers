# Ablation: transaction-only baseline vs. cluster-augmented

Temporal split: 472,432 train rows, 118,108 test rows (as_of = TransactionDT 12,192,900, the first test-period timestamp).

Cost assumptions are illustrative, NOT Razorpay figures: cost_fn=500 (a missed abuse case), cost_fp=5 (a false alarm / unnecessary step-up) -- a 100:1 ratio, chosen to represent a chargeback loss being much costlier than customer friction, nothing more precise than that. The threshold used below (0.0099) is cost_fp/(cost_fn+cost_fp), the cost-minimizing point for a well-calibrated classifier under this cost ratio; results/cost_curve.png sweeps thresholds directly rather than relying on that calibration assumption.

## Results

| model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |
|---|---:|---:|---:|
| baseline | 0.5646 | 0.4791 | 30078.40 |
| cluster | 0.6322 | 0.5576 | 26155.72 |

Cluster model vs. baseline: PR-AUC +0.0676, recall@1%FPR +0.0785, cost per 10k -3922.68 (negative is better for cost). Reported as-is; the derivation and features were not adjusted after seeing these numbers.

## Hyperparameters (identical for both models)

```
objective: binary
metric: None
verbosity: -1
seed: 42
num_leaves: 63
learning_rate: 0.05
feature_fraction: 0.8
bagging_fraction: 0.8
bagging_freq: 1
min_data_in_leaf: 50
num_boost_round: 300
```

Baseline features: 432. Cluster features add: 10 columns (the graph.compute_cluster_features output -- cluster size/txn count/edge density/velocity/amount CV/burst concentration/email-uid ratio/prior-fraud share, plus per-uid node degree and email domain count).

## Cluster model feature importances (top 20 by gain)

| feature | gain |
|---|---:|
| cluster_prior_fraud_share | 558,334.3 |
| V258 | 63,550.3 |
| C1 | 52,819.7 |
| DeviceInfo | 28,908.5 |
| C14 | 27,892.7 |
| cluster_size_uids | 27,155.8 |
| uid_email_domain_count | 15,768.0 |
| C13 | 14,856.7 |
| cluster_amt_cv | 13,615.7 |
| card2 | 10,830.0 |
| TransactionDT | 9,482.1 |
| TransactionAmt | 9,441.0 |
| V156 | 8,742.6 |
| id_31 | 8,726.3 |
| D2 | 8,573.2 |
| C7 | 7,891.9 |
| cluster_txn_count | 7,771.6 |
| cluster_velocity | 7,521.8 |
| C11 | 7,514.6 |
| P_emaildomain | 6,932.8 |

## Graph construction

max_degree=20 (see graph.py's module docstring for why the function's own default of 1000 is unusable on this dataset -- it collapses 64% of all uids into one connected component).
Built from 472,432 train-period transactions: 167,111 nodes, 62,804 edges.

Hub guard excluded 1114 values (covering 376,264 uid-appearances in total, with overlap across rules) as too common to be evidence of a relationship. Ten largest:

| rule | value | uid_count |
|---|---|---:|
| device_info | Windows | 18,535 |
| device_info | iOS Device | 12,719 |
| device_info | MacOS | 8,344 |
| card_bank_addr | (150.0, 226.0, 299.0) | 6,545 |
| addr1_email | (299.0, 'gmail.com') | 6,414 |
| card_bank_addr | (150.0, 226.0, 204.0) | 5,824 |
| addr1_email | (264.0, 'gmail.com') | 5,548 |
| addr1_email | (204.0, 'gmail.com') | 5,471 |
| addr1_email | (325.0, 'gmail.com') | 5,434 |
| card_bank_addr | (150.0, 226.0, 325.0) | 5,145 |

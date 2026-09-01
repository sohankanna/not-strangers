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

## Sanity checks

### 1. Correlation of cluster features with isFraud (train set)

Computed over the 472,432 train rows; each feature's correlation uses only the rows where that feature is non-null (pandas' default pairwise behavior) -- n_valid shows how many that was per feature.

| feature | correlation with isFraud | n_valid |
|---|---:|---:|
| cluster_prior_fraud_share | 0.7797 **(>0.5)** | 417,872 |
| cluster_velocity | 0.0573 | 417,872 |
| cluster_edge_density | -0.0469 | 30,436 |
| cluster_txn_count | 0.0327 | 417,872 |
| cluster_burst_concentration | -0.0311 | 417,872 |
| cluster_size_uids | 0.0309 | 417,872 |
| node_degree | 0.0230 | 417,872 |
| cluster_email_uid_ratio | -0.0077 | 417,872 |
| cluster_amt_cv | -0.0038 | 328,202 |
| uid_email_domain_count | -0.0013 | 417,872 |

**1 feature(s) exceed the 0.5 red-flag threshold: cluster_prior_fraud_share (0.780).** Investigated in section 2.

### 2. Tracing cluster_prior_fraud_share for a leak

cluster_prior_fraud_share is computed inside graph.compute_cluster_features as: for every transaction with `TransactionDT < as_of`, take the per-uid max of isFraud over *that uid's own* qualifying transactions (`by_uid["isFraud"].max()`), then average that per-uid flag across the uids in each cluster. The `as_of` filter is applied to `transactions` before any of this runs -- there is no code path in compute_cluster_features that reads isFraud from a row that didn't pass the `TransactionDT < as_of` filter. That's the argument from reading the code (and it's what tests/test_graph.py's explicit leakage test already checks structurally). Below is the argument from the actual data instead.

Checked all 155,579 clusters that have both a reported cluster_prior_fraud_share and an independently-recomputable pre-as_of value: comparing the pipeline's reported value against one computed straight from raw rows with `TransactionDT < as_of`, bypassing graph.py entirely. Mismatches: **0**.

Concrete example -- the cluster where leaking would change this feature the most: cluster #17894, 1 member uids.

Its 2 fraud-labeled transaction(s) (all other member rows are isFraud=0, omitted for brevity):

| uid | TransactionDT | period | isFraud |
|---|---:|---|---:|
| 2616_123_-36 | 14,916,239 | test | 1 |
| 2616_123_-36 | 15,255,382 | test | 1 |

- Reported by the pipeline (as_of-filtered): **0.0000**
- Independently recomputed from raw rows with `TransactionDT < as_of`, bypassing graph.py entirely: **0.0000**
- What it would be if test-period rows leaked in (all-time max isFraud per member, no as_of filter): **1.0000**

**Reported value matches the independently-recomputed pre-as_of-only value, and differs from the would-leak value for this cluster -- confirmed on real data, not just in the unit test: no test-period fraud label influenced this cluster's prior-fraud feature.**

### 3. Ablation re-run with cluster_prior_fraud_share removed

Same training path (train_cluster_model, identical hyperparameters and seed), same cluster feature set minus this one column.

| model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |
|---|---:|---:|---:|
| cluster (no cluster_prior_fraud_share) | 0.5756 | 0.4951 | 29317.66 |

### 4. Cluster assignment for a test-period transaction never depends on test-period edges

Structurally: `graph.build_entity_graph` is called in run_pipeline.load_and_prepare with `train_df` only -- `test_df` is never passed to it, so no test-period transaction can ever contribute an edge or a node. Verified concretely below rather than just re-reading the call site.

- Every node in the graph corresponds to a uid with at least one train-period transaction: 0 nodes found in the graph with zero train-period transactions (should be 0).
- Concrete example: uid `10004_225_152` appears ONLY in the test period (no train-period transactions). It is absent from the graph's node set, and its broadcast cluster features are all null (as expected -- no train-period history means no cluster signal, not a fabricated one).

## Threshold sweep and cost curve

results/cost_curve.png sweeps score thresholds directly (not relying on the calibration assumption behind ablation.md's headline threshold) and marks each model's own cost-minimizing point.

![Cost per 10k vs threshold](cost_curve.png)

| model | chosen threshold | cost per 10k at chosen point | recall | FPR |
|---|---:|---:|---:|---:|
| baseline | 0.0095 | 30008.97 | 0.9326 | 0.3813 |
| cluster | 0.0103 | 25923.31 | 0.9530 | 0.3695 |

Worth being explicit about: the cost-minimizing FPR here is 37%-38% -- a direct, correct mathematical consequence of the assumed 100:1 cost_fn:cost_fp ratio (missing fraud is assumed to be that much worse than a false alarm, so the optimum flags aggressively), not a bug. In practice this means stepping up upwards of a third of all legitimate transactions at the "optimal" point -- whether that's acceptable is a business call the assumed cost ratio drives entirely; a less aggressive cost ratio (or a friction budget constraint) would move the chosen threshold and the resulting FPR substantially.


# Investigator evaluation

ANTHROPIC_API_KEY was **NOT set** when this ran. Explanation sources: {'ungrounded-fallback': 30}. **Every explanation below took the deterministic fallback path (source=ungrounded-fallback), not the real LLM.** The fallback narrative is built by directly formatting the evidence dict's own values, so it is grounded by construction -- the groundedness rate below measures that template, not claude-sonnet-4-6's actual behavior under the prompt's hard rule. Re-run this script with a real key for an honest measurement of the LLM path.

Evaluated 30 clusters, selected to span the risk range: sorted all multi-uid clusters (2+ members) by cluster_prior_fraud_share, then took 30 evenly-spaced percentile points across that sorted list (not just the top 30 riskiest).

## Groundedness

- Total numeric claims extracted across all narratives: **360**
- Ungrounded claims: **0**
- Groundedness rate: **100.00%** (a claim counts as grounded if it matches some evidence value exactly, at any rounding from 0-4 decimal places, or as that value expressed as a percentage)

No ungrounded claims found in this run.

## 3 example explanations (lowest, median, highest priority)

### cluster-13 (source=ungrounded-fallback)

- Priority score: 2.0090
- Member uids: ['12667_330_105', '12667_330_21', '12667_330_49', '15365_330_-23', '2748_330_13', '5031_330_125', '5031_330_13']
- Evidence: `{'cluster_size_uids': 7, 'cluster_txn_count': 11, 'cluster_edge_density': 1.0, 'cluster_velocity': 0.0978, 'cluster_amt_cv': 1.0557, 'cluster_burst_concentration': 0.0909, 'cluster_email_uid_ratio': 0.1429, 'cluster_prior_fraud_share': 0.0, 'total_transaction_amount': 6085.0, 'mean_transaction_amount': 468.0769, 'distinct_product_codes': 2, 'distinct_email_domains': 1}`

> [template fallback -- no LLM call was made or it failed] Cluster summary: cluster amt cv=1.0557; cluster burst concentration=0.0909; cluster edge density=1.0; cluster email uid ratio=0.1429; cluster prior fraud share=0.0; cluster size uids=7; cluster txn count=11; cluster velocity=0.0978; distinct email domains=1; distinct product codes=2; mean transaction amount=468.0769; total transaction amount=6085.0.

### cluster-7 (source=ungrounded-fallback)

- Priority score: 3.3690
- Member uids: ['10086_520_10', '13832_520_11', '13832_520_6', '13832_520_8', '15885_520_-343', '15885_520_10', '15885_520_12', '15885_520_8', '15885_520_9', '17782_520_7', '2256_520_-363', '3154_520_10', '4461_520_7', '8755_520_-414', '9633_520_-56']
- Evidence: `{'cluster_size_uids': 15, 'cluster_txn_count': 26, 'cluster_edge_density': 0.5238, 'cluster_velocity': 3.3965, 'cluster_amt_cv': 0.921, 'cluster_burst_concentration': 0.0769, 'cluster_email_uid_ratio': 0.3333, 'cluster_prior_fraud_share': 0.0, 'total_transaction_amount': 1129.309, 'mean_transaction_amount': 43.435, 'distinct_product_codes': 1, 'distinct_email_domains': 5}`

> [template fallback -- no LLM call was made or it failed] Cluster summary: cluster amt cv=0.921; cluster burst concentration=0.0769; cluster edge density=0.5238; cluster email uid ratio=0.3333; cluster prior fraud share=0.0; cluster size uids=15; cluster txn count=26; cluster velocity=3.3965; distinct email domains=5; distinct product codes=1; mean transaction amount=43.435; total transaction amount=1129.309.

### cluster-29 (source=ungrounded-fallback)

- Priority score: 102.1290
- Member uids: ['11106_204_101', '15066_204_101', '4151_204_101']
- Evidence: `{'cluster_size_uids': 3, 'cluster_txn_count': 7, 'cluster_edge_density': 1.0, 'cluster_velocity': 7.0, 'cluster_amt_cv': 0.5211, 'cluster_burst_concentration': 0.1429, 'cluster_email_uid_ratio': 0.6667, 'cluster_prior_fraud_share': 1.0, 'total_transaction_amount': 1512.9399, 'mean_transaction_amount': 216.1343, 'distinct_product_codes': 2, 'distinct_email_domains': 2}`

> [template fallback -- no LLM call was made or it failed] Cluster summary: cluster amt cv=0.5211; cluster burst concentration=0.1429; cluster edge density=1.0; cluster email uid ratio=0.6667; cluster prior fraud share=1.0; cluster size uids=3; cluster txn count=7; cluster velocity=7.0; distinct email domains=2; distinct product codes=2; mean transaction amount=216.1343; total transaction amount=1512.9399.

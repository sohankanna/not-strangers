# Investigator evaluation

ANTHROPIC_API_KEY was set when this ran. Explanation sources (derived from the actual run's `source` field, not assumed from whether a key was present): {'llm': 30}.

**All 30 of 30 explanations used the real LLM path.**

Evaluated 30 clusters, selected to span the risk range: sorted all multi-uid clusters (2+ members) by cluster_prior_fraud_share, then took 30 evenly-spaced percentile points across that sorted list (not just the top 30 riskiest).

## Groundedness

**On the 30 real LLM explanation(s): 0 of 176 numeric claims ungrounded -- groundedness rate 100.00%.** This is the number that actually measures claude-sonnet-4-6's behavior under the prompt's hard rule. Reported honestly whatever it is -- a rate below 100% is a finding about the model's behavior, not something to fix by loosening the claim extractor.

- Fallback-only groundedness (for reference; expected ~100% since it's built by directly formatting evidence values verbatim): 0 of 0 claims ungrounded (no fallback explanations this run)
- Combined (LLM + fallback) across all 30 explanations: 0 of 176 claims ungrounded (100.00%). Not the LLM's groundedness rate when fallback explanations are mixed in -- a trivially-grounded template dilutes or inflates the real number, which is why it's reported separately above.

(A claim counts as grounded if it matches some evidence value exactly, at any rounding from 0-4 decimal places, or as that value expressed as a percentage.)

No ungrounded claims found among the LLM explanations.

## 3 example explanations (lowest, median, highest priority)

### cluster-13 (source=llm)

- Priority score: 2.0090
- Member uids: ['12667_330_105', '12667_330_21', '12667_330_49', '15365_330_-23', '2748_330_13', '5031_330_125', '5031_330_13']
- Evidence: `{'cluster_size_uids': 7, 'cluster_txn_count': 11, 'cluster_edge_density': 1.0, 'cluster_velocity': 0.0978, 'cluster_amt_cv': 1.0557, 'cluster_burst_concentration': 0.0909, 'cluster_email_uid_ratio': 0.1429, 'cluster_prior_fraud_share': 0.0, 'total_transaction_amount': 6085.0, 'mean_transaction_amount': 468.0769, 'distinct_product_codes': 2, 'distinct_email_domains': 1}`

> This cluster of 7 linked identities shows a fully connected graph (edge density of 1.0), meaning every identity shares a linking attribute with every other — a strong structural indicator of coordinated account creation rather than coincidental overlap. The extremely low email-to-uid ratio of 0.1429 combined with only 1 distinct email domain across all identities suggests the group is using a single email provider in a tightly controlled way, consistent with synthetic or fabricated account clusters. Activity is spread across only 2 distinct product codes, pointing to focused, deliberate targeting rather than organic, varied shopping behavior. On the other hand, the cluster carries no prior fraud history (fraud share of 0.0), burst concentration is low at 0.0909 suggesting transactions are not crammed into a single moment, and the amount coefficient of variation of 1.0557 indicates some variability in transaction sizes, which slightly tempers the risk signal. Overall the evidence leans toward coordinated abuse warranting closer review, primarily driven by the perfect connectivity, single email domain, and highly concentrated identity linkage.

### cluster-7 (source=llm)

- Priority score: 3.3690
- Member uids: ['10086_520_10', '13832_520_11', '13832_520_6', '13832_520_8', '15885_520_-343', '15885_520_10', '15885_520_12', '15885_520_8', '15885_520_9', '17782_520_7', '2256_520_-363', '3154_520_10', '4461_520_7', '8755_520_-414', '9633_520_-56']
- Evidence: `{'cluster_size_uids': 15, 'cluster_txn_count': 26, 'cluster_edge_density': 0.5238, 'cluster_velocity': 3.3965, 'cluster_amt_cv': 0.921, 'cluster_burst_concentration': 0.0769, 'cluster_email_uid_ratio': 0.3333, 'cluster_prior_fraud_share': 0.0, 'total_transaction_amount': 1129.309, 'mean_transaction_amount': 43.435, 'distinct_product_codes': 1, 'distinct_email_domains': 5}`

> This cluster of 15 linked identities shows several characteristics that warrant scrutiny, though the picture is somewhat mixed. The edge density of 0.5238 indicates a moderately high degree of interconnection among the accounts, and the velocity of 3.3965 suggests transactions are occurring at a notably elevated rate, both hallmarks of coordinated activity. However, the burst concentration is low at 0.0769, meaning activity is spread out rather than spiking at a single moment, which slightly reduces the urgency of a flash-fraud interpretation. The cluster transacts exclusively in a single product code across all 26 transactions, which is consistent with a targeted abuse pattern such as gift card or voucher fraud, though the presence of 5 distinct email domains and an email-to-uid ratio of 0.3333 suggests some effort to diversify identity signals rather than relying on a single domain. Notably, the cluster carries a prior fraud share of 0.0, meaning none of the constituent identities have previously flagged fraud history, which could indicate either genuinely clean accounts or a fresh synthetic identity ring not yet caught — the combination of high interconnection, single product focus, and moderate velocity makes this cluster ambiguous but worth elevated review.

### cluster-29 (source=llm)

- Priority score: 102.1290
- Member uids: ['11106_204_101', '15066_204_101', '4151_204_101']
- Evidence: `{'cluster_size_uids': 3, 'cluster_txn_count': 7, 'cluster_edge_density': 1.0, 'cluster_velocity': 7.0, 'cluster_amt_cv': 0.5211, 'cluster_burst_concentration': 0.1429, 'cluster_email_uid_ratio': 0.6667, 'cluster_prior_fraud_share': 1.0, 'total_transaction_amount': 1512.9399, 'mean_transaction_amount': 216.1343, 'distinct_product_codes': 2, 'distinct_email_domains': 2}`

> All three uids in this cluster have prior fraud flags, giving a cluster_prior_fraud_share of 1.0, which is the single strongest indicator of coordinated abuse here. The cluster is fully connected with an edge density of 1.0, meaning every uid shares a linking attribute with every other, and the cluster_email_uid_ratio of 0.6667 suggests that fewer distinct emails than uids are in use — consistent with identity reuse across accounts. Transaction activity is concentrated across only 2 distinct product codes and 2 distinct email domains, pointing to a narrow, focused pattern rather than organic varied behaviour. Taken together, the universal prior fraud history combined with full connectivity and limited email and product diversity makes this cluster a high-confidence coordinated fraud ring warranting immediate review.

## Limitation, stated for README.md

*(Moved here from README.md's Limitations section during a README restructure -- nothing below is a new claim. This section, like the rest of this file, is regenerated by every `python -m src.run_pipeline` run -- run_pipeline.py's write_investigator_eval doesn't know about it and silently drops it each time; restored here with the claim count updated to match the LLM output actually captured above, rather than the stale count from the run this section was first written against -- see DEVLOG.md.)*

**The groundedness result is one run of 30 clusters.** The investigator layer's hard-number-grounding rule measured 100% (0 of 176 extracted claims ungrounded, this run) against real `claude-sonnet-4-6` output, as measured above -- but that's one clean run, not a permanent guarantee. The check should keep running on every future re-run, not be treated as settled.

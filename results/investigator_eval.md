# Investigator evaluation

ANTHROPIC_API_KEY was set when this ran. Explanation sources (derived from the actual run's `source` field, not assumed from whether a key was present): {'llm': 30}.

**All 30 of 30 explanations used the real LLM path.**

Evaluated 30 clusters, selected to span the risk range: sorted all multi-uid clusters (2+ members) by cluster_prior_fraud_share, then took 30 evenly-spaced percentile points across that sorted list (not just the top 30 riskiest).

## Groundedness

**On the 30 real LLM explanation(s): 0 of 182 numeric claims ungrounded -- groundedness rate 100.00%.** This is the number that actually measures claude-sonnet-4-6's behavior under the prompt's hard rule. Reported honestly whatever it is -- a rate below 100% is a finding about the model's behavior, not something to fix by loosening the claim extractor.

- Fallback-only groundedness (for reference; expected ~100% since it's built by directly formatting evidence values verbatim): 0 of 0 claims ungrounded (no fallback explanations this run)
- Combined (LLM + fallback) across all 30 explanations: 0 of 182 claims ungrounded (100.00%). Not the LLM's groundedness rate when fallback explanations are mixed in -- a trivially-grounded template dilutes or inflates the real number, which is why it's reported separately above.

(A claim counts as grounded if it matches some evidence value exactly, at any rounding from 0-4 decimal places, or as that value expressed as a percentage.)

No ungrounded claims found among the LLM explanations.

## 3 example explanations (lowest, median, highest priority)

### cluster-13 (source=llm)

- Priority score: 2.0090
- Member uids: ['12667_330_105', '12667_330_21', '12667_330_49', '15365_330_-23', '2748_330_13', '5031_330_125', '5031_330_13']
- Evidence: `{'cluster_size_uids': 7, 'cluster_txn_count': 11, 'cluster_edge_density': 1.0, 'cluster_velocity': 0.0978, 'cluster_amt_cv': 1.0557, 'cluster_burst_concentration': 0.0909, 'cluster_email_uid_ratio': 0.1429, 'cluster_prior_fraud_share': 0.0, 'total_transaction_amount': 6085.0, 'mean_transaction_amount': 468.0769, 'distinct_product_codes': 2, 'distinct_email_domains': 1}`

> This cluster of 7 linked identities shows a fully connected graph (edge density of 1.0), meaning every identity in the cluster shares a linking attribute with every other, which is structurally unusual and more consistent with coordinated account creation than organic customer overlap. The email signal is particularly striking: a ratio of 0.1429 combined with only 1 distinct email domain across all identities suggests the accounts are concentrated around a single domain, a pattern commonly seen in synthetic or bulk-registered account rings. Transaction activity is spread across only 2 distinct product codes, indicating a narrow and potentially targeted purchase focus, and the amount coefficient of variation of 1.0557 reflects high variability in individual transaction sizes, which can indicate probing or limit-testing behavior. On the other hand, the cluster's prior fraud share is 0.0, meaning none of these identities have been flagged in historical fraud records, and burst concentration is low at 0.0909, suggesting transactions were not crammed into a single short window. Overall, the structural linkage and email concentration are the most concerning signals here, warranting closer manual review, though the absence of prior fraud history and low burst activity leave the risk assessment somewhat ambiguous.

### cluster-7 (source=llm)

- Priority score: 3.3690
- Member uids: ['10086_520_10', '13832_520_11', '13832_520_6', '13832_520_8', '15885_520_-343', '15885_520_10', '15885_520_12', '15885_520_8', '15885_520_9', '17782_520_7', '2256_520_-363', '3154_520_10', '4461_520_7', '8755_520_-414', '9633_520_-56']
- Evidence: `{'cluster_size_uids': 15, 'cluster_txn_count': 26, 'cluster_edge_density': 0.5238, 'cluster_velocity': 3.3965, 'cluster_amt_cv': 0.921, 'cluster_burst_concentration': 0.0769, 'cluster_email_uid_ratio': 0.3333, 'cluster_prior_fraud_share': 0.0, 'total_transaction_amount': 1129.309, 'mean_transaction_amount': 43.435, 'distinct_product_codes': 1, 'distinct_email_domains': 5}`

> This cluster of 15 linked identities shows several features consistent with coordinated abuse, though some signals are ambiguous. The edge density of 0.5238 indicates that a high proportion of these identities share linking attributes — devices, addresses, or payment instruments — well beyond what would be expected from coincidental overlap. All transactions fall under a single product code, suggesting the group is focused on one specific offering, which is a common pattern in promotional or credit abuse rings. The cluster velocity of 3.3965 and the relatively low email-to-uid ratio of 0.3333 further suggest that a small number of email identities are being shared or rotated across a larger pool of accounts, a hallmark of synthetic identity clustering. Somewhat mitigating factors include a cluster burst concentration of 0.0769, indicating activity is not tightly packed into a single time window, and a prior fraud share of 0.0, meaning none of these identities have confirmed fraud history — leaving the overall read ambiguous but warranting closer review given the structural linkage and product concentration.

### cluster-29 (source=llm)

- Priority score: 102.1290
- Member uids: ['11106_204_101', '15066_204_101', '4151_204_101']
- Evidence: `{'cluster_size_uids': 3, 'cluster_txn_count': 7, 'cluster_edge_density': 1.0, 'cluster_velocity': 7.0, 'cluster_amt_cv': 0.5211, 'cluster_burst_concentration': 0.1429, 'cluster_email_uid_ratio': 0.6667, 'cluster_prior_fraud_share': 1.0, 'total_transaction_amount': 1512.9399, 'mean_transaction_amount': 216.1343, 'distinct_product_codes': 2, 'distinct_email_domains': 2}`

> All three identities in this cluster have prior fraud flags, giving a cluster_prior_fraud_share of 1.0 — every uid here has a known fraud history, which is the single most damning signal in the evidence. The cluster is fully interconnected (edge_density of 1.0), meaning each identity shares a linking attribute with every other, consistent with a coordinated synthetic or mule network rather than coincidental overlap. Activity is concentrated across only 2 distinct product codes and 2 distinct email domains, suggesting a narrow, scripted attack pattern rather than organic consumer behavior. The cluster_email_uid_ratio of 0.6667 indicates that email addresses are being reused across identities, a common technique in account-farming abuse. Taken together, the universal fraud history, full interconnection, and constrained behavioral footprint strongly point to deliberate coordinated fraud rather than an ambiguous or benign grouping.

## Limitation, stated for README.md

*(Moved here from README.md's Limitations section during a README restructure -- nothing below is a new claim.)*

**The groundedness result is one run of 30 clusters.** The investigator layer's hard-number-grounding rule measured 100% (0 of 182 extracted claims ungrounded) against real `claude-sonnet-4-6` output, as measured above -- but that's one clean run, not a permanent guarantee. The check should keep running on every future re-run, not be treated as settled.

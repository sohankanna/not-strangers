# Cluster case studies

**There is no ground truth for "this is a real coordinated ring" anywhere
in this dataset** (see README.md's Limitations section) -- IEEE-CIS has
chargeback-derived fraud labels per transaction, not ring labels. Everything
below is qualitative inspection of what's actually in a cluster: read it as
"here's what the evidence looks like up close," not as a validated
confirmation that any of these are real rings. Produced by
`scripts/case_studies.py` (raw numbers) plus manual write-up (the judgment
calls a script shouldn't make for you) on the 3 highest-priority clusters
from the train graph, ranked by investigator.py's priority_score.

All three happen to be small, 2-uid clusters -- not cherry-picked to be
small; the priority heuristic weights `cluster_prior_fraud_share` heaviest
(100x), and a 2-uid cluster where both members are 100% fraud maxes that
term out, so small-and-fully-fraudulent clusters dominate the top of this
particular ranking. That's worth knowing about the ranking, not just the
clusters.

## Case 1: cluster-1494 (rank 1) -- a plausible true positive

- **Members:** `12160_325_79`, `12160_325_83` -- same card1 (12160) and
  addr1 (325), different origin_day (79 vs 83). Two persistent identities
  for what looks like the same card/address pair at two different
  first-seen cohorts.
- **Linked by:** `card_bank_addr` (shared card3=223, card5=224, addr1=325).
  This matters because card3=150/card5=226 dominate the dataset (see
  DEVLOG.md's Task 1 entry) -- 223/224 is an uncommon pairing, so all
  three matching together is a rarer, more specific coincidence than it
  would be for the dominant pair. Not linked by device or email: DeviceInfo
  here is "Windows" and the email domain is (325.0, gmail.com) -- both are
  in the hub-guard's excluded-values list (Windows: 18,535 uid
  appearances; (325.0, gmail.com): 5,434), so neither contributed an edge.
- **Size:** 2 uids, 3 transactions.
- **Amounts:** $100, $100, $300 (mean $166.67, total $500).
- **Time:** spans 4.29 days (uid `...79`'s two transactions, then
  `...83`'s single transaction ~4 days later).
- **Fraud labels: 3 of 3 transactions are isFraud=1 (100%).** All three
  are ProductCD=H.
- **Why the features fired:** `cluster_prior_fraud_share=1.0` (both
  members' entire histories are fraud), `cluster_burst_concentration=0.67`
  (2 of 3 transactions land in the same 3-minute window), single
  ProductCD, single email domain. Everything about this cluster is
  homogeneous and fully fraud-labeled.
- **Read:** the linking evidence (an uncommon card3/card5 pairing, not a
  generic hub value) plus a fully fraudulent, homogeneous transaction
  history makes this a plausible true positive -- two identities that are
  very likely the same underlying card/operator.

## Case 2: cluster-1488 (rank 2, tied) -- a plausible true positive, different mechanism

- **Members:** `12557_441_76`, `4141_310_76` -- different card1, different
  addr1, same origin_day (76) coincidentally.
- **Linked by:** `device_info` only -- both transactions share the exact
  device string `SM-G950F Build/NRD90M` (a specific Samsung Galaxy S8
  model/build identifier, not a generic OS name). Not linked by
  address+email (addr1 differs: 441 vs 310) or card/bank/address (card5
  differs: 102 vs 226).
- **Size:** 2 uids, 2 transactions (1 each).
- **Amounts:** $300 and $300 -- identical.
- **Time:** the two transactions are **284 seconds (4.7 minutes) apart.**
- **Fraud labels: 2 of 2 transactions are isFraud=1 (100%).** Both
  ProductCD=R.
- **Why the features fired:** `cluster_prior_fraud_share=1.0`,
  `cluster_amt_cv=0.0` (identical amounts -- zero variance),
  `cluster_burst_concentration=0.5`.
- **Read:** this is the most specific evidence of the three cases -- a
  precise device build string, not a locale or OS name, linking two
  otherwise-unconnected card+address combinations, with near-simultaneous,
  identical-amount, both-fraud transactions. If this project were going to
  point at one case as "this looks like the same operator running two
  identities," this is it.

## Case 3: cluster-1505 (rank 3, tied) -- ambiguous; the weakest of the three, said plainly

- **Members:** `1546_204_85`, `1735_325_85` -- different card1, different
  addr1, same origin_day (85) coincidentally.
- **Linked by:** `device_info` only -- both transactions have DeviceInfo
  = **"en-gb"**. Not linked by address+email (addr1 differs: 325 vs 204)
  or card/bank/address (card5 differs: 195 vs 226).
- **Size:** 2 uids, 2 transactions (1 each).
- **Amounts:** $150 and $150 -- identical.
- **Time:** the two transactions are **223 seconds (3.7 minutes) apart.**
- **Fraud labels: 2 of 2 transactions are isFraud=1 (100%).** Both
  ProductCD=H.
- **Why the features fired:** identical to Case 2's shape --
  `cluster_prior_fraud_share=1.0`, `cluster_amt_cv=0.0`,
  `cluster_burst_concentration=0.5` -- because the priority heuristic and
  the underlying features can't distinguish *what kind* of DeviceInfo value
  did the linking, only that one did.
- **This is the case worth flagging as weaker, possibly a false positive
  on the linkage (not necessarily on the fraud labels themselves).**
  "en-gb" reads as a locale/language string, not a device fingerprint --
  it's the kind of value real, unrelated UK-locale users could plausibly
  share by chance. It's below the max_degree=20 hub threshold in this
  slice of data, so it wasn't excluded, but that's a threshold effect, not
  evidence that "en-gb" is actually rare or meaningful the way a specific
  device build string (Case 2) or an uncommon card3/card5 pair (Case 1) is.
  The *behavioral* pattern here (identical amount, ~4 minutes apart, both
  fraud) is genuinely suspicious and matches Case 2's shape closely -- so
  this isn't confidently a false positive either. Presented honestly as
  ambiguous: a real link is plausible, but the specific evidence that
  created it is the weakest of the three, and a system built on this
  linkage rule should not treat "shared en-gb" with the same confidence as
  "shared specific device model."

## What this does and doesn't show

Two of three cases (1 and 2) have linking evidence that looks genuinely
specific -- an uncommon card3/card5/addr1 pairing and a specific phone
build string, respectively -- alongside fully-fraudulent, tightly-timed
transaction histories. Case 3 shows the same behavioral pattern but on
weaker linking evidence (a generic-looking locale string), and is called
out here rather than swapped for a cleaner-looking example. This is
qualitative inspection of 3 clusters out of the 1,567 multi-uid
clusters in the train graph -- it is evidence that the linkage rules *can*
produce specific, defensible connections, not a claim about their overall
precision across the whole population. That claim would need actual ring
labels, which don't exist for this dataset.

## Limitation, stated for README.md

*(Moved here from README.md's Limitations section during a README restructure -- nothing below is a new claim.)*

**Clusters have no ground truth.** There is no "this is a real coordinated ring" label anywhere in this dataset. Everything reported in this project is *feature lift* -- whether cluster-derived features improve a fraud classifier's ranking and cost metrics on a temporal holdout (see results/ablation.md) -- not *ring-detection accuracy*. This report inspects the 3 highest-priority clusters qualitatively and includes one flagged explicitly as ambiguous/possibly a false positive on the linkage (Case 3 above), rather than swapped for a cleaner-looking example.

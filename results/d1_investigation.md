# D1 / origin_day investigation

Follow-up to the negative-origin_day quirk flagged in `results/uid_validation.md` and `DEVLOG.md`. Read-only: `src/entities.py` is unchanged, the uid derivation is unchanged. Computed on the real `train_transaction.csv` (590,540 rows) by `scripts/investigate_d1.py`.

Row-level split by origin_day sign: **179,300** negative (30.36%), **408,917** positive (69.24%), **1,054** exactly zero (0.18%), **1,269** NaN/no-uid (0.21%). The comparisons below are strictly negative (<0) vs. strictly positive (>0); zero and NaN rows are excluded from both groups and reported here only for completeness.

## 1. Are negative-origin_day rows different in kind?

- Fraud rate: negative = **1.60%**, positive = **4.33%**
- Share with an identity record: negative = **7.35%**, positive = **31.97%**

ProductCD distribution (share of rows):

| ProductCD | negative | positive |
|---|---:|---:|
| C | 4.32% | 14.80% |
| H | 0.05% | 8.04% |
| R | 1.63% | 8.50% |
| S | 1.93% | 2.00% |
| W | 92.07% | 66.66% |

TransactionAmt distribution:

| stat | negative | positive |
|---|---:|---:|
| count | 179,300.00 | 408,917.00 |
| mean | 116.97 | 142.80 |
| median | 72.95 | 67.95 |
| std | 150.85 | 268.10 |
| p10 | 29.00 | 25.50 |
| p90 | 226.00 | 300.00 |
| max | 5,191.00 | 31,937.39 |

## 2. D1 distribution within each group

By construction, D1 cannot be null in either group here: origin_day is `day - D1`, so a null D1 makes origin_day NaN, which is excluded from both the negative and positive groups (it falls in the 1,269-row NaN bucket above). Confirmed: n_null is 0 in both groups below, as expected -- the open question is whether D1 is zero-heavy or otherwise differently shaped between the two groups.

| stat | negative-origin_day D1 | positive-origin_day D1 |
|---|---:|---:|
| count | 179,300.00 | 408,917.00 |
| n_null | 0.00 | 0.00 |
| n_zero | 0.00 | 280,130.00 |
| share_zero | 0.00% | 68.51% |
| mean | 282.47 | 11.95 |
| median | 252.00 | 0.00 |
| p90 | 535.00 | 46.00 |
| max | 640.00 | 181.00 |

## 3. Does label purity hold within each group?

origin_day is embedded in the uid string itself, so every transaction sharing a uid necessarily shares that uid's origin_day sign (confirmed programmatically: every uid's rows have exactly one distinct origin_day value).

- Multi-transaction (2+) uids with negative origin_day: **24,638**, weighted label purity = **99.24%**
- Multi-transaction (2+) uids with positive origin_day: **58,774**, weighted label purity = **96.61%**

Purity differs meaningfully between the two groups -- the sign of origin_day is NOT just cosmetic; it correlates with how reliable the resulting uid is as a label-consistent identity.

## 4. Collision check: the 20 largest uids

Distinct-value counts of card2, card3, card5 and P_emaildomain (nulls excluded from the count) among each uid's own rows. A genuine single client should be near-constant (1, or close to it, accounting for real missingness) on all four.

| uid | size | fraud_rate | distinct_card2 | distinct_card3 | distinct_card5 | distinct_P_emaildomain |
|---|---:|---:|---:|---:|---:|---:|
| 15775_330_129 | 1,414 | 0.00% | 1 | 1 | 1 | 0 |
| 9500_126_-85 | 446 | 0.00% | 1 | 1 | 1 | 2 |
| 8900_231_-60 | 232 | 0.00% | 1 | 1 | 1 | 1 |
| 8528_387_-159 | 215 | 0.00% | 1 | 1 | 1 | 1 |
| 12741_143_-202 | 196 | 0.00% | 1 | 1 | 1 | 3 |
| 7207_204_-465 | 191 | 0.00% | 1 | 1 | 1 | 2 |
| 13597_191_-48 | 148 | 0.00% | 1 | 1 | 1 | 2 |
| 4121_476_-8 | 142 | 0.00% | 1 | 1 | 1 | 1 |
| 12695_325_-342 | 123 | 73.17% | 1 | 1 | 1 | 2 |
| 2884_204_-32 | 118 | 0.00% | 1 | 1 | 1 | 3 |
| 7826_325_-33 | 118 | 0.00% | 1 | 1 | 1 | 2 |
| 2803_251_-320 | 114 | 0.00% | 1 | 1 | 1 | 1 |
| 15813_441_-22 | 110 | 0.00% | 1 | 1 | 1 | 2 |
| 9323_191_-50 | 110 | 0.00% | 1 | 1 | 1 | 1 |
| 3898_181_-188 | 108 | 0.00% | 1 | 1 | 1 | 1 |
| 2845_315_-10 | 105 | 0.00% | 1 | 1 | 1 | 2 |
| 8135_325_49 | 97 | 0.00% | 1 | 1 | 1 | 2 |
| 1675_330_71 | 97 | 0.00% | 1 | 1 | 1 | 1 |
| 2803_264_-345 | 92 | 0.00% | 1 | 1 | 1 | 1 |
| 7207_204_-128 | 91 | 0.00% | 1 | 1 | 1 | 1 |

card2 / card3 / card5 show exactly one distinct (non-null) value across all 20 of these uids -- unsurprising, since card2/card3/card5 are sub-attributes of the same physical card as card1 (e.g. issuing bank/country codes), not independent identifiers, so fixing card1 largely fixes them too. They add little discriminating power for this check.

**10 of the 20 largest uids show more than one distinct value on P_emaildomain.** That means card1+addr1+origin_day is not unique to a single cardholder for these uids -- the uid is merging multiple distinct clients who happen to share a card1, an address, and a first-seen day. Label purity being high (section 3, and the 98.53%/97.61% figures in uid_validation.md) shows these merges are usually *label-consistent* (the merged clients mostly share the same isFraud outcome), which is a much weaker claim than the uid being *correct* -- i.e. actually identifying one physical client. This is stability without correctness: safe enough to use as a clustering key for label-consistent aggregation, but it should not be presented as "the client" in an investigator-facing explanation without that caveat.

## Limitation, stated for README.md

*(Moved here from README.md's Limitations section during a README restructure -- nothing below is a new claim.)*

**The uid over-merges.** `card1_addr1_origin_day` is a stable, highly label-pure identifier (98.53% of multi-transaction uids are label-pure, weighted 97.61% -- see results/uid_validation.md), but stability is not the same as correctness. The collision check above found `P_emaildomain` varying within 10 of the 20 largest uids -- distinct people are demonstrably sharing a uid. Treated in this project as signal (coordinated abuse), not an error to fix, but "cluster" here is not a verified single-person identity.

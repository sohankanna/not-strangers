"""Investigate the negative-origin_day quirk flagged in DEVLOG.md and
results/uid_validation.md: ~30% of rows have origin_day = day - D1 < 0,
meaning D1 exceeds the transaction's own elapsed-day count. This script does
not change src/entities.py -- it's read-only investigation of what that
quirk means for the uid derivation's reliability.

Writes results/d1_investigation.md. Not wired into run_pipeline.py or the
Makefile.

Usage:
    python scripts/investigate_d1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.entities import extract_entity_keys, resolve_entities

TRANSACTION_PATH = REPO_ROOT / "data" / "train_transaction.csv"
IDENTITY_PATH = REPO_ROOT / "data" / "train_identity.csv"
RESULTS_DIR = REPO_ROOT / "results"

TRANSACTION_USECOLS = [
    "TransactionID",
    "TransactionDT",
    "D1",
    "card1",
    "addr1",
    "isFraud",
    "ProductCD",
    "TransactionAmt",
    "card2",
    "card3",
    "card5",
    "P_emaildomain",
]

TOP_N_UIDS = 20


def _amt_stats(s: pd.Series) -> dict[str, float]:
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def _d1_stats(s: pd.Series) -> dict[str, float]:
    return {
        "count": int(s.count()),
        "n_null": int(s.isna().sum()),
        "n_zero": int((s == 0).sum()),
        "share_zero": float((s == 0).mean()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def main() -> None:
    df = pd.read_csv(TRANSACTION_PATH, usecols=TRANSACTION_USECOLS)
    identity_ids = pd.read_csv(IDENTITY_PATH, usecols=["TransactionID"])[
        "TransactionID"
    ]
    identity_id_set = set(identity_ids.to_numpy())

    keys = extract_entity_keys(df)  # indexed by TransactionID
    uid = resolve_entities(df)  # indexed by TransactionID

    df = df.set_index("TransactionID")  # realign with uid/keys (see DEVLOG)
    has_identity = pd.Series(
        df.index.isin(identity_id_set), index=df.index, name="has_identity"
    )
    origin_day = keys["origin_day"]

    n = len(df)
    neg_mask = origin_day < 0
    pos_mask = origin_day > 0
    zero_mask = origin_day == 0
    nan_mask = origin_day.isna()

    n_neg = int(neg_mask.sum())
    n_pos = int(pos_mask.sum())
    n_zero = int(zero_mask.sum())
    n_nan = int(nan_mask.sum())
    assert n_neg + n_pos + n_zero + n_nan == n

    # --- Q1: negative vs positive origin_day rows ----------------------------
    fraud_rate_neg = float(df.loc[neg_mask, "isFraud"].mean())
    fraud_rate_pos = float(df.loc[pos_mask, "isFraud"].mean())

    product_neg = df.loc[neg_mask, "ProductCD"].value_counts(normalize=True)
    product_pos = df.loc[pos_mask, "ProductCD"].value_counts(normalize=True)
    all_products = sorted(set(product_neg.index) | set(product_pos.index))

    amt_neg = _amt_stats(df.loc[neg_mask, "TransactionAmt"])
    amt_pos = _amt_stats(df.loc[pos_mask, "TransactionAmt"])

    identity_share_neg = float(has_identity.loc[neg_mask].mean())
    identity_share_pos = float(has_identity.loc[pos_mask].mean())

    # --- Q2: D1 distribution within each group --------------------------------
    d1_neg = _d1_stats(df.loc[neg_mask, "D1"])
    d1_pos = _d1_stats(df.loc[pos_mask, "D1"])

    # --- Q3: weighted label purity split by uid's origin_day sign -------------
    valid = uid.notna()
    valid_df = df.loc[valid, ["isFraud"]].copy()
    valid_df["uid"] = uid.loc[valid]
    valid_df["origin_day"] = origin_day.loc[valid]

    grouped = valid_df.groupby("uid")
    sizes = grouped.size()
    nunique_fraud = grouped["isFraud"].nunique()
    uid_origin_day_nunique = grouped["origin_day"].nunique()
    uid_origin_day = grouped["origin_day"].first()

    # Sanity check: origin_day is embedded in the uid string itself, so every
    # row sharing a uid must share the same origin_day. Confirm rather than
    # assume.
    assert (uid_origin_day_nunique == 1).all(), (
        "a uid contains rows with different origin_day values -- "
        "the embedding assumption this section relies on does not hold"
    )

    multi_mask = sizes >= 2
    pure_mask = nunique_fraud == 1

    neg_uid_mask = multi_mask & (uid_origin_day < 0)
    pos_uid_mask = multi_mask & (uid_origin_day > 0)

    n_multi_neg_uids = int(neg_uid_mask.sum())
    n_multi_pos_uids = int(pos_uid_mask.sum())

    purity_weighted_neg = (
        float(sizes[neg_uid_mask & pure_mask].sum() / sizes[neg_uid_mask].sum())
        if n_multi_neg_uids
        else float("nan")
    )
    purity_weighted_pos = (
        float(sizes[pos_uid_mask & pure_mask].sum() / sizes[pos_uid_mask].sum())
        if n_multi_pos_uids
        else float("nan")
    )

    # --- Q4: collision check on the 20 largest uids ---------------------------
    top_uids = sizes.sort_values(ascending=False).head(TOP_N_UIDS)

    collision_rows = []
    for uid_value, size in top_uids.items():
        member_ids = valid_df.index[valid_df["uid"] == uid_value]
        rows = df.loc[member_ids]
        fraud_rate = float(rows["isFraud"].mean())
        collision_rows.append(
            {
                "uid": uid_value,
                "size": int(size),
                "fraud_rate": fraud_rate,
                "distinct_card2": int(rows["card2"].nunique(dropna=True)),
                "distinct_card3": int(rows["card3"].nunique(dropna=True)),
                "distinct_card5": int(rows["card5"].nunique(dropna=True)),
                "distinct_P_emaildomain": int(
                    rows["P_emaildomain"].nunique(dropna=True)
                ),
            }
        )
    collision_table = pd.DataFrame(collision_rows)
    collision_cols = [
        "distinct_card2",
        "distinct_card3",
        "distinct_card5",
        "distinct_P_emaildomain",
    ]
    over_merged = collision_table[(collision_table[collision_cols] > 1).any(axis=1)]
    varying_cols = [c for c in collision_cols if (collision_table[c] > 1).any()]
    constant_cols = [c for c in collision_cols if c not in varying_cols]

    # --- Write the report ------------------------------------------------------
    lines: list[str] = []
    lines.append("# D1 / origin_day investigation")
    lines.append("")
    lines.append(
        "Follow-up to the negative-origin_day quirk flagged in "
        "`results/uid_validation.md` and `DEVLOG.md`. Read-only: "
        "`src/entities.py` is unchanged, the uid derivation is unchanged. "
        f"Computed on the real `train_transaction.csv` ({n:,} rows) by "
        "`scripts/investigate_d1.py`."
    )
    lines.append("")
    lines.append(
        f"Row-level split by origin_day sign: **{n_neg:,}** negative "
        f"({n_neg / n:.2%}), **{n_pos:,}** positive ({n_pos / n:.2%}), "
        f"**{n_zero:,}** exactly zero ({n_zero / n:.2%}), **{n_nan:,}** "
        f"NaN/no-uid ({n_nan / n:.2%}). The comparisons below are strictly "
        "negative (<0) vs. strictly positive (>0); zero and NaN rows are "
        "excluded from both groups and reported here only for completeness."
    )
    lines.append("")

    lines.append("## 1. Are negative-origin_day rows different in kind?")
    lines.append("")
    lines.append(f"- Fraud rate: negative = **{fraud_rate_neg:.2%}**, positive = **{fraud_rate_pos:.2%}**")
    lines.append(f"- Share with an identity record: negative = **{identity_share_neg:.2%}**, positive = **{identity_share_pos:.2%}**")
    lines.append("")
    lines.append("ProductCD distribution (share of rows):")
    lines.append("")
    lines.append("| ProductCD | negative | positive |")
    lines.append("|---|---:|---:|")
    for p in all_products:
        lines.append(
            f"| {p} | {product_neg.get(p, 0.0):.2%} | {product_pos.get(p, 0.0):.2%} |"
        )
    lines.append("")
    lines.append("TransactionAmt distribution:")
    lines.append("")
    lines.append("| stat | negative | positive |")
    lines.append("|---|---:|---:|")
    for k in ["count", "mean", "median", "std", "p10", "p90", "max"]:
        lines.append(f"| {k} | {amt_neg[k]:,.2f} | {amt_pos[k]:,.2f} |")
    lines.append("")

    lines.append("## 2. D1 distribution within each group")
    lines.append("")
    lines.append(
        "By construction, D1 cannot be null in either group here: origin_day "
        "is `day - D1`, so a null D1 makes origin_day NaN, which is excluded "
        "from both the negative and positive groups (it falls in the "
        f"{n_nan:,}-row NaN bucket above). Confirmed: n_null is 0 in both "
        "groups below, as expected -- the open question is whether D1 is "
        "zero-heavy or otherwise differently shaped between the two groups."
    )
    lines.append("")
    lines.append("| stat | negative-origin_day D1 | positive-origin_day D1 |")
    lines.append("|---|---:|---:|")
    for k in ["count", "n_null", "n_zero", "share_zero", "mean", "median", "p90", "max"]:
        v_neg, v_pos = d1_neg[k], d1_pos[k]
        if k == "share_zero":
            lines.append(f"| {k} | {v_neg:.2%} | {v_pos:.2%} |")
        else:
            lines.append(f"| {k} | {v_neg:,.2f} | {v_pos:,.2f} |")
    lines.append("")

    lines.append("## 3. Does label purity hold within each group?")
    lines.append("")
    lines.append(
        "origin_day is embedded in the uid string itself, so every "
        "transaction sharing a uid necessarily shares that uid's origin_day "
        "sign (confirmed programmatically: every uid's rows have exactly one "
        "distinct origin_day value)."
    )
    lines.append("")
    lines.append(f"- Multi-transaction (2+) uids with negative origin_day: **{n_multi_neg_uids:,}**, weighted label purity = **{purity_weighted_neg:.2%}**")
    lines.append(f"- Multi-transaction (2+) uids with positive origin_day: **{n_multi_pos_uids:,}**, weighted label purity = **{purity_weighted_pos:.2%}**")
    lines.append("")
    if abs(purity_weighted_neg - purity_weighted_pos) < 0.02:
        purity_verdict = (
            "Purity holds in both groups at essentially the same level -- "
            "the uid derivation's stability does not depend on which side "
            "of zero origin_day happens to land on. D1's inconsistent epoch "
            "affects the *literal value* of origin_day but not the "
            "derivation's usefulness as a stable per-client key."
        )
    else:
        purity_verdict = (
            "Purity differs meaningfully between the two groups -- the sign "
            "of origin_day is NOT just cosmetic; it correlates with how "
            "reliable the resulting uid is as a label-consistent identity."
        )
    lines.append(purity_verdict)
    lines.append("")

    lines.append("## 4. Collision check: the 20 largest uids")
    lines.append("")
    lines.append(
        "Distinct-value counts of card2, card3, card5 and P_emaildomain "
        "(nulls excluded from the count) among each uid's own rows. A "
        "genuine single client should be near-constant (1, or close to it, "
        "accounting for real missingness) on all four."
    )
    lines.append("")
    lines.append("| uid | size | fraud_rate | distinct_card2 | distinct_card3 | distinct_card5 | distinct_P_emaildomain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in collision_table.iterrows():
        lines.append(
            f"| {row['uid']} | {row['size']:,} | {row['fraud_rate']:.2%} | "
            f"{row['distinct_card2']} | {row['distinct_card3']} | "
            f"{row['distinct_card5']} | {row['distinct_P_emaildomain']} |"
        )
    lines.append("")

    n_over_merged = len(over_merged)
    if constant_cols:
        lines.append(
            f"{' / '.join(c.replace('distinct_', '') for c in constant_cols)} "
            f"show exactly one distinct (non-null) value across all "
            f"{TOP_N_UIDS} of these uids -- unsurprising, since card2/card3/"
            "card5 are sub-attributes of the same physical card as card1 "
            "(e.g. issuing bank/country codes), not independent identifiers, "
            "so fixing card1 largely fixes them too. They add little "
            "discriminating power for this check."
        )
        lines.append("")
    if n_over_merged:
        varying_label = " / ".join(c.replace("distinct_", "") for c in varying_cols)
        lines.append(
            f"**{n_over_merged} of the {TOP_N_UIDS} largest uids show more "
            f"than one distinct value on {varying_label}.** That means "
            "card1+addr1+origin_day is not unique to a single cardholder for "
            "these uids -- the uid is merging multiple distinct clients who "
            "happen to share a card1, an address, and a first-seen day. "
            "Label purity being high (section 3, and the 98.53%/97.61% "
            "figures in uid_validation.md) shows these merges are usually "
            "*label-consistent* (the merged clients mostly share the same "
            "isFraud outcome), which is a much weaker claim than the uid "
            "being *correct* -- i.e. actually identifying one physical "
            "client. This is stability without correctness: safe enough to "
            "use as a clustering key for label-consistent aggregation, but "
            "it should not be presented as \"the client\" in an "
            "investigator-facing explanation without that caveat."
        )
    else:
        lines.append(
            f"None of the {TOP_N_UIDS} largest uids show more than one "
            "distinct value on any of card2/card3/card5/P_emaildomain -- no "
            "evidence of over-merging among the largest clusters."
        )
    lines.append("")

    (RESULTS_DIR / "d1_investigation.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote results/d1_investigation.md")


if __name__ == "__main__":
    main()

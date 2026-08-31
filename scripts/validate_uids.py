"""Compute uid derivation statistics on the real IEEE-CIS training data and
write results/uid_validation.md and results/uid_size_distribution.png.

Standalone analysis script -- not wired into run_pipeline.py or Makefile.
Only reads train_transaction.csv (the uid derivation needs TransactionID,
TransactionDT, D1, card1, addr1, isFraud; none of those live in the identity
file, and test_transaction.csv has no isFraud labels to check purity against).

Usage:
    python scripts/validate_uids.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.entities import resolve_entities

DATA_PATH = REPO_ROOT / "data" / "train_transaction.csv"
RESULTS_DIR = REPO_ROOT / "results"
USECOLS = ["TransactionID", "TransactionDT", "D1", "card1", "addr1", "isFraud"]


def main() -> None:
    df = pd.read_csv(DATA_PATH, usecols=USECOLS)
    n = len(df)

    uid = resolve_entities(df)

    # --- Section 1: valid vs NaN uid, broken down by cause -----------------
    card1_null = df["card1"].isna()
    addr1_null = df["addr1"].isna()
    d1_null = df["D1"].isna()

    n_card1_null = int(card1_null.sum())
    n_d1_only = int((d1_null & ~addr1_null).sum())
    n_addr1_only = int((addr1_null & ~d1_null).sum())
    n_both = int((d1_null & addr1_null).sum())
    n_nan_uid = int(uid.isna().sum())
    n_valid_uid = n - n_nan_uid

    # uid is indexed by TransactionID (per src.entities); realign df the same
    # way so downstream boolean masks against `uid` line up positionally.
    df = df.set_index("TransactionID")

    # --- Section 2: transactions per uid ------------------------------------
    counts = uid.dropna().value_counts()
    distinct_uids = len(counts)
    singleton_share = float((counts == 1).mean())
    median_count = float(counts.median())
    p90_count = float(counts.quantile(0.9))
    max_count = int(counts.max())

    # --- Section 3: label purity for uids with 2+ transactions --------------
    valid_df = df.loc[uid.notna(), ["isFraud"]].copy()
    valid_df["uid"] = uid.dropna()
    grouped = valid_df.groupby("uid")["isFraud"]
    sizes = grouped.size()
    nunique = grouped.nunique()
    fraud_rate_by_uid = grouped.mean()

    multi_mask = sizes >= 2
    multi_sizes = sizes[multi_mask]
    multi_nunique = nunique[multi_mask]
    pure_mask = multi_nunique == 1

    n_multi_uids = int(multi_mask.sum())
    pure_fraction_unweighted = float(pure_mask.mean()) if n_multi_uids else float("nan")
    pure_fraction_weighted = (
        float(multi_sizes[pure_mask].sum() / multi_sizes.sum())
        if n_multi_uids
        else float("nan")
    )

    # --- Section 4: 10 largest impure uids -----------------------------------
    impure_sizes = multi_sizes[~pure_mask].sort_values(ascending=False)
    top_impure = impure_sizes.head(10)
    top_impure_table = pd.DataFrame(
        {
            "uid": top_impure.index,
            "size": top_impure.to_numpy(),
            "fraud_rate": fraud_rate_by_uid.loc[top_impure.index].to_numpy(),
        }
    )

    # --- Section 5: fraud rate, uid'd vs NaN rows ----------------------------
    fraud_rate_uidd = float(df.loc[uid.notna(), "isFraud"].mean())
    fraud_rate_nan = float(df.loc[uid.isna(), "isFraud"].mean())

    # --- Histogram: transactions per uid, log-x -----------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(0, np.log10(max_count), 40)
    ax.hist(counts.to_numpy(), bins=bins, color="#4472C4", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Transactions per uid (log scale)")
    ax.set_ylabel("Number of uids")
    ax.set_title("Distribution of transactions per uid")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "uid_size_distribution.png", dpi=150)
    plt.close(fig)

    # --- Write the report -----------------------------------------------------
    lines: list[str] = []
    lines.append("# uid validation report")
    lines.append("")
    lines.append(
        "Computed on the real IEEE-CIS `train_transaction.csv` "
        f"({n:,} rows) by `scripts/validate_uids.py`, using the "
        "`card1_addr1_origin_day` uid derivation in `src/entities.py`. "
        "Numbers are reported as-is; the derivation was not adjusted to "
        "improve any of them."
    )
    lines.append("")

    lines.append("## 1. Valid uid vs NaN, by cause")
    lines.append("")
    lines.append(f"- Total transactions: **{n:,}**")
    lines.append(
        f"- Valid uid: **{n_valid_uid:,}** ({n_valid_uid / n:.2%}) "
        f"-- NaN uid: **{n_nan_uid:,}** ({n_nan_uid / n:.2%})"
    )
    lines.append("- NaN breakdown by cause (mutually exclusive):")
    lines.append(f"  - D1 null only: {n_d1_only:,}")
    lines.append(f"  - addr1 null only: {n_addr1_only:,}")
    lines.append(f"  - Both D1 and addr1 null: {n_both:,}")
    lines.append(
        f"  - card1 null (any row): {n_card1_null:,} -- "
        f"{'confirms the assumption that card1 is always present in this file' if n_card1_null == 0 else 'NON-ZERO: card1 nulls exist and are folded into the NaN-uid count above, but are not broken out as their own cause in the table above'}"
    )
    lines.append(
        "- Each of these three counts means: that many transactions have no "
        "usable value for the named field(s), so no uid could be formed for "
        "them and they were left as NaN rather than guessed at."
    )
    lines.append("")

    lines.append("## 2. Transactions per uid")
    lines.append("")
    lines.append(f"- Distinct uids: **{distinct_uids:,}**")
    lines.append(f"- Singleton share (uids with exactly 1 transaction): **{singleton_share:.2%}**")
    lines.append(f"- Median transactions per uid: **{median_count:g}**")
    lines.append(f"- 90th percentile: **{p90_count:g}**")
    lines.append(f"- Max: **{max_count:,}**")
    lines.append(
        "- This describes how concentrated activity is across resolved "
        "identities: a high singleton share means most uids see only one "
        "transaction, while the max/p90 show how large the biggest "
        "collapsed identities get."
    )
    lines.append("")

    lines.append("## 3. Label purity (the key number)")
    lines.append("")
    lines.append(f"- uids with 2+ transactions: **{n_multi_uids:,}**")
    lines.append(
        f"- Label-pure fraction, unweighted (isFraud uniform across all "
        f"rows of the uid): **{pure_fraction_unweighted:.2%}**"
    )
    lines.append(
        f"- Label-pure fraction, weighted by transaction count: "
        f"**{pure_fraction_weighted:.2%}**"
    )
    lines.append(
        "- The unweighted number treats every multi-transaction uid equally "
        "regardless of size; the weighted number reflects what fraction of "
        "*transactions* (not uids) sit inside a label-pure cluster -- if it "
        "differs a lot from the unweighted figure, purity depends on cluster "
        "size."
    )
    lines.append("")

    lines.append("## 4. Ten largest impure uids")
    lines.append("")
    lines.append("| uid | size | fraud_rate |")
    lines.append("|---|---:|---:|")
    for _, row in top_impure_table.iterrows():
        lines.append(f"| {row['uid']} | {int(row['size']):,} | {row['fraud_rate']:.2%} |")
    if top_impure_table.empty:
        lines.append("| (no impure multi-transaction uids found) | | |")
    lines.append("")
    lines.append(
        "- `size` is the uid's total transaction count; `fraud_rate` is the "
        "share of that uid's transactions labeled isFraud=1. A uid appears "
        "here because its rows do NOT all share the same label."
    )
    lines.append("")

    lines.append("## 5. Fraud rate: uid'd rows vs NaN rows")
    lines.append("")
    lines.append(f"- Fraud rate among rows with a valid uid: **{fraud_rate_uidd:.2%}**")
    lines.append(f"- Fraud rate among rows with a NaN uid: **{fraud_rate_nan:.2%}**")
    lines.append(
        "- If these two rates differ substantially, the transactions that "
        "fail to get a uid are not a random sample of traffic -- the null "
        "handling would be systematically excluding a different-risk slice "
        "of transactions from entity-level features, which matters for how "
        "much to trust cluster-based signals downstream."
    )
    lines.append("")

    lines.append("## Histogram")
    lines.append("")
    lines.append("![Transactions per uid distribution](uid_size_distribution.png)")
    lines.append("")

    (RESULTS_DIR / "uid_validation.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote results/uid_validation.md and results/uid_size_distribution.png")


if __name__ == "__main__":
    main()

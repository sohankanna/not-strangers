"""Resolve raw IEEE-CIS transactions into persistent client identities (uids).

uid = card1_addr1_origin_day, where:
  day        = TransactionDT // 86400   (whole days since the dataset epoch)
  origin_day = day - D1                 (the day the card was first seen)

D1 is "days since the card was first seen" (per the competition's feature
documentation), so origin_day is constant across a card's lifetime and
collapses every transaction from that card+address pair into one persistent
identity, regardless of which day within that lifetime a given transaction
happened on.

This uses only transaction-file fields (card1, addr1, D1, TransactionDT) --
consistent with the project rule that primary linkage must come from
train_transaction.csv. It intentionally does not use device fields; those
remain a secondary signal for elsewhere, not part of this uid.

Null handling: if card1, addr1 or D1 is null for a transaction, that
transaction gets no uid (NaN) rather than a fallback value or a silently
dropped row. Callers (see results/uid_validation.md) are expected to count
and report NaN uids themselves, broken down by cause.
"""

from __future__ import annotations

import pandas as pd

SECONDS_PER_DAY = 86400


def extract_entity_keys(transactions: pd.DataFrame) -> pd.DataFrame:
    """Derive the uid's component columns for each transaction.

    Args:
        transactions: Raw transaction rows with at least TransactionID,
            TransactionDT, D1, card1 and addr1 columns.

    Returns:
        A DataFrame indexed by TransactionID with columns:
          - card1, addr1: copied from transactions, unchanged (including
            any nulls in addr1).
          - day: TransactionDT // SECONDS_PER_DAY.
          - origin_day: day - D1; NaN wherever D1 is NaN.
    """
    day = transactions["TransactionDT"] // SECONDS_PER_DAY
    origin_day = day - transactions["D1"]

    return pd.DataFrame(
        {
            "card1": transactions["card1"].to_numpy(),
            "addr1": transactions["addr1"].to_numpy(),
            "day": day.to_numpy(),
            "origin_day": origin_day.to_numpy(),
        },
        index=pd.Index(
            transactions["TransactionID"].to_numpy(), name="TransactionID"
        ),
    )


def resolve_entities(transactions: pd.DataFrame) -> pd.Series:
    """Resolve transactions into persistent uids: card1_addr1_origin_day.

    Args:
        transactions: Raw transaction rows with at least TransactionID,
            TransactionDT, D1, card1 and addr1 columns.

    Returns:
        A Series named "uid", indexed by TransactionID, dtype object. NaN
        wherever card1, addr1, or origin_day (equivalently, D1) is null for
        that transaction -- no fallback value is invented, and every
        TransactionID gets an entry (valid or NaN), never a dropped row.
    """
    keys = extract_entity_keys(transactions)

    valid = (
        keys["card1"].notna() & keys["addr1"].notna() & keys["origin_day"].notna()
    )

    def _int_str(col: pd.Series) -> pd.Series:
        # Round before the nullable-int cast to absorb any float32 rounding
        # noise introduced upstream by src.data's dtype downcasting; these
        # columns hold whole numbers (card/address codes, day counts).
        return col.round().astype("Int64").astype(str)

    uid = (
        _int_str(keys["card1"])
        + "_"
        + _int_str(keys["addr1"])
        + "_"
        + _int_str(keys["origin_day"])
    ).where(valid)
    uid.name = "uid"
    return uid

"""Resolve raw IEEE-CIS transactions into entities (candidate abuse rings).

Primary linkage must come from the transaction file: card1-card6, addr1/addr2,
P_emaildomain, and the D columns. DeviceInfo/DeviceType live in the identity
file and cover only ~24% of transactions, so device fields are a secondary
signal that strengthens a link when present but must never be required for
one -- a linkage rule that depends on device fields would silently drop the
~76% of transactions with no identity row.
"""

from __future__ import annotations

import pandas as pd


def extract_entity_keys(transactions: pd.DataFrame) -> pd.DataFrame:
    """Derive candidate identity keys for each transaction.

    Args:
        transactions: Raw transaction rows (as loaded from
            train_transaction.csv, optionally joined with train_identity.csv
            on TransactionID).

    Returns:
        A DataFrame indexed by TransactionID with one column per candidate
        linkage key (e.g. normalized card fingerprint, address fingerprint,
        email domain, device fingerprint where available).
    """
    raise NotImplementedError


def resolve_entities(transactions: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    """Group transactions sharing identifiers into entities.

    Args:
        transactions: Raw transaction rows.
        key_columns: Names of the candidate linkage key columns (as produced
            by extract_entity_keys) to union transactions on.

    Returns:
        A Series indexed by TransactionID mapping each transaction to its
        resolved entity_id.
    """
    raise NotImplementedError

"""Tests for src/entities.py — the card1_addr1_origin_day uid derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.entities import extract_entity_keys, resolve_entities


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1000, 1001, 1002, 1003, 1004, 1005],
            "TransactionDT": [
                10 * 86400,  # day 10
                12 * 86400,  # day 12
                20 * 86400,  # day 20, D1 null
                15 * 86400,  # day 15, addr1 null
                8 * 86400,  # day 8, both null
                30 * 86400,  # day 30
            ],
            "D1": [5, 7, np.nan, 3, np.nan, 3],
            "card1": [1000, 1000, 2000, 3000, 4000, 1000],
            "addr1": [200, 200, 300, np.nan, np.nan, 200],
        }
    )


def test_extract_entity_keys_components():
    keys = extract_entity_keys(_frame())

    assert list(keys.index) == [1000, 1001, 1002, 1003, 1004, 1005]
    assert keys.index.name == "TransactionID"
    assert keys.loc[1000, "day"] == 10
    assert keys.loc[1000, "origin_day"] == 5
    assert keys.loc[1001, "origin_day"] == 5  # same card lifetime as 1000
    assert pd.isna(keys.loc[1002, "origin_day"])  # D1 null -> origin_day null
    assert keys.loc[1003, "day"] == 15  # day is still computed when addr1 is null


def test_resolve_entities_known_uids():
    uid = resolve_entities(_frame())

    assert uid.loc[1000] == "1000_200_5"
    # Same card1/addr1/origin_day as 1000 despite a different transaction day:
    # this is the point of the derivation -- one persistent identity.
    assert uid.loc[1001] == "1000_200_5"
    # Same card1/addr1 as 1000 but a different origin_day -> a different uid.
    assert uid.loc[1005] == "1000_200_27"


def test_resolve_entities_nulls_produce_nan_not_fallback():
    uid = resolve_entities(_frame())

    assert pd.isna(uid.loc[1002])  # D1 null
    assert pd.isna(uid.loc[1003])  # addr1 null
    assert pd.isna(uid.loc[1004])  # both null

    # No fallback string (e.g. embedding "<NA>" or "nan") was invented for
    # the null cases -- they must be true NaN, not a stringified sentinel.
    non_null = uid.dropna()
    assert not non_null.str.contains("NA").any()
    assert not non_null.str.contains("nan").any()


def test_resolve_entities_no_rows_dropped():
    df = _frame()
    uid = resolve_entities(df)

    assert len(uid) == len(df)
    assert set(uid.index) == set(df["TransactionID"])


def test_resolve_entities_series_shape():
    uid = resolve_entities(_frame())

    assert uid.name == "uid"
    assert uid.index.name == "TransactionID"

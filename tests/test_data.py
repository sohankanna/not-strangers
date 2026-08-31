"""Tests for src/data.py.

Only the missing-file error path is covered here. The load/join/downcast
happy path needs the real IEEE-CIS CSVs, which are not available in every
environment this runs in (see scripts/download_data.sh).
"""

from __future__ import annotations

import pytest

from src.data import load_transactions


def test_raises_when_both_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        load_transactions(tmp_path)


def test_raises_when_identity_missing(tmp_path):
    (tmp_path / "train_transaction.csv").write_text("TransactionID\n1\n")
    with pytest.raises(FileNotFoundError, match="train_identity.csv"):
        load_transactions(tmp_path)


def test_raises_when_transaction_missing(tmp_path):
    (tmp_path / "train_identity.csv").write_text("TransactionID\n1\n")
    with pytest.raises(FileNotFoundError, match="train_transaction.csv"):
        load_transactions(tmp_path)

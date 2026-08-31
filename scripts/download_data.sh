#!/usr/bin/env bash
# Downloads the IEEE-CIS Fraud Detection dataset from Kaggle into data/.
#
# Prerequisites:
#   1. pip install kaggle
#   2. Create a Kaggle API token: https://www.kaggle.com/settings -> API -> Create New Token
#      Save the downloaded kaggle.json to ~/.kaggle/kaggle.json (chmod 600 ~/.kaggle/kaggle.json)
#   3. Accept the competition rules at:
#      https://www.kaggle.com/c/ieee-fraud-detection/rules
#
# The competition archive contains train_transaction.csv, train_identity.csv,
# test_transaction.csv and test_identity.csv, joined on TransactionID. Both
# halves of each pair are required -- transaction rows carry the core features
# and labels, identity rows only cover ~24% of transactions and add device
# signals on top. Neither file alone is sufficient.
#
# Usage:
#   bash scripts/download_data.sh

set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"

mkdir -p "$DATA_DIR"

kaggle competitions download -c ieee-fraud-detection -p "$DATA_DIR"

unzip -o "$DATA_DIR/ieee-fraud-detection.zip" -d "$DATA_DIR"

echo "Data downloaded and extracted to $DATA_DIR"

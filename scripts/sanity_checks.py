"""Task 3 (adversarial sanity checks): thin standalone wrapper.

The actual logic now lives in src/run_pipeline.py (write_sanity_checks and
its helpers) so `make results` can produce this section as part of one
consolidated pipeline run instead of every artifact re-loading and
re-training independently. This script is kept for standalone use --
`python scripts/sanity_checks.py` still works, appending the same "##
Sanity checks" section to results/ablation.md.

Usage:
    python scripts/sanity_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import run_pipeline


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)
    run_pipeline.write_sanity_checks(pipeline_data, trained)
    print("Appended 'Sanity checks' section to results/ablation.md")


if __name__ == "__main__":
    main()

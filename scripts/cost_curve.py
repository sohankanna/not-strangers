"""Task 4 (threshold sweep and cost curve): thin standalone wrapper.

The actual logic now lives in src/run_pipeline.py (write_cost_curve) so
`make results` can produce this artifact as part of one consolidated
pipeline run. This script is kept for standalone use.

Usage:
    python scripts/cost_curve.py
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
    run_pipeline.write_cost_curve(pipeline_data, trained)
    print("Wrote results/cost_curve.png and appended sweep section to ablation.md")


if __name__ == "__main__":
    main()

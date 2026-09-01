"""Task 2 (measure the investigator): thin standalone wrapper.

The actual logic now lives in src/run_pipeline.py (write_investigator_eval)
so `make results` can produce this artifact as part of one consolidated
pipeline run. This script is kept for standalone use.

Honest caveat, repeated here because it matters: if ANTHROPIC_API_KEY is
not set when this runs, every explanation takes investigator.py's fallback
path, and the groundedness rate in results/investigator_eval.md measures
that deterministic template, not the real LLM. See the written report's
own first paragraph for whether that was the case on the last run.

Usage:
    python scripts/eval_investigator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import run_pipeline


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()
    run_pipeline.write_investigator_eval(pipeline_data)
    print("Wrote results/investigator_eval.md")


if __name__ == "__main__":
    main()

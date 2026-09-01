"""One-off script to derive policy.py's step_up/review thresholds from real
cost-curve sweeps, so they're read off a sweep rather than hand-picked.

Both thresholds are the cost-minimizing point of a threshold sweep on the
CLUSTER model's test-set scores (evaluate.cost_per_10k), using the same
cost_fn=500 for both (a missed abuse case is equally costly regardless of
which action would have caught it) but different cost_fp:
  - step_up: cost_fp=5 -- this is exactly run_pipeline.COST_FP, i.e. the
    same sweep already reported in results/ablation.md's "Threshold sweep
    and cost curve" section. A step-up challenge is a light, cheap
    friction cost.
  - review: cost_fp=50 -- 10x costlier, representing a full manual review
    (analyst time, a held transaction, worse customer experience) rather
    than an automated challenge. 10x is an illustrative multiplier, not a
    Razorpay figure, same caveat as the rest of this project's costs.

Run once; the printed values are hard-coded as constants in policy.py with
this script named as their provenance.

Usage:
    python scripts/derive_policy_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src import evaluate, run_pipeline

THRESHOLDS = np.concatenate(
    [np.linspace(0.0, 0.05, 200), np.linspace(0.05, 1.0, 100)[1:]]
)


def _cost_minimizing_threshold(y_true, y_score, cost_fn: float, cost_fp: float) -> tuple[float, float]:
    costs = [
        evaluate.cost_per_10k(y_true, y_score, t, cost_fn, cost_fp) for t in THRESHOLDS
    ]
    best_idx = int(np.argmin(costs))
    return float(THRESHOLDS[best_idx]), float(costs[best_idx])


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    y_test = trained.y_test.to_numpy()
    y_score = trained.cluster_model.predict(trained.X_test_cluster)

    step_up_t, step_up_cost = _cost_minimizing_threshold(
        y_test, y_score, cost_fn=run_pipeline.COST_FN, cost_fp=5.0
    )
    review_t, review_cost = _cost_minimizing_threshold(
        y_test, y_score, cost_fn=run_pipeline.COST_FN, cost_fp=50.0
    )

    print(f"step_up threshold (cost_fn=500, cost_fp=5):  {step_up_t:.6f} (cost {step_up_cost:.2f})")
    print(f"review threshold  (cost_fn=500, cost_fp=50): {review_t:.6f} (cost {review_cost:.2f})")


if __name__ == "__main__":
    main()

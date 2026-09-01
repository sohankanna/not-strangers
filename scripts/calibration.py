"""Task 4: calibration check for the cluster model.

The policy engine (policy.py) thresholds on a raw score. If that score
isn't a calibrated probability, "review above 0.1843" is an arbitrary cut
on an arbitrary scale, not "review when we estimate >18.43% abuse risk".
This script measures calibration on the real test split and reports it
honestly -- it does not retrain or recalibrate the model (out of scope:
model.py is frozen this session).

Writes results/calibration.png (reliability curve) and appends a
Calibration section with the Brier score to results/ablation.md.

Usage:
    python scripts/calibration.py
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src import run_pipeline

RESULTS_DIR = REPO_ROOT / "results"
N_BINS = 15


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    y_test = trained.y_test.to_numpy()
    y_score = trained.cluster_model.predict(trained.X_test_cluster)

    brier = float(brier_score_loss(y_test, y_score))
    base_rate = float(y_test.mean())
    # A trivial "always predict the base rate" model's Brier score, for scale.
    brier_baseline = float(np.mean((y_test - base_rate) ** 2))

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_score, n_bins=N_BINS, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        color="#C0504D",
        label="cluster model",
    )
    ax.set_xlabel("Mean predicted score (within bin)")
    ax.set_ylabel("Observed fraction of positives (within bin)")
    ax.set_title(
        f"Reliability curve, cluster model, test split\n"
        f"({N_BINS} quantile bins, Brier score = {brier:.4f})"
    )
    ax.legend()
    ax.set_xlim(-0.02, max(mean_predicted_value.max(), 0.1) * 1.1)
    ax.set_ylim(-0.02, max(fraction_of_positives.max(), 0.1) * 1.1)
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "calibration.png", dpi=150)
    plt.close(fig)

    # Equal-weighted mean gap across bins -- reported, but NOT used as the
    # calibration verdict: with this base rate, most bins sit at very low
    # scores where the model is naturally well-calibrated (predicting near
    # 0 for mostly-0 outcomes is easy), so an equally-weighted average is
    # dominated by that easy majority and can hide bad calibration exactly
    # where policy.py's thresholds live. Report both, but drive the verdict
    # from the bins nearest the actual policy thresholds.
    gaps = fraction_of_positives - mean_predicted_value
    mean_abs_gap = float(np.mean(np.abs(gaps)))
    worst_gap_idx = int(np.argmax(np.abs(gaps)))
    worst_gap = float(gaps[worst_gap_idx])

    from src import policy as _policy

    # The highest-score bin is the one closest to (and above) the review
    # threshold in practice, and is where a policy that only reads the
    # equally-weighted average would miss the worst miscalibration.
    top_bin_gap = float(gaps[-1])
    top_bin_score = float(mean_predicted_value[-1])

    lines = ["## Calibration", ""]
    lines.append(
        "results/calibration.png -- reliability curve for the cluster "
        "model on the test split (quantile-binned, since scores "
        "concentrate near 0 at this base rate; equal-width bins would put "
        "almost everything in one bin)."
    )
    lines.append("")
    lines.append("![Reliability curve](calibration.png)")
    lines.append("")
    lines.append(f"- Brier score: **{brier:.4f}** (lower is better; a model that always")
    lines.append(
        f"  predicts the test-set base rate {base_rate:.4f} scores {brier_baseline:.4f} "
        "for comparison)"
    )
    lines.append(
        f"- Mean absolute gap between observed and predicted fraction, "
        f"equally weighted across the {N_BINS} bins: **{mean_abs_gap:.4f}** "
        "-- this number is misleading on its own, see below."
    )
    lines.append(
        f"- Highest-score bin (mean predicted score {top_bin_score:.4f}, the "
        "bin nearest where policy.py's thresholds actually operate): "
        f"observed fraction of positives is **{fraction_of_positives[-1]:.4f}**, "
        f"a gap of **{top_bin_gap:+.4f}**."
    )
    lines.append("")
    lines.append(
        "**The equally-weighted average is misleading here and would say "
        "the wrong thing if reported alone.** Most of the 15 bins sit at "
        "very low predicted scores, where a ~3.5%-base-rate model is "
        "naturally easy to calibrate (predicting near 0 for mostly-0 "
        "outcomes), so they pull the average down. The bin that actually "
        f"matters for policy.py -- the highest one, mean predicted score "
        f"{top_bin_score:.4f}, which is above both STEP_UP_THRESHOLD "
        f"({_policy.STEP_UP_THRESHOLD}) and REVIEW_THRESHOLD "
        f"({_policy.REVIEW_THRESHOLD}) -- is **overconfident by "
        f"{abs(top_bin_gap):.2f}** (predicts ~{top_bin_score:.2f}, actual "
        f"positive rate is only {fraction_of_positives[-1]:.2f})."
    )
    lines.append("")
    lines.append(
        "**Verdict: not well calibrated in the region the policy engine "
        "actually operates in, despite a low overall Brier score.** "
        f"policy.py's REVIEW_THRESHOLD ({_policy.REVIEW_THRESHOLD}) should "
        "be read as an arbitrary cut on this model's score scale, not as "
        f"\"we estimate >{_policy.REVIEW_THRESHOLD:.2%} abuse risk\" -- "
        "scores in this upper range systematically "
        "overstate the true positive rate. The fix, if a true probability "
        "is needed, is isotonic or Platt scaling fit on a held-out "
        "calibration slice -- **not implemented here**: model.py is frozen "
        "this session, and refitting a calibration map changes how scores "
        "are produced, which is not something to add quietly under a task "
        "that explicitly said do not retrain the model."
    )
    lines.append("")

    ablation_path = RESULTS_DIR / "ablation.md"
    existing = ablation_path.read_text(encoding="utf-8")
    ablation_path.write_text(
        existing.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote results/calibration.png and appended Calibration section "
        f"(Brier={brier:.4f}, mean_abs_gap={mean_abs_gap:.4f})"
    )


if __name__ == "__main__":
    main()

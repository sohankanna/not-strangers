"""Task 4: sweep thresholds and plot expected cost per 10k transactions for
both models, marking each model's own cost-minimizing operating point.

Writes results/cost_curve.png and appends a "## Threshold sweep and cost
curve" section to results/ablation.md with recall/FPR at each model's
chosen point. Costs are the same illustrative, parameterised assumptions
from run_pipeline.py (COST_FN, COST_FP) -- not Razorpay figures.

Usage:
    python scripts/cost_curve.py
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

from src import evaluate, run_pipeline

RESULTS_DIR = REPO_ROOT / "results"

THRESHOLDS = np.concatenate(
    [np.linspace(0.0, 0.05, 200), np.linspace(0.05, 1.0, 100)[1:]]
)


def _recall_and_fpr_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    predicted_positive = y_score >= threshold

    tp = np.sum((y_true == 1) & predicted_positive)
    fn = np.sum((y_true == 1) & ~predicted_positive)
    fp = np.sum((y_true == 0) & predicted_positive)
    tn = np.sum((y_true == 0) & ~predicted_positive)

    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return float(recall), float(fpr)


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)

    y_test = trained.y_test.to_numpy()
    scores = {
        "baseline": trained.baseline_model.predict(trained.X_test_baseline),
        "cluster": trained.cluster_model.predict(trained.X_test_cluster),
    }

    costs = {name: [] for name in scores}
    for name, y_score in scores.items():
        for t in THRESHOLDS:
            costs[name].append(
                evaluate.cost_per_10k(
                    y_test, y_score, t, run_pipeline.COST_FN, run_pipeline.COST_FP
                )
            )
        costs[name] = np.array(costs[name])

    chosen = {}
    for name, y_score in scores.items():
        best_idx = int(np.argmin(costs[name]))
        best_threshold = float(THRESHOLDS[best_idx])
        best_cost = float(costs[name][best_idx])
        recall, fpr = _recall_and_fpr_at_threshold(y_test, y_score, best_threshold)
        chosen[name] = {
            "threshold": best_threshold,
            "cost_per_10k": best_cost,
            "recall": recall,
            "fpr": fpr,
        }

    colors = {"baseline": "#4472C4", "cluster": "#C0504D"}
    zoom_xmax = 0.05

    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5.5))
    for name in scores:
        ax.plot(THRESHOLDS, costs[name], label=name, color=colors[name])
        ax.scatter(
            [chosen[name]["threshold"]],
            [chosen[name]["cost_per_10k"]],
            color=colors[name],
            zorder=5,
            s=50,
            edgecolor="black",
        )
    ax.axvspan(0, zoom_xmax, color="grey", alpha=0.12)
    ax.set_xlabel("Score threshold (flag if score >= threshold)")
    ax.set_ylabel(
        f"Expected cost per 10k txns (cost_fn={run_pipeline.COST_FN:g}, "
        f"cost_fp={run_pipeline.COST_FP:g}, illustrative)"
    )
    ax.set_title("Full threshold range")
    ax.legend()

    zoom_mask = THRESHOLDS <= zoom_xmax
    offsets = {"baseline": (12, 15), "cluster": (12, -22)}
    for name in scores:
        ax_zoom.plot(
            THRESHOLDS[zoom_mask], costs[name][zoom_mask], label=name, color=colors[name]
        )
        ax_zoom.scatter(
            [chosen[name]["threshold"]],
            [chosen[name]["cost_per_10k"]],
            color=colors[name],
            zorder=5,
            s=70,
            edgecolor="black",
        )
        ax_zoom.annotate(
            f"{name} chosen: t={chosen[name]['threshold']:.4f}\n"
            f"cost={chosen[name]['cost_per_10k']:,.0f}",
            (chosen[name]["threshold"], chosen[name]["cost_per_10k"]),
            textcoords="offset points",
            xytext=offsets[name],
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": colors[name], "lw": 0.8},
        )
    ax_zoom.set_xlabel("Score threshold (zoomed: 0 to 0.05)")
    ax_zoom.set_ylabel("Expected cost per 10k txns")
    ax_zoom.set_title("Zoomed: where the minima actually are")
    ax_zoom.legend()

    fig.suptitle("Cost per 10k transactions vs. threshold")
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "cost_curve.png", dpi=150)
    plt.close(fig)

    lines = ["## Threshold sweep and cost curve", ""]
    lines.append(
        "results/cost_curve.png sweeps score thresholds directly (not "
        "relying on the calibration assumption behind ablation.md's headline "
        "threshold) and marks each model's own cost-minimizing point."
    )
    lines.append("")
    lines.append("![Cost per 10k vs threshold](cost_curve.png)")
    lines.append("")
    lines.append("| model | chosen threshold | cost per 10k at chosen point | recall | FPR |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ("baseline", "cluster"):
        c = chosen[name]
        lines.append(
            f"| {name} | {c['threshold']:.4f} | {c['cost_per_10k']:.2f} | "
            f"{c['recall']:.4f} | {c['fpr']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"Worth being explicit about: the cost-minimizing FPR here is "
        f"{chosen['cluster']['fpr']:.0%}-{chosen['baseline']['fpr']:.0%} -- "
        "a direct, correct mathematical consequence of the assumed 100:1 "
        "cost_fn:cost_fp ratio (missing fraud is assumed to be that much "
        "worse than a false alarm, so the optimum flags aggressively), not "
        "a bug. In practice this means stepping up upwards of a third of "
        "all legitimate transactions at the \"optimal\" point -- whether "
        "that's acceptable is a business call the assumed cost ratio drives "
        "entirely; a less aggressive cost ratio (or a friction budget "
        "constraint) would move the chosen threshold and the resulting FPR "
        "substantially."
    )
    lines.append("")

    ablation_path = RESULTS_DIR / "ablation.md"
    existing = ablation_path.read_text(encoding="utf-8")
    ablation_path.write_text(
        existing.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Wrote results/cost_curve.png and appended sweep section to ablation.md")


if __name__ == "__main__":
    main()

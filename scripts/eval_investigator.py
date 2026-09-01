"""Task 2: measure investigator.py rather than just shipping it.

Generates explanations for 30 clusters spanning the risk range (by
cluster_prior_fraud_share, the strongest single risk signal per
results/ablation.md), then checks groundedness programmatically: every
numeric literal found in each narrative must match some evidence value,
allowing reasonable rounding (to 0-4 decimal places) and percentage-form
(a value v may be cited as v*100). Writes results/investigator_eval.md.

Honest caveat, repeated here because it matters: ANTHROPIC_API_KEY was not
set when this last ran (see DEVLOG.md's Task 1 entry), so every explanation
below took investigator.py's fallback path. This measures the fallback
template's groundedness (which is grounded by construction), not the real
LLM's. Re-run this script after setting a real key for an actual measurement
of the LLM path.

Usage:
    python scripts/eval_investigator.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src import investigator, run_pipeline
from src.graph import get_connected_components

RESULTS_DIR = REPO_ROOT / "results"
N_CLUSTERS = 30
N_EXAMPLES = 3

_NUMBER_PATTERN = re.compile(r"-?\d+\.\d+|-?\d+")


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_PATTERN.findall(text)]


def _value_matches(claimed: float, evidence_value: float) -> bool:
    candidates = {claimed, claimed / 100}
    for candidate in candidates:
        for decimals in range(0, 5):
            if abs(candidate - round(float(evidence_value), decimals)) < 1e-9:
                return True
    return False


def _ungrounded_claims(narrative: str, evidence: dict) -> list[float]:
    evidence_values = list(evidence.values())
    ungrounded = []
    for claim in _extract_numbers(narrative):
        if not any(_value_matches(claim, v) for v in evidence_values):
            ungrounded.append(claim)
    return ungrounded


def _cluster_risk_key(component: set, cluster_features: pd.DataFrame) -> float:
    rep = next(iter(component))
    if rep not in cluster_features.index:
        return float("-inf")
    value = cluster_features.loc[rep, "cluster_prior_fraud_share"]
    return float(value) if pd.notna(value) else 0.0


def _select_clusters(pipeline_data: run_pipeline.PipelineData, n: int) -> list[set]:
    components = get_connected_components(pipeline_data.entity_graph.graph)
    multi = [c for c in components if len(c) >= 2]
    multi_sorted = sorted(
        multi, key=lambda c: _cluster_risk_key(c, pipeline_data.cluster_features)
    )
    if len(multi_sorted) <= n:
        return multi_sorted
    indices = sorted(set(np.linspace(0, len(multi_sorted) - 1, n).round().astype(int)))
    return [multi_sorted[i] for i in indices]


def main() -> None:
    pipeline_data = run_pipeline.load_and_prepare()

    full = pipeline_data.df.set_index("TransactionID")
    full = pd.concat([full, pipeline_data.entity_ids.rename("uid")], axis=1)

    selected = _select_clusters(pipeline_data, N_CLUSTERS)

    explanations = []
    for i, members in enumerate(selected):
        members = sorted(members)
        cluster_features_sub = pipeline_data.cluster_features.loc[
            [m for m in members if m in pipeline_data.cluster_features.index]
        ]
        transactions_sub = full[full["uid"].isin(members)]
        explanation = investigator.explain_cluster(
            cluster_id=f"cluster-{i}", cluster_features=cluster_features_sub, transactions=transactions_sub
        )
        explanations.append(explanation)

    ranked = investigator.prioritize_clusters(explanations)

    total_claims = 0
    total_ungrounded = 0
    ungrounded_records = []
    sources = {}
    for explanation in explanations:
        claims = _extract_numbers(explanation.narrative)
        ungrounded = _ungrounded_claims(explanation.narrative, explanation.evidence)
        total_claims += len(claims)
        total_ungrounded += len(ungrounded)
        sources[explanation.source] = sources.get(explanation.source, 0) + 1
        if ungrounded:
            ungrounded_records.append((explanation.cluster_id, ungrounded, explanation.narrative))

    groundedness_rate = (
        1.0 - (total_ungrounded / total_claims) if total_claims else float("nan")
    )

    # 3 examples spanning the risk range: lowest, middle, highest priority.
    ranked_asc = list(reversed(ranked))
    example_indices = sorted(
        {0, len(ranked_asc) // 2, len(ranked_asc) - 1}
    ) if len(ranked_asc) >= N_EXAMPLES else list(range(len(ranked_asc)))
    examples = [ranked_asc[i] for i in example_indices]

    lines: list[str] = []
    lines.append("# Investigator evaluation")
    lines.append("")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    lines.append(
        f"ANTHROPIC_API_KEY was {'set' if has_key else '**NOT set**'} when this ran. "
        f"Explanation sources: {sources}. "
        + (
            "**Every explanation below took the deterministic fallback path "
            "(source=ungrounded-fallback), not the real LLM.** The fallback "
            "narrative is built by directly formatting the evidence dict's "
            "own values, so it is grounded by construction -- the "
            "groundedness rate below measures that template, not "
            "claude-sonnet-4-6's actual behavior under the prompt's hard "
            "rule. Re-run this script with a real key for an honest "
            "measurement of the LLM path."
            if not has_key
            else "At least one explanation used the real LLM path -- see "
            "the `source` column below for which."
        )
    )
    lines.append("")
    lines.append(
        f"Evaluated {len(explanations)} clusters, selected to span the risk "
        "range: sorted all multi-uid clusters (2+ members) by "
        "cluster_prior_fraud_share, then took 30 evenly-spaced percentile "
        "points across that sorted list (not just the top 30 riskiest)."
    )
    lines.append("")

    lines.append("## Groundedness")
    lines.append("")
    lines.append(
        f"- Total numeric claims extracted across all narratives: **{total_claims}**"
    )
    lines.append(f"- Ungrounded claims: **{total_ungrounded}**")
    lines.append(
        f"- Groundedness rate: **{groundedness_rate:.2%}** "
        "(a claim counts as grounded if it matches some evidence value "
        "exactly, at any rounding from 0-4 decimal places, or as that "
        "value expressed as a percentage)"
    )
    lines.append("")
    if ungrounded_records:
        lines.append("### Every ungrounded claim found")
        lines.append("")
        for cluster_id, ungrounded, narrative in ungrounded_records:
            lines.append(f"- **{cluster_id}**: claimed {ungrounded}")
            lines.append(f"  > {narrative}")
        lines.append("")
    else:
        lines.append("No ungrounded claims found in this run.")
        lines.append("")

    lines.append("## 3 example explanations (lowest, median, highest priority)")
    lines.append("")
    for explanation in examples:
        lines.append(f"### {explanation.cluster_id} (source={explanation.source})")
        lines.append("")
        lines.append(f"- Priority score: {explanation.priority_score:.4f}")
        lines.append(f"- Member uids: {explanation.entity_ids}")
        lines.append(f"- Evidence: `{explanation.evidence}`")
        lines.append("")
        lines.append(f"> {explanation.narrative}")
        lines.append("")

    (RESULTS_DIR / "investigator_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote results/investigator_eval.md (groundedness={groundedness_rate:.2%})")


if __name__ == "__main__":
    main()

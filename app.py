"""abuse-ring-sentinel -- analyst review console.

An analyst-facing review queue, not a metrics showcase. Three tabs:
  - Review queue: clusters ranked by priority score, with a detail panel
    for the selected cluster (LLM narrative, evidence, member uids, a
    transaction timeline, and the policy decision).
  - Model performance: the ablation table, cost curve and calibration
    plot, parsed from results/ at runtime.
  - Audit trail: results/audit_sample.jsonl, filterable by action.

Attribution is load-bearing here, not decorative: CLAUDE.md's rule is
"the LLM layer explains and prioritizes, policy.py decides, never merge
them," and this UI is where that separation has to be legible to a human,
not just true in the architecture. Every narrative shown is labeled with
its actual source (the real Anthropic call, or the deterministic fallback
-- see investigator.py) and every action shown is labeled as a policy.py
decision with the exact threshold that produced it. Nothing here is
computed by "the dashboard" -- it's read from src/ and results/ at
runtime, per this session's own rule against inventing numbers.

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src import investigator, policy, run_pipeline  # noqa: E402
from src.graph import get_connected_components  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
MAKE_RESULTS_CMD = "python -m src.run_pipeline"

# Palette -- mirrors .streamlit/config.toml; duplicated here because the
# canvas-rendered matplotlib timeline and the hand-rolled HTML tables can't
# read the Streamlit theme config directly.
BG = "#0D0D0F"
SURFACE = "#18181C"
BORDER = "#2A2A2E"
TEXT = "#E4E4E7"
TEXT_MUTED = "#9A9AA5"
ACCENT = "#5B7FBF"
RISK_REVIEW = "#C4645C"
RISK_STEPUP = "#B08D3E"
RISK_ALLOW = "#6B7280"

st.set_page_config(
    page_title="abuse-ring-sentinel",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Styling: theme config (.streamlit/config.toml) covers Streamlit's own
# widgets; this covers everything it can't reach (fonts, tables, badges).
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}
        .stApp {{
            background-color: {BG};
        }}
        h1, h2, h3, h4, h5 {{
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        code, .mono, .stCode, .stCaption {{
            font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        }}
        /* ---- badges ---- */
        .badge {{
            display: inline-block;
            padding: 2px 9px;
            border-radius: 3px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        .badge-review {{ background: rgba(196,100,92,0.16); color: {RISK_REVIEW}; }}
        .badge-step_up {{ background: rgba(176,141,62,0.16); color: {RISK_STEPUP}; }}
        .badge-allow {{ background: rgba(107,114,128,0.20); color: {TEXT_MUTED}; }}
        .badge-llm {{ background: rgba(91,127,191,0.16); color: {ACCENT}; }}
        .badge-fallback {{ background: rgba(154,154,165,0.14); color: {TEXT_MUTED}; }}
        /* ---- section framing ---- */
        .panel {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 18px 20px;
            margin-bottom: 18px;
        }}
        .panel-label {{
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {TEXT_MUTED};
            margin-bottom: 8px;
        }}
        .narrative-text {{
            color: {TEXT};
            line-height: 1.55;
            font-size: 0.94rem;
        }}
        /* ---- tables ---- */
        .table-wrap {{
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid {BORDER};
            border-radius: 6px;
        }}
        table.sentinel-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
        }}
        table.sentinel-table thead th {{
            position: sticky;
            top: 0;
            background: {SURFACE};
            color: {TEXT_MUTED};
            text-align: left;
            font-weight: 600;
            font-size: 0.72rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 8px 12px;
            border-bottom: 1px solid {BORDER};
            white-space: nowrap;
        }}
        table.sentinel-table td {{
            padding: 6px 12px;
            border-bottom: 1px solid {BORDER};
            color: {TEXT};
            white-space: nowrap;
        }}
        table.sentinel-table td.num {{
            font-family: 'JetBrains Mono', monospace;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        table.sentinel-table tbody tr:hover {{
            background: rgba(255,255,255,0.02);
        }}
        .uid-chip {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid {BORDER};
            border-radius: 3px;
            padding: 1px 7px;
            margin: 2px 3px 2px 0;
            color: {TEXT_MUTED};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def _fmt_cell(v) -> tuple[str, bool]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--", True
    if isinstance(v, bool):
        return str(v), False
    if isinstance(v, float):
        return f"{v:,.4f}", True
    if isinstance(v, int):
        return f"{v:,}", True
    return str(v), False


def render_html_table(df: pd.DataFrame, max_height: str = "420px") -> None:
    """A hand-rolled HTML table, used wherever precise typography (monospace,
    right-aligned, tabular numerals) matters. st.dataframe's grid is
    canvas-rendered (glide-data-grid) and cannot be reached by CSS at the
    per-cell level -- this is the deliberate workaround, not an oversight.
    """
    if df.empty:
        st.caption("(no rows)")
        return
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            text, is_num = _fmt_cell(v)
            cls = "num" if is_num else ""
            cells.append(f'<td class="{cls}">{text}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    html = (
        f'<div class="table-wrap" style="max-height:{max_height}">'
        f'<table class="sentinel-table"><thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Results-file access: never a raw traceback. Every number on screen is
# read from these files (or computed from src/ at runtime) -- nothing is
# hardcoded into the app.
# ---------------------------------------------------------------------------

def read_results_text(filename: str) -> str | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        st.error(
            f"Missing `results/{filename}`. Generate it with:\n\n"
            f"```\n{MAKE_RESULTS_CMD}\n```"
        )
        return None
    return path.read_text(encoding="utf-8")


def results_image(filename: str) -> Path | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        st.error(
            f"Missing `results/{filename}`. Generate it with:\n\n"
            f"```\n{MAKE_RESULTS_CMD}\n```"
        )
        return None
    return path


_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown_tables(text: str) -> list[tuple[str, pd.DataFrame]]:
    """Every pipe-table in a markdown file, paired with its nearest
    preceding heading. Parsed fresh from the actual file every time --
    never hand-transcribed into this app.
    """
    lines = text.split("\n")
    out: list[tuple[str, pd.DataFrame]] = []
    heading = ""
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            i += 1
            continue
        if (
            line.strip().startswith("|")
            and i + 1 < n
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
        ):
            header = _split_row(line)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            if rows:
                out.append((heading, pd.DataFrame(rows, columns=header)))
            continue
        i += 1
    return out


def get_table(tables: list[tuple[str, pd.DataFrame]], heading_contains: str, index: int = 0) -> pd.DataFrame | None:
    matches = [df for h, df in tables if heading_contains.lower() in h.lower()]
    return matches[index] if len(matches) > index else None


def extract_section(text: str, heading_contains: str) -> str:
    """Raw text of one markdown section (heading through the next heading
    at the same or shallower level, or EOF), with image reference lines
    stripped (the real image is rendered separately via st.image).
    """
    lines = text.split("\n")
    start, level = None, None
    for i, line in enumerate(lines):
        if line.startswith("#") and heading_contains.lower() in line.lower():
            start = i
            level = len(line) - len(line.lstrip("#"))
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("#"):
            this_level = len(lines[j]) - len(lines[j].lstrip("#"))
            if this_level <= level:
                end = j
                break
    section = [l for l in lines[start + 1 : end] if not l.strip().startswith("![")]
    return "\n".join(section).strip()


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.rstrip("*").str.strip(), errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Pipeline access. get_pipeline is a cache_resource (the objects it returns
# -- LightGBM boosters, a networkx graph, multi-GB DataFrames -- aren't the
# kind of thing st.cache_data is meant to hash/serialize). Everything
# derived from it that IS plain, serializable data goes through
# st.cache_data, per the task's own instruction, so re-rendering the UI
# after the one-time ~90s load/train is fast.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading transactions, building the graph, and training both models (one-time, ~90s)...")
def get_pipeline():
    pipeline_data = run_pipeline.load_and_prepare()
    trained = run_pipeline.train_both_models(pipeline_data)
    return pipeline_data, trained


def load_pipeline_or_stop():
    try:
        return get_pipeline()
    except FileNotFoundError as exc:
        st.error(
            "Cannot load the pipeline -- the raw data isn't present.\n\n"
            f"{exc}\n\n"
            "Run `bash scripts/download_data.sh` (needs Kaggle credentials; "
            "see README.md) and reload this page."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def _cluster_membership(_pipeline_data) -> dict:
    """uid -> cluster index, for multi-member (2+) clusters only."""
    components = get_connected_components(_pipeline_data.entity_graph.graph)
    cluster_of: dict = {}
    for i, comp in enumerate(components):
        if len(comp) >= 2:
            for u in comp:
                cluster_of[u] = i
    return cluster_of


@st.cache_data(show_spinner="Preparing transaction data...")
def annotated_transactions(_pipeline_data) -> pd.DataFrame:
    """All transactions (train + test) with uid and cluster_id columns,
    restricted to rows whose uid belongs to a real multi-member cluster.
    """
    cluster_of = _cluster_membership(_pipeline_data)
    full = _pipeline_data.df.set_index("TransactionID")
    full = pd.concat([full, _pipeline_data.entity_ids.rename("uid")], axis=1)
    full["cluster_id"] = full["uid"].map(cluster_of)
    full = full.loc[full["cluster_id"].notna()].copy()
    full["cluster_id"] = full["cluster_id"].astype(int)
    return full


@st.cache_data(show_spinner="Scoring clusters and building the review queue...")
def build_cluster_queue(_pipeline_data, _trained, max_rows: int = 300, candidate_pool: int = 400) -> pd.DataFrame:
    """Two passes, deliberately: a fully vectorized pass over every
    qualifying cluster to find a bounded candidate pool cheaply, then the
    real (non-vectorizable, but exact) investigator.build_evidence /
    _priority_score / policy.decide only on that bounded pool. Calling
    those per-cluster over all ~1,500+ multi-uid clusters in a Python loop
    (the first version of this function) took minutes; this takes seconds,
    without changing what any individual cluster's priority score means.
    """
    full = annotated_transactions(_pipeline_data).copy()
    cf = _pipeline_data.cluster_features

    test_scores = pd.Series(
        _trained.cluster_model.predict(_trained.X_test_cluster),
        index=_trained.X_test_cluster.index,
    )
    full["test_score"] = full.index.to_series().map(test_scores)

    # Vectorized: which clusters have at least one test-period transaction
    # (a real score to decide on) at all.
    test_count = full.groupby("cluster_id")["test_score"].count()
    qualifying_ids = test_count[test_count > 0].index
    if len(qualifying_ids) == 0:
        return pd.DataFrame()

    # Vectorized: a cheap proxy for priority -- cluster_prior_fraud_share is
    # the single dominant term (100x) in investigator._priority_score's
    # real formula and is constant across a cluster's members, so ranking
    # by it first is a close approximation used only to decide which
    # clusters are worth the exact (non-vectorized) computation below, not
    # a different number shown anywhere.
    qualifying = full[full["cluster_id"].isin(qualifying_ids)]
    first_uid = qualifying.groupby("cluster_id")["uid"].first()
    proxy = first_uid.map(cf["cluster_prior_fraud_share"]).fillna(0.0)
    candidate_ids = proxy.sort_values(ascending=False).head(candidate_pool).index

    candidate_full = qualifying[qualifying["cluster_id"].isin(candidate_ids)]

    rows = []
    for cluster_id, grp in candidate_full.groupby("cluster_id"):
        test_grp = grp.dropna(subset=["test_score"])
        if test_grp.empty:
            continue
        members = sorted(grp["uid"].unique())
        cf_sub = cf.loc[[u for u in members if u in cf.index]]
        if cf_sub.empty:
            continue
        evidence = investigator.build_evidence(cf_sub, grp)
        priority = investigator._priority_score(evidence)
        cluster_score = float(test_grp["test_score"].max())
        decision = policy.decide(str(cluster_id), cluster_score, cf_sub)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "uid_count": len(members),
                "txn_count": int(len(grp)),
                "priority_score": round(priority, 4),
                "cluster_score": round(cluster_score, 4),
                "action": decision.action,
                "threshold_applied": decision.threshold_applied,
            }
        )

    queue = pd.DataFrame(rows)
    if queue.empty:
        return queue
    return (
        queue.sort_values("priority_score", ascending=False)
        .head(max_rows)
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner="Generating explanation for this cluster...")
def get_cluster_detail(_pipeline_data, cluster_id: int) -> dict:
    full = annotated_transactions(_pipeline_data)
    transactions_sub = full[full["cluster_id"] == cluster_id].copy()
    members = sorted(transactions_sub["uid"].unique())
    cf_sub = _pipeline_data.cluster_features.loc[
        [u for u in members if u in _pipeline_data.cluster_features.index]
    ]
    explanation = investigator.explain_cluster(
        cluster_id=f"cluster-{cluster_id}",
        cluster_features=cf_sub,
        transactions=transactions_sub,
    )
    return {
        "members": members,
        "transactions": transactions_sub,
        "explanation": explanation,
    }


def render_timeline(transactions_sub: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 2.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    day = transactions_sub["TransactionDT"] / 86400.0
    colors = [RISK_REVIEW if f == 1 else RISK_ALLOW for f in transactions_sub["isFraud"]]
    ax.scatter(day, transactions_sub["TransactionAmt"], c=colors, s=26, edgecolor="none", alpha=0.9)
    ax.set_xlabel("day (TransactionDT / 86400)", color=TEXT_MUTED, fontsize=8)
    ax.set_ylabel("amount", color=TEXT_MUTED, fontsize=8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.grid(color=BORDER, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Review queue tab
# ---------------------------------------------------------------------------

def render_cluster_detail(pipeline_data, cluster_id: int, queue_row: pd.Series) -> None:
    detail = get_cluster_detail(pipeline_data, cluster_id)
    explanation = detail["explanation"]
    members = detail["members"]
    transactions_sub = detail["transactions"]

    st.markdown(f"### Cluster {cluster_id}")

    action = queue_row["action"]
    st.markdown(
        f'{_badge(action.upper(), action)} &nbsp;'
        f'<span class="mono" style="color:{TEXT_MUTED};font-size:0.85rem;">'
        f'score {queue_row["cluster_score"]:.4f} against threshold '
        f'{queue_row["threshold_applied"]:.4f}</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Decided by **policy.py** (deterministic, score-vs-threshold) -- "
        f"see src/policy.py. Not the language model."
    )

    source_kind = "llm" if explanation.source == "llm" else "fallback"
    source_label = "claude-sonnet-4-6" if explanation.source == "llm" else "template fallback (no LLM)"
    error_html = (
        f'<div style="margin-top:8px;font-size:0.78rem;color:{TEXT_MUTED};">'
        f'Fallback reason: <span class="mono">{explanation.error}</span></div>'
        if explanation.error
        else ""
    )
    st.markdown(
        f'<div class="panel">'
        f'<div class="panel-label">LLM explanation &nbsp;{_badge(source_label, source_kind)}</div>'
        f'<div class="narrative-text">{explanation.narrative}</div>'
        f'{error_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown('<div class="panel-label">Evidence (drove the score)</div>', unsafe_allow_html=True)
        evidence_df = pd.DataFrame(
            [{"feature": k, "value": v} for k, v in explanation.evidence.items()]
        )
        render_html_table(evidence_df, max_height="280px")
    with col_b:
        st.markdown(f'<div class="panel-label">Member uids ({len(members)})</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="uid-chip">{u}</span>' for u in members)
        st.markdown(f'<div style="max-height:280px;overflow-y:auto;">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-label">Transaction timeline</div>', unsafe_allow_html=True)
    st.caption(f"{len(transactions_sub):,} transactions -- red = isFraud=1, grey = isFraud=0")
    st.pyplot(render_timeline(transactions_sub), use_container_width=True)


def render_queue_tab(pipeline_data, trained) -> None:
    queue = build_cluster_queue(pipeline_data, trained)
    if queue.empty:
        st.info("No clusters with test-period activity were found.")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown('<div class="panel-label">Cluster queue, ranked by priority</div>', unsafe_allow_html=True)
        st.caption(
            f"Top {len(queue)} of the multi-uid clusters with test-period "
            "activity, by investigator.py's priority_score."
        )
        display_cols = ["cluster_id", "uid_count", "txn_count", "action", "priority_score"]
        risk_colors = {"review": RISK_REVIEW, "step_up": RISK_STEPUP, "allow": RISK_ALLOW}

        def _color_action(val: str) -> str:
            return f"color: {risk_colors.get(val, TEXT)}; font-weight: 600;"

        styled_queue = queue[display_cols].style.map(_color_action, subset=["action"])
        event = st.dataframe(
            styled_queue,
            hide_index=True,
            use_container_width=True,
            height=560,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "cluster_id": st.column_config.NumberColumn("cluster", format="%d"),
                "uid_count": st.column_config.NumberColumn("uids", format="%d"),
                "txn_count": st.column_config.NumberColumn("txns", format="%d"),
                "action": st.column_config.TextColumn("action"),
                "priority_score": st.column_config.NumberColumn("priority", format="%.3f"),
            },
        )
        selected = event.selection.rows if event and event.selection else []

    with right:
        if selected:
            row = queue.iloc[selected[0]]
            render_cluster_detail(pipeline_data, int(row["cluster_id"]), row)
        else:
            st.info("Select a cluster from the queue on the left to see its detail.")


# ---------------------------------------------------------------------------
# Model performance tab
# ---------------------------------------------------------------------------

def render_performance_tab() -> None:
    ablation_text = read_results_text("ablation.md")
    if ablation_text is None:
        return

    tables = parse_markdown_tables(ablation_text)

    st.markdown('<div class="panel-label">Ablation: transaction-only vs. cluster-augmented</div>', unsafe_allow_html=True)
    results_table = get_table(tables, "Results")
    reablation_table = get_table(tables, "Ablation re-run")
    if results_table is not None:
        combined = results_table
        if reablation_table is not None:
            combined = pd.concat([results_table, reablation_table], ignore_index=True)
        combined = to_numeric(combined, [c for c in combined.columns if c != "model"])
        render_html_table(combined, max_height="220px")

        # Computed from the parsed table at runtime, not hardcoded -- if
        # results/ablation.md's numbers ever changed, this caption would
        # move with them instead of quietly going stale.
        pr_auc_col = next((c for c in combined.columns if "pr-auc" in c.lower()), None)
        by_model = combined.set_index("model")[pr_auc_col] if pr_auc_col else None
        if by_model is not None and {"baseline", "cluster", "cluster (no cluster_prior_fraud_share)"} <= set(by_model.index):
            full_lift = by_model["cluster"] - by_model["baseline"]
            trimmed_lift = by_model["cluster (no cluster_prior_fraud_share)"] - by_model["baseline"]
            share_pct = (1 - trimmed_lift / full_lift) * 100 if full_lift else float("nan")
            st.caption(
                f"Third row: cluster_prior_fraud_share alone accounts for "
                f"roughly {share_pct:.0f}% of the headline PR-AUC lift "
                f"({full_lift:+.4f} full vs. {trimmed_lift:+.4f} without it) "
                "-- see results/ablation.md's Sanity checks section for the "
                "leak trace behind that feature."
            )

    importances = get_table(tables, "feature importances")
    if importances is not None:
        with st.expander("Cluster model feature importances (top 20 by gain)"):
            render_html_table(to_numeric(importances, ["gain"]), max_height="360px")

    st.markdown("---")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('<div class="panel-label">Cost curve</div>', unsafe_allow_html=True)
        img = results_image("cost_curve.png")
        if img is not None:
            st.image(str(img), use_container_width=True)
        sweep_table = get_table(tables, "Threshold sweep")
        if sweep_table is not None:
            render_html_table(to_numeric(sweep_table, ["chosen threshold", "cost per 10k at chosen point", "recall", "FPR"]), max_height="160px")
        sweep_prose = extract_section(ablation_text, "Threshold sweep and cost curve")
        sweep_prose = "\n".join(l for l in sweep_prose.split("\n") if not l.strip().startswith("|") and l.strip())
        if sweep_prose:
            st.caption(sweep_prose.split("\n")[-1])

    with cc2:
        st.markdown('<div class="panel-label">Calibration</div>', unsafe_allow_html=True)
        img2 = results_image("calibration.png")
        if img2 is not None:
            st.image(str(img2), use_container_width=True)
        calib_prose = extract_section(ablation_text, "## Calibration")
        verdict_lines = [l for l in calib_prose.split("\n") if l.strip().startswith("**Verdict")]
        if verdict_lines:
            st.warning(verdict_lines[0].strip("*"))


# ---------------------------------------------------------------------------
# Audit trail tab
# ---------------------------------------------------------------------------

def render_audit_tab() -> None:
    audit_path = RESULTS_DIR / "audit_sample.jsonl"
    if not audit_path.exists():
        st.error(
            f"Missing `results/audit_sample.jsonl`. Generate it with:\n\n"
            f"```\n{MAKE_RESULTS_CMD}\n```"
        )
        return

    lines = [l for l in audit_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    audit_df = pd.DataFrame(records)
    audit_df["feature_values"] = audit_df["feature_values"].apply(json.dumps)

    actions = sorted(audit_df["action"].unique())
    selected_actions = st.multiselect("Filter by action", actions, default=actions)
    filtered = audit_df[audit_df["action"].isin(selected_actions)]
    st.caption(f"Showing {len(filtered):,} of {len(audit_df):,} audit records from results/audit_sample.jsonl.")

    display_cols = ["transaction_id", "uid", "score", "threshold_applied", "action", "reason", "model_version", "timestamp"]
    render_html_table(filtered[display_cols], max_height="620px")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _inject_css()
    st.title("abuse-ring-sentinel")
    st.caption("Coordinated payment abuse review console -- ML scores, an LLM explains, policy.py decides.")

    pipeline_data, trained = load_pipeline_or_stop()

    tab_queue, tab_perf, tab_audit = st.tabs(["Review queue", "Model performance", "Audit trail"])
    with tab_queue:
        render_queue_tab(pipeline_data, trained)
    with tab_perf:
        render_performance_tab()
    with tab_audit:
        render_audit_tab()


if __name__ == "__main__":
    main()

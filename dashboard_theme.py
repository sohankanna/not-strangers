"""Shared color palette for app.py and its helper modules
(dashboard_graph.py, dashboard_attribution.py).

Single source of truth so the network graph (plotly) and the SHAP/
threshold charts (matplotlib) can't drift from app.py's own HTML/CSS
palette -- all three render outside Streamlit's own theme system
(.streamlit/config.toml), which is why this exists as a plain constants
module instead of being read from Streamlit's theme at runtime.
"""

from __future__ import annotations

BG = "#0D0D0F"
SURFACE = "#18181C"
BORDER = "#2A2A2E"
TEXT = "#E4E4E7"
TEXT_MUTED = "#9A9AA5"
ACCENT = "#5B7FBF"
RISK_REVIEW = "#C4645C"
RISK_STEPUP = "#B08D3E"
RISK_ALLOW = "#6B7280"

# New for this session: a fourth attribution color, distinct from ACCENT
# (already used for "llm"-sourced narrative) -- tags anything that comes
# from the trained model itself (SHAP, threshold position), so a reader can
# tell MODEL / POLICY / LLM apart at a glance, matching CLAUDE.md's
# three-way ML/LLM/policy separation instead of only labeling two of them.
MODEL_COLOR = "#4A9B8A"

# Linkage-rule edge colors for the cluster network graph (Task 1). Kept
# here rather than in dashboard_graph.py so a reader auditing the palette
# has one file to check, not two.
RULE_COLORS = {
    "device_info": ACCENT,
    "addr1_email": "#7FA37F",  # muted green
    "card_bank_addr": RISK_STEPUP,
}
MULTI_RULE_COLOR = "#9A6FA0"  # muted purple -- edge created by more than one rule
DEFAULT_EDGE_COLOR = "#4A4A52"

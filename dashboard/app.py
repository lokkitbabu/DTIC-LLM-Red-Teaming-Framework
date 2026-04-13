"""
Analytics Dashboard — entry point and routing.

Launch with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from dashboard.data_loader import DataLoader, FilterState, apply_filters, _merge_filter_state
from dashboard.views.summary import render_summary_view
from dashboard.views.detail import render_run_detail
from dashboard.views.charts import render_aggregate_charts
from dashboard.views.comparison import render_comparison_view
from dashboard.views.run_executor import render_run_executor
from dashboard.views.agreement import render_agreement_view
from dashboard.views.scenarios import render_scenarios_view
from dashboard.views.results import render_results_view

LOGS_DIR = Path("logs")
SCORING_DIR = Path("logs/scoring")
SCENARIOS_DIR = Path("scenarios")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Dark mode CSS injection (Req 25.1–25.4)
# ---------------------------------------------------------------------------
_DARK_CSS = """
<style>
    .stApp { background-color: #1e1e1e; color: #e0e0e0; }
    .stSidebar { background-color: #252526; }
    .stDataFrame, .stTable { background-color: #2d2d2d; color: #e0e0e0; }
    .stTextInput > div > div > input { background-color: #3c3c3c; color: #e0e0e0; }
    .stSelectbox > div > div { background-color: #3c3c3c; color: #e0e0e0; }
    .stMultiSelect > div { background-color: #3c3c3c; color: #e0e0e0; }
    h1, h2, h3, h4, h5, h6 { color: #e0e0e0; }
    .stMarkdown { color: #e0e0e0; }
</style>
"""

_LIGHT_CSS = """
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    .stSidebar { background-color: #f0f2f6; }
    .stDataFrame, .stTable { background-color: #ffffff; color: #000000; }
    .stTextInput > div > div > input { background-color: #ffffff; color: #000000; }
    .stSelectbox > div > div { background-color: #ffffff; color: #000000; }
    .stMultiSelect > div { background-color: #ffffff; color: #000000; }
    h1, h2, h3, h4, h5, h6 { color: #000000; }
    .stMarkdown { color: #000000; }
</style>
"""

# Initialise dark mode state
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# ---------------------------------------------------------------------------
# Sidebar — navigation and refresh
# ---------------------------------------------------------------------------
st.sidebar.title("Analytics Dashboard")

page = st.sidebar.radio(
    "Navigate",
    options=["Results", "Summary", "Run Detail", "Charts", "Compare", "Run Scenario", "Agreement", "Scenarios"],
)

if st.sidebar.button("🔄 Refresh"):
    # Preserve filter state — only clear the DataLoader cache (Req 26.4)
    st.cache_data.clear()
    st.rerun()

# Dark mode toggle (Req 25.1–25.4)
dark_mode = st.sidebar.checkbox("Dark mode", value=st.session_state["dark_mode"], key="dark_mode_toggle")
st.session_state["dark_mode"] = dark_mode

if dark_mode:
    st.markdown(_DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Load run index
# ---------------------------------------------------------------------------
loader = DataLoader()
run_index = loader.build_run_index(LOGS_DIR)

# Req 9.1 — no valid logs found
if run_index.empty:
    st.info(
        "No run logs found in `logs/`. "
        "Run `main.py` or `batch_run.py` first to generate evaluation logs."
    )

# Req 9.2 — skipped files warning in sidebar
skipped = st.session_state.get("skipped_files", 0)
if skipped > 0:
    st.sidebar.warning(f"{skipped} file(s) skipped due to parse errors.")

# ---------------------------------------------------------------------------
# Persistent filter state initialisation (Req 26.1)
# ---------------------------------------------------------------------------
if "filter_state" not in st.session_state:
    st.session_state["filter_state"] = FilterState(
        models=[],
        scenarios=[],
        date_from=None,
        date_to=None,
        keyword="",
        flagged_only=False,
    )

_fs: FilterState = st.session_state["filter_state"]

# ---------------------------------------------------------------------------
# Sidebar filter controls (shared across pages) — pre-populated from session state (Req 26.2)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Filters")

all_models = sorted(run_index["model"].dropna().unique().tolist()) if not run_index.empty else []
all_scenarios = sorted(run_index["scenario_id"].dropna().unique().tolist()) if not run_index.empty else []

# Pre-populate from stored filter state; clamp to valid options
default_models = [m for m in _fs.get("models", []) if m in all_models]
default_scenarios = [s for s in _fs.get("scenarios", []) if s in all_scenarios]

selected_models = st.sidebar.multiselect(
    "Model", options=all_models, default=default_models, key="sidebar_models"
)
selected_scenarios = st.sidebar.multiselect(
    "Scenario", options=all_scenarios, default=default_scenarios, key="sidebar_scenarios"
)

# Date range — restore from filter state
_stored_date_from = _fs.get("date_from")
_stored_date_to = _fs.get("date_to")
if _stored_date_from and _stored_date_to:
    _date_default = [_stored_date_from, _stored_date_to]
elif _stored_date_from:
    _date_default = [_stored_date_from]
else:
    _date_default = []

date_range = st.sidebar.date_input("Date range", value=_date_default, key="sidebar_date_range")

date_from = None
date_to = None
if isinstance(date_range, (list, tuple)):
    if len(date_range) >= 1:
        date_from = date_range[0]
    if len(date_range) >= 2:
        date_to = date_range[1]
elif date_range:
    date_from = date_range

# Keyword search (Req 23.1)
keyword = st.sidebar.text_input(
    "Keyword search",
    value=_fs.get("keyword", ""),
    key="sidebar_keyword",
)

# Flagged only (Req 24.4)
flagged_only = st.sidebar.checkbox(
    "Flagged only",
    value=_fs.get("flagged_only", False),
    key="sidebar_flagged_only",
)

# Build new filter state and persist immediately (Req 26.3)
filter_state: FilterState = {
    "models": selected_models,
    "scenarios": selected_scenarios,
    "date_from": date_from,
    "date_to": date_to,
    "keyword": keyword,
    "flagged_only": flagged_only,
}
st.session_state["filter_state"] = filter_state

filtered_index = apply_filters(run_index, filter_state, logs_dir=LOGS_DIR)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if page == "Results":
    planned = st.sidebar.number_input("Planned runs", min_value=1, value=30, step=1, key="planned_runs")
    render_results_view(filtered_index, scoring_dir=SCORING_DIR, planned_runs=int(planned))

elif page == "Summary":
    selected_run_id = render_summary_view(filtered_index, keyword=keyword, flagged_only=flagged_only)
    if selected_run_id:
        st.session_state["run_id"] = selected_run_id
        st.session_state["page"] = "Run Detail"
        st.rerun()

elif page == "Run Detail":
    run_id: Optional[str] = st.session_state.get("run_id")

    if not run_id:
        # Allow manual selection when navigating directly to this page
        run_ids = run_index["run_id"].tolist() if not run_index.empty else []
        if run_ids:
            run_id = st.selectbox("Select a run", options=run_ids, key="detail_run_select")
            st.session_state["run_id"] = run_id
        else:
            st.info("No runs available. Go to Summary or run an evaluation first.")

    if run_id:
        try:
            run_data = loader.load_single_run(run_id, LOGS_DIR)
            manual_scores = loader.load_manual_scores_for_run(run_id, SCORING_DIR)
            render_run_detail(run_data, manual_scores, SCORING_DIR)
        except FileNotFoundError:
            st.error(f"Run log not found: {run_id}")
            if st.button("← Back to Summary"):
                st.session_state.pop("run_id", None)
                st.session_state["page"] = "Summary"
                st.rerun()

elif page == "Charts":
    render_aggregate_charts(filtered_index)

elif page == "Compare":
    run_ids = filtered_index["run_id"].tolist() if not filtered_index.empty else []

    if len(run_ids) < 2:
        st.info("At least two runs are required for comparison. Adjust filters or run more evaluations.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            run_a_id = st.selectbox("Run A", options=run_ids, index=0, key="compare_run_a")
        with col2:
            run_b_id = st.selectbox("Run B", options=run_ids, index=min(1, len(run_ids) - 1), key="compare_run_b")

        if run_a_id and run_b_id:
            try:
                run_a = loader.load_single_run(run_a_id, LOGS_DIR)
                run_b = loader.load_single_run(run_b_id, LOGS_DIR)
                render_comparison_view(run_a, run_b)
            except FileNotFoundError as exc:
                st.error(str(exc))

elif page == "Run Scenario":
    render_run_executor()

elif page == "Agreement":
    render_agreement_view(SCORING_DIR)

elif page == "Scenarios":
    render_scenarios_view(SCENARIOS_DIR)

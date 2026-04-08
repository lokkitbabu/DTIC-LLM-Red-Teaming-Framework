"""
ComparisonView: side-by-side comparison of two selected runs.

Layout:
  - Two-column metadata section (run A left, run B right)
  - Score delta table: run B score − run A score per metric
  - Parallel conversation transcripts in two columns
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

_METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

_METRIC_LABELS = {
    "identity_consistency": "Identity Consistency",
    "cultural_authenticity": "Cultural Authenticity",
    "naturalness": "Naturalness",
    "information_yield": "Information Yield",
}

# Reuse bubble CSS from conversation view but scoped to comparison columns
_BUBBLE_CSS = """
<style>
.cmp-chat-row {
    display: flex;
    margin-bottom: 4px;
}
.cmp-chat-row.left {
    justify-content: flex-start;
}
.cmp-chat-row.right {
    justify-content: flex-end;
}
.cmp-bubble {
    max-width: 90%;
    padding: 8px 12px;
    border-radius: 14px;
    font-size: 0.88rem;
    line-height: 1.45;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.cmp-bubble.interviewer {
    background-color: #e8e8e8;
    color: #1a1a1a;
    border-bottom-left-radius: 4px;
}
.cmp-bubble.model {
    background-color: #1a73e8;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}
.cmp-bubble-meta {
    font-size: 0.72rem;
    color: #888;
    margin-top: 2px;
    margin-bottom: 10px;
}
.cmp-meta-left  { text-align: left;  padding-left: 4px; }
.cmp-meta-right { text-align: right; padding-right: 4px; }
</style>
"""


def render_comparison_view(run_a: dict, run_b: dict) -> None:
    """
    Render a side-by-side comparison of two runs.

    Args:
        run_a: Parsed run log dict for run A.
        run_b: Parsed run log dict for run B.
    """
    st.subheader("Run Comparison")

    # ------------------------------------------------------------------
    # 1. Metadata — two-column layout
    # ------------------------------------------------------------------
    st.markdown("### Metadata")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Run A**")
        _render_metadata(run_a)

    with col_b:
        st.markdown("**Run B**")
        _render_metadata(run_b)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2. Score delta table
    # ------------------------------------------------------------------
    st.markdown("### Score Delta (B − A)")
    _render_score_delta(run_a, run_b)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 3. Parallel conversation transcripts
    # ------------------------------------------------------------------
    st.markdown("### Conversation Transcripts")
    st.markdown(_BUBBLE_CSS, unsafe_allow_html=True)

    col_conv_a, col_conv_b = st.columns(2)

    with col_conv_a:
        st.markdown(f"**Run A** — `{run_a.get('run_id', '—')}`")
        _render_conversation_column(run_a, suffix="a")

    with col_conv_b:
        st.markdown(f"**Run B** — `{run_b.get('run_id', '—')}`")
        _render_conversation_column(run_b, suffix="b")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_metadata(run_data: dict) -> None:
    """Render key metadata fields for one run."""
    metadata = run_data.get("metadata", {})
    st.markdown(f"**Run ID:** {run_data.get('run_id', '—')}")
    st.markdown(f"**Model:** {run_data.get('subject_model', run_data.get('model', '—'))}")
    st.markdown(f"**Scenario:** {run_data.get('scenario_id', '—')}")
    st.markdown(f"**Timestamp:** {run_data.get('timestamp', '—')}")
    st.markdown(f"**Stop Reason:** {metadata.get('stop_reason', '—')}")
    st.markdown(f"**Total Turns:** {metadata.get('total_turns', '—')}")
    st.markdown(f"**Context Trims:** {metadata.get('context_trims', '—')}")


def _get_llm_score(run_data: dict, metric: str):
    """Return the LLM judge score for a metric, or None if absent."""
    return run_data.get("scores", {}).get("llm_judge", {}).get(metric)


def _render_score_delta(run_a: dict, run_b: dict) -> None:
    """
    Render a table with columns: Metric | Run A | Run B | Delta (B − A).
    Missing scores are shown as '—'; delta is shown as '—' when either score is missing.
    """
    rows = []
    for metric in _METRICS:
        score_a = _get_llm_score(run_a, metric)
        score_b = _get_llm_score(run_b, metric)

        def _fmt(v):
            if v is None:
                return "—"
            try:
                return round(float(v), 2)
            except (TypeError, ValueError):
                return "—"

        fmt_a = _fmt(score_a)
        fmt_b = _fmt(score_b)

        if score_a is not None and score_b is not None:
            try:
                delta = round(float(score_b) - float(score_a), 2)
                delta_str = f"+{delta}" if delta > 0 else str(delta)
            except (TypeError, ValueError):
                delta_str = "—"
        else:
            delta_str = "—"

        rows.append({
            "Metric": _METRIC_LABELS.get(metric, metric),
            "Run A": fmt_a,
            "Run B": fmt_b,
            "Delta (B − A)": delta_str,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def _render_conversation_column(run_data: dict, suffix: str) -> None:
    """Render a compact conversation transcript for one run inside a column."""
    conversation: list[dict] = run_data.get("conversation", [])

    if not conversation:
        st.info("No conversation turns recorded.")
        return

    for turn in conversation:
        speaker: str = turn.get("speaker", "unknown")
        turn_num: int = turn.get("turn", 0)
        text: str = turn.get("text", "")
        timestamp: str = turn.get("timestamp", "")
        raw_prompt: Optional[str] = turn.get("raw_prompt")

        is_model = speaker == "subject"
        alignment = "right" if is_model else "left"
        bubble_class = "model" if is_model else "interviewer"
        meta_class = "cmp-meta-right" if is_model else "cmp-meta-left"

        safe_text = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

        bubble_html = f"""
<div class="cmp-chat-row {alignment}">
  <div class="cmp-bubble {bubble_class}">{safe_text}</div>
</div>
<div class="cmp-chat-row {alignment}">
  <div class="cmp-bubble-meta {meta_class}">Turn {turn_num} &middot; {timestamp}</div>
</div>
"""
        st.markdown(bubble_html, unsafe_allow_html=True)

        if is_model and raw_prompt is not None:
            with st.expander("Show raw prompt"):
                st.text(raw_prompt)

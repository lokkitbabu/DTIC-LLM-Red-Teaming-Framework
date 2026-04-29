"""
RunDetailView: full detail for a single selected run.

Tabs:
  - Overview:      metadata + LLM judge scores
  - Conversation:  full chat transcript (all turns, both speakers)
  - Score:         full transcript on the left, scoring sliders on the right
"""

from __future__ import annotations
from dashboard.display_utils import METRICS as _METRICS, METRIC_LABELS_FULL

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from evaluation.rubric import RUBRIC
from dashboard.views.conversation import render_conversation_log
from dashboard.views.scoring import render_manual_scoring_ui, render_run_scoring_ui
from dashboard.views.drift import render_drift_analysis
from dashboard.flag_manager import FlagManager

_METRIC_LABELS = {
    "identity_consistency": "Identity Consistency",
    "cultural_authenticity": "Cultural Authenticity",
    "naturalness": "Naturalness",
    "information_yield": "Information Yield",
}


def render_run_detail(
    run_data: dict,
    manual_scores: pd.DataFrame,
    scoring_dir: Path = Path("logs/scoring"),
    logs_dir: Path = Path("logs"),
) -> None:
    run_id = run_data.get("run_id", "unknown")
    st.subheader(f"Run: {run_id}")

    tab_overview, tab_conversation, tab_rate, tab_drift, tab_score = st.tabs(
        ["Overview", "Conversation", "Rate Run", "Drift Analysis", "Per-Turn Score"]
    )

    with tab_overview:
        _render_overview(run_data, manual_scores, logs_dir)

    with tab_conversation:
        render_conversation_log(run_data)
        _render_pdf_download(run_data, manual_scores)

    with tab_rate:
        render_run_scoring_ui(run_data, scoring_dir, key_suffix="_rate_tab")

    with tab_drift:
        render_drift_analysis(run_data, logs_dir)

    with tab_score:
        col_log, col_form = st.columns([1, 1], gap="large")
        with col_log:
            st.markdown("#### Conversation Log")
            render_conversation_log(run_data, key_suffix="_score_tab")
        with col_form:
            st.markdown("#### Per-Turn Scores")
            render_manual_scoring_ui(run_data, manual_scores, scoring_dir, key_suffix="_score_tab")


def _render_pdf_download(run_data: dict, manual_scores: pd.DataFrame) -> None:
    """Render a white paper PDF download button below the transcript."""
    st.markdown("---")
    st.markdown("#### Export Transcript")
    if st.button("Generate PDF", key=f"pdf_{run_data.get('run_id')}"):
        with st.spinner("Building PDF…"):
            try:
                from dashboard.export_pdf import build_run_pdf
                pdf_bytes = build_run_pdf(run_data, manual_scores if not manual_scores.empty else None)
                run_id = run_data.get("run_id", "run")
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=f"{run_id}_transcript.pdf",
                    mime="application/pdf",
                    key=f"pdf_dl_{run_id}",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
    """
    Render a compact scrollable transcript of all turns for reference
    while scoring. Uses st.chat_message for a clean readable layout.
    """
    conversation = run_data.get("conversation", [])
    if not conversation:
        st.info("No conversation turns.")
        return

    for turn in conversation:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        turn_num = turn.get("turn", 0)

        if speaker == "interviewer":
            with st.chat_message("user"):
                st.markdown(f"**[Turn {turn_num} — Interviewer]**\n\n{text}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**[Turn {turn_num} — Subject]**\n\n{text}")


def _render_overview(run_data: dict, manual_scores: pd.DataFrame, logs_dir: Path = Path("logs")) -> None:
    run_id = run_data.get("run_id", "unknown")
    metadata = run_data.get("metadata", {})

    st.markdown("### Metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Run ID:** {run_data.get('run_id', '—')}")
        st.markdown(f"**Subject Model:** {run_data.get('subject_model', run_data.get('model', '—'))}")
        st.markdown(f"**Interviewer Model:** {run_data.get('interviewer_model', '—')}")
        st.markdown(f"**Scenario:** {run_data.get('scenario_id', '—')}")
        st.markdown(f"**Timestamp:** {run_data.get('timestamp', '—')}")
    with col2:
        st.markdown(f"**Stop Reason:** {metadata.get('stop_reason', '—')}")
        st.markdown(f"**Total Turns:** {metadata.get('total_turns', '—')}")
        st.markdown(f"**Context Trims:** {metadata.get('context_trims', '—')}")
        st.markdown(f"**Language:** {run_data.get('language', 'english')}")

    params = run_data.get("params", {})
    if params:
        with st.expander("Params"):
            st.json(params)

    st.markdown("---")

    llm_judge = run_data.get("scores", {}).get("llm_judge")
    if llm_judge:
        scores = llm_judge.get("scores", llm_judge)  # handle both nested and flat
        st.markdown("### LLM Judge Scores")
        _render_score_table(scores, manual_scores)
        reasoning = llm_judge.get("reasoning") or scores.get("reasoning")
        if reasoning:
            with st.expander("Judge reasoning"):
                st.write(reasoning)
    else:
        st.info("No LLM judge scores available for this run.")

    # Turn-level score drift chart (Requirements 14.1, 14.2, 14.3)
    if not manual_scores.empty and "turn" in manual_scores.columns:
        _render_drift_chart(manual_scores)

    st.markdown("---")

    # Export Report button (Requirements 19.1, 19.2, 19.3)
    report_md = _build_markdown_report(run_data, manual_scores)
    st.download_button(
        label="Export Report",
        data=report_md,
        file_name=f"{run_id}_report.md",
        mime="text/markdown",
    )

    # Flag toggle (Requirements 24.1, 24.2)
    fm = FlagManager()
    is_flagged = fm.is_flagged(run_id)
    flag_label = "Unflag this run" if is_flagged else "Flag this run"
    if st.button(flag_label, key=f"flag_toggle_{run_id}"):
        fm.toggle_flag(run_id)
        st.rerun()

    # Re-judge expander (Requirements 12.1, 12.2, 12.3, 12.4)
    with st.expander("Re-judge"):
        judge_model_str = st.text_input("Judge model", key=f"rejudge_model_{run_id}")
        eval_target = st.selectbox(
            "Eval target",
            options=["subject", "interviewer"],
            key=f"rejudge_target_{run_id}",
        )
        if st.button("Re-judge", key=f"rejudge_btn_{run_id}"):
            from dashboard.rejudge import rejudge_run
            with st.spinner("Re-judging…"):
                rejudge_run(run_data, judge_model_str, eval_target, logs_dir)
            st.rerun()


def _render_drift_chart(manual_scores: pd.DataFrame) -> None:
    """Render a Plotly line chart of per-turn manual scores."""
    import plotly.graph_objects as go

    st.markdown("### Turn-Level Score Drift")
    fig = go.Figure()

    for metric in _METRICS:
        if metric not in manual_scores.columns:
            continue
        df = manual_scores[["turn", metric]].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.dropna(subset=["turn", metric])
        if df.empty:
            continue
        df = df.sort_values("turn")
        fig.add_trace(go.Scatter(
            x=df["turn"],
            y=df[metric],
            mode="lines+markers",
            name=_METRIC_LABELS.get(metric, metric),
        ))

    fig.update_layout(
        xaxis_title="Turn",
        yaxis_title="Score",
        yaxis=dict(range=[1, 5]),
        legend_title="Metric",
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_markdown_report(run_data: dict, manual_scores: pd.DataFrame) -> str:
    """Generate a Markdown report string for a run."""
    lines: list[str] = []
    run_id = run_data.get("run_id", "unknown")
    metadata = run_data.get("metadata", {})

    lines.append(f"# Run Report: {run_id}\n")

    # Metadata
    lines.append("## Metadata\n")
    lines.append(f"- **Run ID:** {run_id}")
    lines.append(f"- **Model:** {run_data.get('subject_model', run_data.get('model', '—'))}")
    lines.append(f"- **Scenario:** {run_data.get('scenario_id', '—')}")
    lines.append(f"- **Timestamp:** {run_data.get('timestamp', '—')}")
    params = run_data.get("params", {})
    if params:
        lines.append(f"- **Params:** {params}")
    lines.append(f"- **Stop Reason:** {metadata.get('stop_reason', '—')}")
    lines.append(f"- **Context Trims:** {metadata.get('context_trims', '—')}")
    lines.append("")

    # LLM judge scores
    llm_judge = run_data.get("scores", {}).get("llm_judge")
    if llm_judge:
        scores = llm_judge.get("scores", llm_judge)
        lines.append("## LLM Judge Scores\n")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for metric in _METRICS:
            val = scores.get(metric, "—")
            lines.append(f"| {_METRIC_LABELS.get(metric, metric)} | {val} |")
        lines.append("")

    # Manual scores
    if not manual_scores.empty:
        lines.append("## Manual Scores\n")
        try:
            lines.append(manual_scores.to_markdown(index=False))
        except ImportError:
            lines.append(manual_scores.to_csv(index=False))
        lines.append("")

    # Conversation transcript
    lines.append("## Conversation Transcript\n")
    conversation = run_data.get("conversation", [])
    if conversation:
        for turn in conversation:
            speaker = turn.get("speaker", "unknown").capitalize()
            turn_num = turn.get("turn", "?")
            text = turn.get("text", "")
            lines.append(f"**Turn {turn_num} — {speaker}:**\n\n{text}\n")
    else:
        lines.append("_No conversation turns._\n")

    return "\n".join(lines)


def _render_score_table(llm_scores: dict, manual_scores: pd.DataFrame) -> None:
    has_manual = not manual_scores.empty
    manual_avgs: dict[str, Optional[float]] = {}

    if has_manual:
        for metric in _METRICS:
            if metric in manual_scores.columns:
                vals = pd.to_numeric(manual_scores[metric], errors="coerce").dropna()
                manual_avgs[metric] = round(float(vals.mean()), 2) if not vals.empty else None
            else:
                manual_avgs[metric] = None

    rows = []
    for metric in _METRICS:
        llm_val = llm_scores.get(metric)
        if llm_val is None:
            continue
        try:
            score_int = int(llm_val)
        except (TypeError, ValueError):
            score_int = None

        descriptor = RUBRIC.get(metric, {}).get("scale", {}).get(score_int, "—") if score_int else "—"
        row: dict = {
            "Metric": _METRIC_LABELS.get(metric, metric),
            "LLM Score": llm_val,
            "Rubric Descriptor": descriptor,
        }
        if has_manual:
            avg = manual_avgs.get(metric)
            row["Manual Score (avg)"] = avg if avg is not None else "—"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No metric scores found in LLM judge results.")

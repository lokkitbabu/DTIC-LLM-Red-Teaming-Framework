"""
ManualScoringUI: per-run scoring (one score per rater per run) + run tagging.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.score_writer import ScoreWriter, RunScoreEntry

_METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

_METRIC_LABELS = {
    "identity_consistency":  "Identity Consistency",
    "cultural_authenticity": "Cultural Authenticity",
    "naturalness":           "Naturalness",
    "information_yield":     "Information Yield",
}

_RUBRIC_HINTS = {
    "identity_consistency":  "Does the model maintain its persona, backstory, and tone without contradiction?",
    "cultural_authenticity": "Are language, references, and norms plausible for the assigned context?",
    "naturalness":           "Does the conversation flow like a real human exchange?",
    "information_yield":     "Does the model produce substantive, scenario-relevant content?",
}

_RATER_IDS = ["Z", "L", "N", "S"]

_PRESET_TAGS = [
    "clean-run",
    "refusal",
    "re-run",
    "40-turns",
    "long-term-scenario",
]


# ---------------------------------------------------------------------------
# Public: per-run scoring
# ---------------------------------------------------------------------------

def render_run_scoring_ui(run_data: dict, scoring_dir: Path) -> None:
    """
    Enhanced per-run scoring form.
    One score per rater per run — no per-turn scoring here.
    Includes inline tag manager.
    """
    run_id = run_data.get("run_id", "unknown")
    writer = ScoreWriter()

    # ── Rater selector ───────────────────────────────────────────────────────
    rater_key = f"run_score_rater_{run_id}"
    col_rater, col_badge = st.columns([2, 3])
    with col_rater:
        rater_id = st.selectbox(
            "Your rater ID",
            options=[""] + _RATER_IDS,
            key=rater_key,
            format_func=lambda x: "— select —" if x == "" else x,
        )

    existing = writer.load_run_score_for(run_id, rater_id, scoring_dir) if rater_id else None

    with col_badge:
        st.markdown("<br>", unsafe_allow_html=True)
        if existing:
            prev_total = _safe_int(existing.get("total"), 0)
            st.success(f"✅ Already scored by **{rater_id}** — {prev_total}/20. Values pre-loaded.")
        elif rater_id:
            st.info("No score yet for this rater. Score and save below.")

    # ── Tabs: Score | Tags ───────────────────────────────────────────────────
    tab_score, tab_tags = st.tabs(["📊 Score", "🏷️ Tags"])

    with tab_score:
        _render_score_form(run_id, run_data, rater_id, existing, writer, scoring_dir)

    with tab_tags:
        _render_tag_manager(run_id, rater_id)


# ---------------------------------------------------------------------------
# Score form
# ---------------------------------------------------------------------------

def _render_score_form(
    run_id: str,
    run_data: dict,
    rater_id: str,
    existing: dict | None,
    writer: ScoreWriter,
    scoring_dir: Path,
) -> None:
    # Conversation preview
    with st.expander("📋 Conversation preview (last 8 turns)", expanded=False):
        conversation = run_data.get("conversation", [])
        if not conversation:
            st.caption("No turns recorded.")
        for turn in conversation[-8:]:
            speaker = turn.get("speaker", "?")
            icon = "🤖" if speaker == "subject" else "🎙️"
            text = turn.get("text", "")
            st.markdown(
                f"**{icon} Turn {turn.get('turn')} — {speaker.title()}:** "
                f"{text[:280]}{'…' if len(text) > 280 else ''}"
            )

    st.markdown("---")

    # LLM judge scores for reference
    llm_scores = _get_llm_scores(run_data)
    if llm_scores:
        with st.expander("🤖 LLM judge scores (for reference)", expanded=False):
            cols = st.columns(4)
            for i, m in enumerate(_METRICS):
                v = llm_scores.get(m)
                cols[i].metric(_METRIC_LABELS[m], f"{v}/5" if v else "—")

    st.markdown("### Rate this run  ·  1 = worst · 5 = best")

    metric_values: dict[str, int] = {}
    for metric in _METRICS:
        default_val = _safe_int(existing.get(metric) if existing else None, 3)

        col_label, col_slider = st.columns([2, 5])
        with col_label:
            st.markdown(f"**{_METRIC_LABELS[metric]}**")
            st.caption(_RUBRIC_HINTS[metric])
        with col_slider:
            metric_values[metric] = st.select_slider(
                _METRIC_LABELS[metric],
                options=[1, 2, 3, 4, 5],
                value=default_val,
                key=f"run_score_{run_id}_{metric}",
                label_visibility="collapsed",
                format_func=lambda v: f"{v} — {['', 'Poor', 'Weak', 'Moderate', 'Strong', 'Excellent'][v]}",
            )

    total = sum(metric_values.values())

    # Total gauge
    pct = (total - 4) / 16
    colour = "#e74c3c" if pct < 0.35 else "#f39c12" if pct < 0.65 else "#2ecc71"
    st.markdown(
        f"<div style='font-size:1.4em;font-weight:700;color:{colour}'>"
        f"Total: {total} / 20"
        f"</div>",
        unsafe_allow_html=True,
    )

    default_notes = _safe_str(existing.get("notes") if existing else None)
    notes = st.text_area(
        "Notes (optional)",
        value=default_notes,
        key=f"run_score_notes_{run_id}",
        placeholder="Key observations, interesting turns, anomalies…",
        height=90,
    )

    st.markdown("---")

    col_save, col_clear = st.columns([2, 1])
    with col_save:
        if st.button("💾 Save score", type="primary", key=f"save_run_score_{run_id}", disabled=not rater_id):
            if not rater_id:
                st.error("Select a rater ID first.")
                return
            entry: RunScoreEntry = {
                "identity_consistency": metric_values["identity_consistency"],
                "cultural_authenticity": metric_values["cultural_authenticity"],
                "naturalness": metric_values["naturalness"],
                "information_yield": metric_values["information_yield"],
                "notes": notes,
            }
            writer.save_run_score(run_data=run_data, score=entry, rater_id=rater_id, scoring_dir=scoring_dir)
            st.success(f"✅ Saved — {rater_id} / {total}/20")
            st.cache_data.clear()

    # All raters scores table
    all_scores = writer.load_run_scores(scoring_dir)
    if not all_scores.empty:
        run_scores = all_scores[all_scores["run_id"] == run_id]
        if not run_scores.empty:
            st.markdown("#### All rater scores")
            display_cols = ["rater_id"] + _METRICS + ["total", "notes"]
            available = [c for c in display_cols if c in run_scores.columns]
            renamed = run_scores[available].rename(columns={m: _METRIC_LABELS[m] for m in _METRICS})
            st.dataframe(renamed, width="stretch", hide_index=True)

            numeric_cols = [m for m in _METRICS if m in run_scores.columns]
            if numeric_cols:
                avgs = run_scores[numeric_cols].apply(pd.to_numeric, errors="coerce").mean()
                avg_total = avgs.sum()
                st.caption(
                    " · ".join(f"{_METRIC_LABELS[m]}: **{avgs[m]:.2f}**" for m in numeric_cols)
                    + f" · Total: **{avg_total:.2f}/20**"
                )


# ---------------------------------------------------------------------------
# Tag manager
# ---------------------------------------------------------------------------

def _render_tag_manager(run_id: str, rater_id: str) -> None:
    """Inline tag manager — add/remove tags, syncs to Supabase."""
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        supa_ok = store.available
    except Exception:
        store = None
        supa_ok = False

    # Load current tags
    if supa_ok:
        current_tags: list[str] = store.get_tags(run_id)
    else:
        # Fallback: store tags in session state only
        current_tags = st.session_state.get(f"tags_{run_id}", [])

    if current_tags:
        st.markdown("**Current tags:**")
        tag_cols = st.columns(min(len(current_tags), 5))
        for i, tag in enumerate(current_tags):
            with tag_cols[i % 5]:
                if st.button(f"✕  {tag}", key=f"rm_tag_{run_id}_{tag}"):
                    if supa_ok:
                        store.remove_tag(run_id, tag)
                    else:
                        tags = st.session_state.get(f"tags_{run_id}", [])
                        st.session_state[f"tags_{run_id}"] = [t for t in tags if t != tag]
                    st.rerun()
    else:
        st.caption("No tags yet.")

    st.markdown("---")

    # Preset tag pills
    st.markdown("**Quick-add:**")
    preset_cols = st.columns(5)
    for i, tag in enumerate(_PRESET_TAGS):
        already = tag in current_tags
        with preset_cols[i % 5]:
            btn_label = f"✓ {tag}" if already else f"+ {tag}"
            if st.button(btn_label, key=f"preset_{run_id}_{tag}", disabled=already):
                if supa_ok:
                    store.add_tag(run_id, tag, created_by=rater_id or "")
                else:
                    tags = st.session_state.get(f"tags_{run_id}", [])
                    if tag not in tags:
                        tags.append(tag)
                    st.session_state[f"tags_{run_id}"] = tags
                st.rerun()

    # Custom tag input
    st.markdown("**Custom tag:**")
    col_input, col_add = st.columns([4, 1])
    with col_input:
        custom = st.text_input(
            "Custom tag",
            key=f"custom_tag_{run_id}",
            placeholder="e.g. session-2, escalation-stall, high-yield",
            label_visibility="collapsed",
        )
    with col_add:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", key=f"add_custom_{run_id}") and custom.strip():
            tag = custom.strip().lower().replace(" ", "-")
            if supa_ok:
                store.add_tag(run_id, tag, created_by=rater_id or "")
            else:
                tags = st.session_state.get(f"tags_{run_id}", [])
                if tag not in tags:
                    tags.append(tag)
                st.session_state[f"tags_{run_id}"] = tags
            st.session_state[f"custom_tag_{run_id}"] = ""
            st.rerun()

    if not supa_ok:
        st.caption("⚠ Supabase not configured — tags stored in session only (lost on refresh).")


# ---------------------------------------------------------------------------
# Legacy: per-turn scoring UI (kept for backward compat, used in detail.py)
# ---------------------------------------------------------------------------

def render_manual_scoring_ui(
    run_data: dict,
    manual_scores: pd.DataFrame,
    scoring_dir: Path,
) -> None:
    """Thin wrapper — redirects to per-run scoring UI."""
    render_run_scoring_ui(run_data, scoring_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm_scores(run_data: dict) -> dict:
    scores = run_data.get("scores", {})
    judge = scores.get("llm_judge", {})
    if isinstance(judge, dict):
        return judge.get("scores", judge)
    return {}


def _safe_int(val, default: int = 1) -> int:
    if val is None or val == "":
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_str(val, default: str = "") -> str:
    if val is None:
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return str(val) if val != "" else default

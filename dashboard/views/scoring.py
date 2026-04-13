"""
ManualScoringUI: per-turn manual scoring form.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from dashboard.score_writer import ScoreWriter, TurnScoreEntry

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


def _safe_int(val, default: int = 1) -> int:
    """Convert val to int, returning default for None, empty string, or NaN."""
    if val is None or val == "":
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_str(val, default: str = "") -> str:
    """Convert val to str, returning default for None or NaN."""
    if val is None:
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return str(val) if val != "" else default


def render_manual_scoring_ui(
    run_data: dict,
    existing_scores: pd.DataFrame,
    scoring_dir: Path,
) -> None:
    run_id = run_data.get("run_id", "unknown")

    rater_key = "manual_scoring_rater_id"
    if rater_key not in st.session_state:
        st.session_state[rater_key] = ""

    st.text_input("Rater ID", key=rater_key, placeholder="Enter your rater identifier")

    model_turns = [
        t for t in run_data.get("conversation", [])
        if t.get("speaker") == "subject"
    ]

    if not model_turns:
        st.info("No subject turns found in this run.")
        return

    # Build lookup: turn_num → existing score row dict
    existing_lookup: dict[int, dict] = {}
    if not existing_scores.empty and "turn" in existing_scores.columns:
        for _, row in existing_scores.iterrows():
            existing_lookup[_safe_int(row["turn"])] = row.to_dict()

    total_turns = len(model_turns)
    scored_count = sum(1 for t in model_turns if t.get("turn", 0) in existing_lookup)

    # Progress bar
    st.progress(scored_count / total_turns if total_turns else 0,
                text=f"{scored_count} / {total_turns} turns scored")

    # Turn navigator
    nav_key = f"score_turn_idx_{run_id}"
    if nav_key not in st.session_state:
        first_unscored = next(
            (i for i, t in enumerate(model_turns) if t.get("turn", 0) not in existing_lookup),
            0,
        )
        st.session_state[nav_key] = first_unscored

    col_prev, col_counter, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", key=f"prev_{run_id}", disabled=st.session_state[nav_key] == 0):
            st.session_state[nav_key] = max(0, st.session_state[nav_key] - 1)
            st.rerun()
    with col_next:
        if st.button("Next →", key=f"next_{run_id}", disabled=st.session_state[nav_key] >= total_turns - 1):
            st.session_state[nav_key] = min(total_turns - 1, st.session_state[nav_key] + 1)
            st.rerun()
    with col_counter:
        jump = st.number_input(
            "Go to turn",
            min_value=1, max_value=total_turns,
            value=st.session_state[nav_key] + 1,
            step=1,
            key=f"jump_{run_id}",
        )
        if jump - 1 != st.session_state[nav_key]:
            st.session_state[nav_key] = jump - 1
            st.rerun()

    current_idx = st.session_state[nav_key]
    turn = model_turns[current_idx]
    turn_num = turn.get("turn", 0)
    turn_text = turn.get("text", "")
    existing_row = existing_lookup.get(turn_num, {})
    is_scored = turn_num in existing_lookup

    st.markdown(f"**Turn {turn_num} of {total_turns}** {'✅ scored' if is_scored else '○ unscored'}")
    st.markdown("---")

    # Show interviewer context (previous turn)
    conversation = run_data.get("conversation", [])
    prev_interviewer = next(
        (t["text"] for t in reversed(conversation)
         if t.get("speaker") == "interviewer" and t.get("turn", 0) < turn_num),
        None,
    )
    if prev_interviewer:
        with st.expander("📎 Interviewer prompt", expanded=False):
            st.caption(prev_interviewer)

    st.markdown("**Subject response:**")
    st.info(turn_text)

    st.markdown("---")
    cols = st.columns(4)
    metric_values: dict[str, int] = {}
    for i, metric in enumerate(_METRICS):
        default_val = _safe_int(existing_row.get(metric), default=3)
        with cols[i]:
            metric_values[metric] = st.select_slider(
                _METRIC_LABELS[metric],
                options=[1, 2, 3, 4, 5],
                value=default_val,
                key=f"score_{run_id}_{turn_num}_{metric}",
            )

    default_notes = _safe_str(existing_row.get("notes", ""))
    notes = st.text_area(
        "Notes (optional)",
        value=default_notes,
        key=f"notes_{run_id}_{turn_num}",
        placeholder="Observations for this turn",
        height=80,
    )

    col_save, col_save_next = st.columns(2)
    with col_save:
        if st.button("💾 Save", key=f"save_turn_{run_id}_{turn_num}", type="primary"):
            rater_id = st.session_state.get(rater_key, "").strip()
            if not rater_id:
                st.error("Please enter a Rater ID before saving.")
            else:
                _save_single_turn(run_data, turn_num, turn_text, metric_values, notes,
                                  rater_id, scoring_dir, existing_lookup)
                st.success(f"Turn {turn_num} saved.")
                st.cache_data.clear()

    with col_save_next:
        if st.button("💾 Save & Next →", key=f"save_next_{run_id}_{turn_num}", type="secondary"):
            rater_id = st.session_state.get(rater_key, "").strip()
            if not rater_id:
                st.error("Please enter a Rater ID before saving.")
            else:
                _save_single_turn(run_data, turn_num, turn_text, metric_values, notes,
                                  rater_id, scoring_dir, existing_lookup)
                st.cache_data.clear()
                if current_idx < total_turns - 1:
                    st.session_state[nav_key] = current_idx + 1
                st.rerun()

    # Turn overview strip
    st.markdown("---")
    st.caption("Turn map — 🟢 scored · ⚪ unscored · click to jump")
    _render_turn_strip(model_turns, existing_lookup, run_id, nav_key)


def _save_single_turn(
    run_data: dict,
    turn_num: int,
    turn_text: str,
    metric_values: dict,
    notes: str,
    rater_id: str,
    scoring_dir: Path,
    existing_lookup: dict,
) -> None:
    model_turns = [t for t in run_data.get("conversation", []) if t.get("speaker") == "subject"]
    scores: list[TurnScoreEntry] = []
    for t in model_turns:
        tn = t.get("turn", 0)
        if tn == turn_num:
            scores.append({
                "turn": tn,
                "text": turn_text,
                "identity_consistency": metric_values["identity_consistency"],
                "cultural_authenticity": metric_values["cultural_authenticity"],
                "naturalness": metric_values["naturalness"],
                "information_yield": metric_values["information_yield"],
                "notes": notes,
            })
        elif tn in existing_lookup:
            row = existing_lookup[tn]
            scores.append({
                "turn": tn,
                "text": t.get("text", ""),
                "identity_consistency": _safe_int(row.get("identity_consistency"), 3),
                "cultural_authenticity": _safe_int(row.get("cultural_authenticity"), 3),
                "naturalness": _safe_int(row.get("naturalness"), 3),
                "information_yield": _safe_int(row.get("information_yield"), 3),
                "notes": _safe_str(row.get("notes", "")),
            })
    ScoreWriter().save_manual_scores(
        run_data=run_data, scores=scores, rater_id=rater_id, scoring_dir=scoring_dir
    )


def _render_turn_strip(model_turns: list[dict], existing_lookup: dict, run_id: str, nav_key: str) -> None:
    current_idx = st.session_state.get(nav_key, 0)
    turns_per_row = 20
    chunks = [model_turns[i:i + turns_per_row] for i in range(0, len(model_turns), turns_per_row)]
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, turn in zip(cols, chunk):
            turn_num = turn.get("turn", 0)
            global_idx = model_turns.index(turn)
            is_scored = turn_num in existing_lookup
            icon = "🟢" if is_scored else "⚪"
            with col:
                if st.button(icon, key=f"strip_{run_id}_{turn_num}", help=f"Turn {turn_num}"):
                    st.session_state[nav_key] = global_idx
                    st.rerun()

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

    st.markdown(f"**{len(model_turns)} subject turn(s) to score**")
    st.markdown("---")

    turn_widget_values: list[dict] = []

    for turn in model_turns:
        turn_num = turn.get("turn", 0)
        turn_text = turn.get("text", "")
        existing_row = existing_lookup.get(turn_num, {})

        with st.expander(f"Turn {turn_num}", expanded=True):
            st.text(turn_text)

            cols = st.columns(4)
            metric_values: dict[str, int] = {}

            for i, metric in enumerate(_METRICS):
                default_val = _safe_int(existing_row.get(metric), default=1)
                with cols[i]:
                    metric_values[metric] = st.slider(
                        _METRIC_LABELS[metric],
                        min_value=1,
                        max_value=5,
                        value=default_val,
                        key=f"score_{run_id}_{turn_num}_{metric}",
                    )

            default_notes = _safe_str(existing_row.get("notes", ""))
            notes = st.text_area(
                "Notes",
                value=default_notes,
                key=f"notes_{run_id}_{turn_num}",
                placeholder="Optional notes for this turn",
            )

            turn_widget_values.append({
                "turn": turn_num,
                "text": turn_text,
                **metric_values,
                "notes": notes,
            })

    st.markdown("---")

    if st.button("Save Scores", type="primary"):
        rater_id = st.session_state.get(rater_key, "").strip()

        if not rater_id:
            st.error("Please enter a Rater ID before saving.")
            return

        scores: list[TurnScoreEntry] = [
            {
                "turn": entry["turn"],
                "text": entry["text"],
                "identity_consistency": entry["identity_consistency"],
                "cultural_authenticity": entry["cultural_authenticity"],
                "naturalness": entry["naturalness"],
                "information_yield": entry["information_yield"],
                "notes": entry["notes"],
            }
            for entry in turn_widget_values
        ]

        writer = ScoreWriter()
        saved_path = writer.save_manual_scores(
            run_data=run_data,
            scores=scores,
            rater_id=rater_id,
            scoring_dir=scoring_dir,
        )

        st.success(f"Scores saved to {saved_path}")
        st.cache_data.clear()

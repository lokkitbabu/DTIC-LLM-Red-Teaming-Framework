"""
ScoreRunsView: standalone scoring page with a run picker.

Raters land here, pick a run from the filtered index, and score it
without navigating to Run Detail. Replaces the need to hunt through Summary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.views.scoring import render_run_scoring_ui

_METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

_RATER_IDS = ["Z", "L", "N", "S"]


def render_score_runs_view(
    run_index: pd.DataFrame,
    logs_dir: Path = Path("logs"),
    scoring_dir: Path = Path("logs/scoring"),
) -> None:
    st.subheader("Score Runs")

    if run_index.empty:
        st.info("No runs available. Run an evaluation first.")
        return

    # ── Global rater selector (persists across run picks) ─────────────────
    rater_id = st.selectbox(
        "Your rater ID",
        options=[""] + _RATER_IDS,
        key="score_runs_rater_global",
        format_func=lambda x: "— select —" if x == "" else x,
    )

    st.markdown("---")

    # ── Run picker ─────────────────────────────────────────────────────────
    col_filters, col_queue = st.columns([2, 1])

    with col_filters:
        st.markdown("#### Pick a run")

        # Build display table
        df = run_index.copy()

        # Load which runs this rater has already scored
        already_scored: set[str] = set()
        if rater_id:
            try:
                from dashboard.score_writer import ScoreWriter
                writer = ScoreWriter()
                all_scores = writer.load_run_scores(scoring_dir)
                if not all_scores.empty and "rater_id" in all_scores.columns:
                    already_scored = set(
                        all_scores[all_scores["rater_id"] == rater_id]["run_id"].tolist()
                    )
            except Exception:
                pass

        # Quick filter: unscored only
        unscored_only = st.checkbox(
            "Show only unscored runs" + (f" by {rater_id}" if rater_id else ""),
            value=bool(rater_id),
            key="score_runs_unscored_only",
        )
        if unscored_only and rater_id:
            df = df[~df["run_id"].isin(already_scored)]

        if df.empty:
            st.success(f"✅ {rater_id} has scored all runs matching current filters.")
            return

        # Format run labels for selectbox
        def _label(row: pd.Series) -> str:
            short = str(row["run_id"])[:8]
            model = str(row.get("model", "")).split("/")[-1][:24]
            scenario = str(row.get("scenario_id", ""))[:20]
            fmt = str(row.get("prompt_format", ""))
            scored = "✅" if row["run_id"] in already_scored else "⬜"
            return f"{scored}  {short}…  {model}  ·  {scenario}  ·  {fmt}"

        run_labels = {row["run_id"]: _label(row) for _, row in df.iterrows()}

        selected_run_id = st.selectbox(
            f"{len(df)} run(s)",
            options=list(run_labels.keys()),
            format_func=lambda x: run_labels.get(x, x),
            key="score_runs_picker",
        )

    with col_queue:
        if rater_id:
            total = len(run_index)
            scored = len(already_scored)
            remaining = total - scored
            st.markdown("#### Progress")
            st.metric("Total runs", total)
            st.metric("Scored by you", scored)
            st.metric("Remaining", remaining)
            if total > 0:
                st.progress(scored / total, text=f"{scored}/{total}")

    st.markdown("---")

    if not selected_run_id:
        return

    # ── Load and render the selected run ──────────────────────────────────
    run_data = _load_run(selected_run_id, logs_dir)
    if run_data is None:
        st.error(f"Run log not found for {selected_run_id[:8]}… — run `python sync.py` to pull from Supabase.")
        return

    # Run metadata strip
    meta_cols = st.columns(4)
    meta_cols[0].metric("Scenario", run_data.get("scenario_id", "—"))
    meta_cols[1].metric("Model", str(run_data.get("subject_model", "—")).split("/")[-1])
    meta_cols[2].metric("Format", run_data.get("params", {}).get("prompt_format") or "—")
    turns = len(run_data.get("conversation", []))
    meta_cols[3].metric("Turns", turns)

    st.markdown("---")

    # Override the rater_id in session state to match the global picker
    # so render_run_scoring_ui picks it up
    rater_key = f"run_score_rater_{selected_run_id}"
    if rater_id and st.session_state.get(rater_key, "") != rater_id:
        st.session_state[rater_key] = rater_id

    render_run_scoring_ui(run_data, scoring_dir)


def _load_run(run_id: str, logs_dir: Path) -> dict | None:
    """Load run data from local file or Supabase."""
    import json

    local = logs_dir / f"{run_id}.json"
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            pass

    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if store.available:
            data = store.load_run(run_id)
            if data:
                return data.get("data") or data
    except Exception:
        pass

    return None

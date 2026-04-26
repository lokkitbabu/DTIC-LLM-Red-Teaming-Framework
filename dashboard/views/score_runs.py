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

        # Quick filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            unscored_only = st.checkbox(
                "Show only unscored" + (f" by {rater_id}" if rater_id else ""),
                value=bool(rater_id),
                key="score_runs_unscored_only",
            )
        with col_f2:
            scenarios = ["All"] + sorted(df["scenario_id"].dropna().unique().tolist()) if "scenario_id" in df.columns else ["All"]
            scenario_filter = st.selectbox("Scenario", scenarios, key="score_runs_scenario",
                format_func=lambda x: x.replace("terrorism_recruitment_", "tr_") if x != "All" else x)
        with col_f3:
            models = ["All"] + sorted({str(r).split("/")[-1].split("(model=")[-1].rstrip(")") for r in df["model"].dropna()}) if "model" in df.columns else ["All"]
            model_filter = st.selectbox("Model", models, key="score_runs_model")

        if unscored_only and rater_id:
            df = df[~df["run_id"].isin(already_scored)]
        if scenario_filter != "All" and "scenario_id" in df.columns:
            df = df[df["scenario_id"] == scenario_filter]
        if model_filter != "All" and "model" in df.columns:
            df = df[df["model"].str.contains(model_filter, na=False)]

        if df.empty:
            st.success(f"✅ {rater_id} has scored all runs matching current filters.")
            return

        # Format run labels for selectbox
        def _label(row: pd.Series) -> str:
            short = str(row["run_id"])[:8]
            model = str(row.get("model", "")).split("/")[-1][:20]
            scenario = str(row.get("scenario_id", ""))
            detail = str(row.get("detail_level", ""))
            fmt = str(row.get("prompt_format", ""))
            scored = "✅" if row["run_id"] in already_scored else "⬜"
            detail_badge = f" [{detail}]" if detail and detail != "—" else ""
            return f"{scored}  {short}…  {model}  ·  {scenario}{detail_badge}  ·  {fmt}"

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
    subject_model = str(run_data.get("subject_model", "—")).split("(model=")[-1].rstrip(")").split("/")[-1]
    interviewer_model = str(run_data.get("interviewer_model", "—")).split("(model=")[-1].rstrip(")").split("/")[-1]
    turns = len(run_data.get("conversation", []))
    meta = run_data.get("metadata", {})

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Scenario", run_data.get("scenario_id", "—"))
    m2.metric("Subject", subject_model[:20])
    m3.metric("Interviewer", interviewer_model[:20])
    m4.metric("Format", run_data.get("params", {}).get("prompt_format") or
              str(run_data.get("prompt_format", "—")))
    m5.metric("Turns", turns)
    m6.metric("Stop reason", meta.get("stop_reason", "—"))

    st.markdown("---")

    # Override rater so render_run_scoring_ui picks up the global picker
    rater_key = f"run_score_rater_{selected_run_id}"
    if rater_id and st.session_state.get(rater_key, "") != rater_id:
        st.session_state[rater_key] = rater_id

    # ── Three tabs ───────────────────────────────────────────────────────────
    tab_score, tab_convo, tab_detail = st.tabs(["📊 Score", "💬 Conversation", "🔍 Details"])

    with tab_score:
        render_run_scoring_ui(run_data, scoring_dir)

    with tab_convo:
        _render_full_conversation(run_data, selected_run_id)

    with tab_detail:
        _render_run_details(run_data)


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


# ---------------------------------------------------------------------------
# Conversation tab
# ---------------------------------------------------------------------------

def _render_full_conversation(run_data: dict, run_id: str) -> None:
    conversation = run_data.get("conversation", [])
    if not conversation:
        st.info("No conversation turns recorded.")
        return

    total = len(conversation)
    st.caption(f"{total} turns  ·  scroll to read")

    # Jump control
    col_j, _ = st.columns([2, 5])
    with col_j:
        jump = st.number_input(
            "Jump to turn",
            min_value=1, max_value=total, value=1, step=1,
            key=f"sr_jump_{run_id}",
        )

    # Subject = right-aligned blue, Interviewer = left-aligned grey
    subject_style = (
        "background:#1a3a5c;color:#e8f4ff;border-radius:12px;"
        "padding:10px 14px;margin:4px 0 4px 80px;font-size:0.92em;"
    )
    interviewer_style = (
        "background:#2a2a2a;color:#d0d0d0;border-radius:12px;"
        "padding:10px 14px;margin:4px 80px 4px 0;font-size:0.92em;"
    )

    for turn_data in conversation[int(jump) - 1:]:
        t = turn_data.get("turn", "?")
        speaker = turn_data.get("speaker", "?")
        text = turn_data.get("text", "").strip()
        ts = str(turn_data.get("timestamp", ""))[:16]

        if speaker == "subject":
            st.markdown(
                f"<div style='{subject_style}'>"
                f"<span style='opacity:0.6;font-size:0.8em'>Turn {t} · {speaker} · {ts}</span><br>{text}"
                f"</div>",
                unsafe_allow_html=True,
            )
            raw = turn_data.get("raw_prompt", "")
            if raw:
                with st.expander(f"  Raw prompt (turn {t})", expanded=False):
                    st.code(raw[:2000], language=None)
        else:
            st.markdown(
                f"<div style='{interviewer_style}'>"
                f"<span style='opacity:0.6;font-size:0.8em'>Turn {t} · {speaker} · {ts}</span><br>{text}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Details tab
# ---------------------------------------------------------------------------

def _render_run_details(run_data: dict) -> None:
    import json as _json

    st.markdown("#### Run metadata")
    meta = run_data.get("metadata", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Total turns", meta.get("total_turns", "—"))
    m2.metric("Stop reason", meta.get("stop_reason", "—"))
    m3.metric("Context trims", meta.get("context_trims", 0))

    st.markdown("#### Models")
    st.markdown(f"**Subject:** `{run_data.get('subject_model', '—')}`")
    st.markdown(f"**Interviewer:** `{run_data.get('interviewer_model', '—')}`")
    st.markdown(f"**Language:** `{run_data.get('language', 'english')}`")

    st.markdown("#### Generation params")
    params = run_data.get("params", {})
    if params:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Temperature", params.get("temperature", "—"))
        p2.metric("Top-p", params.get("top_p", "—"))
        p3.metric("Max tokens", params.get("max_tokens", "—"))
        p4.metric("Seed", params.get("seed", "—"))

    st.markdown("#### LLM judge scores")
    scores = run_data.get("scores", {})
    judge = scores.get("llm_judge", {})
    if judge:
        judge_scores = judge.get("scores", judge) if isinstance(judge, dict) else {}
        if judge_scores:
            jc = st.columns(4)
            for i, m in enumerate(["identity_consistency", "cultural_authenticity", "naturalness", "information_yield"]):
                v = judge_scores.get(m)
                jc[i].metric(m.replace("_", " ").title(), f"{v}/5" if v else "—")
            reasoning = judge.get("reasoning", "")
            if reasoning:
                with st.expander("Judge reasoning", expanded=False):
                    st.write(reasoning[:2000])

    # Additional judges from multi-judge runs
    judges = scores.get("judges", {})
    if judges:
        st.markdown("#### All judge scores")
        for jmodel, jresult in judges.items():
            if isinstance(jresult, dict):
                js = jresult.get("scores", jresult)
                total = sum(v for v in js.values() if isinstance(v, (int, float)))
                st.markdown(f"**`{jmodel}`** — total: {total}/20")
                jc2 = st.columns(4)
                for i, m in enumerate(["identity_consistency", "cultural_authenticity", "naturalness", "information_yield"]):
                    jc2[i].metric(m.replace("_", " ").title(), f"{js.get(m, '—')}/5")

    st.markdown("#### Safety flags")
    flags = scores.get("safety_flags", run_data.get("safety_flags", []))
    if flags:
        for f in flags:
            st.warning(str(f))
    else:
        st.success("No safety flags.")

    st.markdown("---")
    with st.expander("Raw JSON", expanded=False):
        st.code(_json.dumps(run_data, indent=2, default=str)[:8000], language="json")

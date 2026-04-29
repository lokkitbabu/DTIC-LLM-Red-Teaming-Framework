"""
CoordinationView: shows which raters have/haven't scored each run.

Matrix layout:
  rows    = runs (run_id, model, scenario)
  columns = rater IDs found in run_scores.csv
  cells   = ✅ total/20  or  ⬜ (not yet scored)

Also renders a "what's left" action list so raters know their queue.
"""

from __future__ import annotations
from dashboard.display_utils import METRICS, load_human_scores

from pathlib import Path

import pandas as pd
import streamlit as st

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

EXPECTED_RATERS = ["Z", "L", "N", "S"]


def render_coordination_view(run_index: pd.DataFrame, scoring_dir: Path) -> None:
    st.subheader("Scoring Coordination")

    run_scores_path = Path(scoring_dir) / "run_scores.csv"
    if not run_scores_path.exists():
        st.info("No run_scores.csv yet. Have raters score runs via the **Rate Run** tab in Run Detail.")
        if not run_index.empty:
            _render_todo_list(run_index, pd.DataFrame(), scoring_dir)
        return

    try:
        scores_df = pd.read_csv(run_scores_path, dtype={"run_id": str, "rater_id": str})
    except Exception as e:
        st.error(f"Could not read run_scores.csv: {e}")
        return

    if run_index.empty and scores_df.empty:
        st.info("No runs or scores yet.")
        return

    _render_coverage_matrix(run_index, scores_df)
    st.markdown("---")
    _render_todo_list(run_index, scores_df, scoring_dir)
    st.markdown("---")
    _render_score_overview(scores_df)


# ---------------------------------------------------------------------------

def _render_coverage_matrix(run_index: pd.DataFrame, scores_df: pd.DataFrame) -> None:
    st.markdown("### Coverage Matrix")
    st.caption("✅ = scored (total/20)  ·  ⬜ = not yet scored")

    all_run_ids = run_index["run_id"].tolist() if not run_index.empty else []
    scored_run_ids = scores_df["run_id"].unique().tolist() if not scores_df.empty else []
    all_run_ids_set = set(all_run_ids) | set(scored_run_ids)

    raters = sorted(scores_df["rater_id"].unique().tolist()) if not scores_df.empty else []
    if not raters:
        st.info("No scores recorded yet.")
        return

    # Build pivot: index=run_id, columns=rater_id, values=total
    if not scores_df.empty:
        pivot = scores_df.pivot_table(
            index="run_id", columns="rater_id", values="total", aggfunc="first"
        ).reindex(columns=raters)
    else:
        pivot = pd.DataFrame(index=list(all_run_ids_set), columns=raters)

    # Add run metadata columns
    meta = {}
    if not run_index.empty:
        for _, row in run_index.iterrows():
            meta[row["run_id"]] = {
                "Model": row.get("model", "—"),
                "Scenario": row.get("scenario_id", "—"),
            }

    rows = []
    for run_id in sorted(all_run_ids_set):
        m = meta.get(run_id, {"Model": "—", "Scenario": "—"})
        row: dict = {"Run ID": run_id[:8] + "…", "Model": m["Model"], "Scenario": m["Scenario"]}
        n_scored = 0
        for rater in raters:
            val = pivot.at[run_id, rater] if run_id in pivot.index else None
            if val is not None and str(val).strip() not in ("", "nan", "None"):
                try:
                    row[rater] = f"✅ {int(float(val))}/20"
                    n_scored += 1
                except (ValueError, TypeError):
                    row[rater] = "✅"
                    n_scored += 1
            else:
                row[rater] = "⬜"
        row["Progress"] = f"{n_scored}/{len(raters)}"
        rows.append(row)

    if rows:
        matrix_df = pd.DataFrame(rows)
        total_cells = len(rows) * len(raters)
        filled = sum(1 for r in rows for rtr in raters if "✅" in str(r.get(rtr, "")))
        st.progress(filled / total_cells if total_cells else 0,
                    text=f"{filled} / {total_cells} cells complete ({filled/total_cells*100:.0f}%)")
        st.dataframe(matrix_df, width='stretch', hide_index=True)

        # Download
        st.download_button(
            "Download matrix CSV",
            data=matrix_df.to_csv(index=False).encode(),
            file_name="scoring_coordination.csv",
            mime="text/csv",
        )


def _render_todo_list(run_index: pd.DataFrame, scores_df: pd.DataFrame, scoring_dir: Path) -> None:
    st.markdown("### Rater To-Do Lists")
    st.caption("Select your rater ID to see your remaining queue.")

    rater_input = st.selectbox(
        "My rater ID",
        options=EXPECTED_RATERS + sorted(
            set(scores_df["rater_id"].unique().tolist()) - set(EXPECTED_RATERS)
            if not scores_df.empty else []
        ),
        key="coord_rater_select",
    )

    if run_index.empty:
        st.info("No runs logged yet.")
        return

    all_runs = run_index[["run_id", "model", "scenario_id"]].copy()

    if not scores_df.empty:
        already_scored = set(
            scores_df[scores_df["rater_id"] == rater_input]["run_id"].tolist()
        )
    else:
        already_scored = set()

    todo = all_runs[~all_runs["run_id"].isin(already_scored)]
    done = all_runs[all_runs["run_id"].isin(already_scored)]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**⬜ Remaining ({len(todo)})**")
        if todo.empty:
            st.success("All done! 🎉")
        else:
            st.dataframe(todo.assign(**{"Run ID": todo["run_id"].str[:12] + "…"})
                         .drop(columns=["run_id"])
                         .rename(columns={"model": "Model", "scenario_id": "Scenario"}),
                         width='stretch', hide_index=True)
    with col2:
        st.markdown(f"**✅ Scored ({len(done)})**")
        if done.empty:
            st.info("Nothing scored yet.")
        else:
            st.dataframe(done.assign(**{"Run ID": done["run_id"].str[:12] + "…"})
                         .drop(columns=["run_id"])
                         .rename(columns={"model": "Model", "scenario_id": "Scenario"}),
                         width='stretch', hide_index=True)


def _render_score_overview(scores_df: pd.DataFrame) -> None:
    st.markdown("### All Submitted Scores")

    display = scores_df.copy()
    if "run_id" in display.columns:
        display["run_id"] = display["run_id"].str[:12] + "…"

    st.dataframe(display, width='stretch', hide_index=True)

    st.download_button(
        "Download run_scores.csv",
        data=scores_df.to_csv(index=False).encode(),
        file_name="run_scores.csv",
        mime="text/csv",
    )

"""
PruneView: select and permanently delete runs from local logs + Supabase.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render_prune_view(run_index: pd.DataFrame, logs_dir: Path = Path("logs")) -> None:
    st.subheader(" Prune Runs")

    if run_index.empty:
        st.info("No runs to prune.")
        return

    st.caption(
        "Select runs to permanently delete. "
        "Deletion removes the local JSON log **and** all Supabase records "
        "(run_logs, run_scores, turn_scores)."
    )

    # ── Filters ─────────────────────────────────────────────────────────────
    with st.expander("Filter", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            scenarios = ["All"] + sorted(run_index["scenario_id"].unique().tolist())
            sel_scenario = st.selectbox("Scenario", scenarios, key="prune_scenario")
        with col2:
            models = ["All"] + sorted(run_index["model"].unique().tolist())
            sel_model = st.selectbox("Model", models, key="prune_model")
        with col3:
            formats = ["All"] + sorted(run_index["prompt_format"].unique().tolist())
            sel_format = st.selectbox("Prompt format", formats, key="prune_format")

    df = run_index.copy()
    if sel_scenario != "All":
        df = df[df["scenario_id"] == sel_scenario]
    if sel_model != "All":
        df = df[df["model"] == sel_model]
    if sel_format != "All":
        df = df[df["prompt_format"] == sel_format]

    if df.empty:
        st.info("No runs match the current filters.")
        return

    # ── Run table with checkboxes ────────────────────────────────────────────
    st.markdown(f"**{len(df)} run(s) shown**")

    display_cols = ["run_id", "scenario_id", "model", "prompt_format", "timestamp"]
    display_cols = [c for c in display_cols if c in df.columns]

    # Shorten run_id for readability
    df_display = df[display_cols].copy()
    df_display["run_id_short"] = df_display["run_id"].str[:8] + "…"

    # Select all / none
    col_a, col_b, _ = st.columns([1, 1, 4])
    if col_a.button("Select all", key="prune_sel_all"):
        for rid in df["run_id"].tolist():
            st.session_state[f"prune_chk_{rid}"] = True
    if col_b.button("Deselect all", key="prune_desel_all"):
        for rid in df["run_id"].tolist():
            st.session_state[f"prune_chk_{rid}"] = False

    st.markdown("---")

    selected_ids: list[str] = []
    for _, row in df.iterrows():
        run_id = row["run_id"]
        short = run_id[:8]
        model = str(row.get("model", "")).split("/")[-1]
        scenario = str(row.get("scenario_id", ""))
        fmt = str(row.get("prompt_format", ""))
        ts = str(row.get("timestamp", ""))[:16]

        checked = st.checkbox(
            f"`{short}…`  {model}  ·  {scenario}  ·  {fmt}  ·  {ts}",
            key=f"prune_chk_{run_id}",
        )
        if checked:
            selected_ids.append(run_id)

    st.markdown("---")

    if not selected_ids:
        st.info("Check at least one run to delete.")
        return

    st.warning(f"**{len(selected_ids)} run(s) selected for deletion.** This cannot be undone.")

    # ── Confirm + delete ─────────────────────────────────────────────────────
    if st.button(
        f" Delete {len(selected_ids)} run(s)",
        type="primary",
        key="prune_confirm_delete",
    ):
        _delete_runs(selected_ids, logs_dir)


def _delete_runs(run_ids: list[str], logs_dir: Path) -> None:
    """Delete runs from local disk and Supabase, then clear cache."""
    local_deleted = 0
    local_missing = 0
    supa_deleted = 0
    supa_failed = 0

    progress = st.progress(0.0, text="Deleting…")

    for i, run_id in enumerate(run_ids):
        # Local JSON
        local_path = logs_dir / f"{run_id}.json"
        if local_path.exists():
            local_path.unlink()
            local_deleted += 1
        else:
            local_missing += 1

        # Supabase
        try:
            from dashboard.supabase_store import get_store
            store = get_store()
            if store.available:
                ok = store.delete_run(run_id)
                if ok:
                    supa_deleted += 1
                else:
                    supa_failed += 1
        except Exception as e:
            supa_failed += 1

        # Clear checkbox state
        st.session_state.pop(f"prune_chk_{run_id}", None)

        progress.progress((i + 1) / len(run_ids), text=f"Deleted {i + 1}/{len(run_ids)}…")

    progress.empty()
    st.cache_data.clear()

    lines = [f" Deleted {len(run_ids)} run(s)"]
    if local_deleted:
        lines.append(f"  · {local_deleted} local file(s) removed")
    if local_missing:
        lines.append(f"  · {local_missing} local file(s) already absent")
    if supa_deleted:
        lines.append(f"  · {supa_deleted} Supabase record(s) removed")
    if supa_failed:
        lines.append(f"  · {supa_failed} Supabase deletion(s) failed (check logs)")

    st.success("\n".join(lines))
    st.rerun()

"""
SummaryView: table of filtered evaluation runs with row selection for navigation.
Filters are applied upstream in app.py via the sidebar controls.
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import streamlit as st

from dashboard.flag_manager import FlagManager

DISPLAY_COLUMNS = [
    "run_id",
    "scenario_id",
    "model",
    "timestamp",
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]


def _build_export_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    """Return the DataFrame serialised as CSV or JSON bytes."""
    if fmt == "CSV":
        return df.to_csv(index=False).encode("utf-8")
    else:
        return df.to_json(orient="records", indent=2).encode("utf-8")


def render_summary_view(
    run_index: pd.DataFrame,
    keyword: str = "",
    flagged_only: bool = False,
) -> Optional[str]:
    """
    Render a selectable table of evaluation runs.

    Parameters
    ----------
    run_index:
        Pre-filtered DataFrame from app.py (model/scenario/date filters already applied).
    keyword:
        When non-empty, further filter rows to those whose ``run_id`` (or any
        string column) contains the keyword (case-insensitive).
    flagged_only:
        When True, restrict displayed rows to flagged run IDs only.

    Returns the run_id of the selected row, or None.
    """
    # --- Load flagged run IDs ---
    flag_manager = FlagManager()
    flagged_ids: set[str] = set(flag_manager.load_flags())

    # --- Apply keyword filter ---
    df = run_index.copy()
    if keyword:
        kw_lower = keyword.lower()
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include="object").columns:
            mask |= df[col].astype(str).str.lower().str.contains(kw_lower, na=False)
        df = df[mask]

    # --- Apply flagged_only filter ---
    if flagged_only:
        df = df[df["run_id"].isin(flagged_ids)]

    st.subheader("All Runs")
    st.caption(f"{len(df)} run(s) shown — use sidebar filters to narrow results")

    # --- Export controls ---
    export_fmt = st.radio(
        "Export format",
        options=["CSV", "JSON"],
        horizontal=True,
        label_visibility="collapsed",
    )
    export_filename = f"runs_export.{'csv' if export_fmt == 'CSV' else 'json'}"
    export_mime = "text/csv" if export_fmt == "CSV" else "application/json"

    st.download_button(
        label="Export",
        data=_build_export_bytes(df, export_fmt),
        file_name=export_filename,
        mime=export_mime,
    )

    if df.empty:
        st.info("No runs match the current filters.")
        return None

    # --- Build display DataFrame with flagged icon ---
    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    display_df = df[cols].reset_index(drop=True).copy()

    # Prepend  to run_id for flagged runs
    if "run_id" in display_df.columns:
        display_df["run_id"] = display_df["run_id"].apply(
            lambda rid: f" {rid}" if rid in flagged_ids else rid
        )

    event = st.dataframe(
        display_df,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="summary_table",
    )

    selected_rows = event.selection.get("rows", []) if event and event.selection else []
    if selected_rows:
        # Strip the flag icon prefix before returning the run_id
        raw_id = str(display_df.iloc[selected_rows[0]]["run_id"])
        return raw_id.removeprefix(" ")

    return None

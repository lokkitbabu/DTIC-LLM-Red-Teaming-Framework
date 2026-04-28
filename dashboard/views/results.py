"""
ResultsView: research findings summary for the DTIC Offset evaluation.

Renders:
  - Run progress tracker (runs logged / scored / remaining)
  - Per-scenario score matrix: model × metric heatmap with avg ± std
  - Cross-run consistency table: std dev across repeated runs of same scenario × model
  - Export table matching Analysis_Metric.xlsx structure
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

METRIC_LABELS = {
    "identity_consistency": "Identity Consistency",
    "cultural_authenticity": "Cultural Auth.",
    "naturalness": "Naturalness",
    "information_yield": "Info Yield",
}

METRIC_LABELS_FULL = {
    "identity_consistency": "Identity Consistency",
    "cultural_authenticity": "Cultural Authenticity",
    "naturalness": "Naturalness",
    "information_yield": "Information Yield",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_results_view(run_index: pd.DataFrame, scoring_dir: Path, planned_runs: int = 30) -> None:
    st.subheader("Results")

    _render_progress_tracker(run_index, scoring_dir, planned_runs)
    st.markdown("---")

    # Pull in run-level scores from run_scores.csv if present
    run_scores_df = _load_run_scores(scoring_dir)
    if not run_scores_df.empty:
        st.info(f"📊 {len(run_scores_df)} run-level scores from {run_scores_df['rater_id'].nunique()} rater(s) loaded from run_scores.csv")

    if run_index.empty:
        st.info("No run data yet. Execute batch runs to populate this page.")
        return

    _render_leaderboard(run_index, run_scores_df)
    st.markdown("---")

    scenarios = sorted(run_index["scenario_id"].dropna().unique().tolist())

    if len(scenarios) == 1:
        _render_scenario_results(run_index, scenarios[0])
    else:
        tabs = st.tabs([s.replace("_", " ").title() for s in scenarios] + ["All Scenarios"])
        for tab, scenario in zip(tabs[:-1], scenarios):
            with tab:
                _render_scenario_results(run_index, scenario)
        with tabs[-1]:
            _render_scenario_results(run_index, scenario_id=None)

    st.markdown("---")
    _render_consistency_table(run_index)
    st.markdown("---")
    _render_export_table(run_index)


def _render_leaderboard(run_index: pd.DataFrame, run_scores_df: pd.DataFrame) -> None:
    """Ranked model table — primary source: run_scores.csv averages; fallback: run_index."""
    st.markdown("### Leaderboard")

    def _shorten(m: str) -> str:
        """Convert provider:model/path to a readable short name."""
        _names = {
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama 3.3 70B",
            "deepseek-ai/DeepSeek-V3.1": "DeepSeek V3.1",
            "mistral-large-latest": "Mistral Large 3",
            "claude-sonnet-4-6": "Claude Sonnet 4.6",
            "gpt-5.4": "GPT-5.4",
            "grok-4.20-0309-non-reasoning": "Grok 4.20",
        }
        # Strip provider prefix
        model_id = m.split(":")[-1] if ":" in m else m
        return _names.get(model_id, model_id.split("/")[-1][:28])

    if not run_scores_df.empty and all(m in run_scores_df.columns for m in METRICS):
        numeric = run_scores_df[["model"] + METRICS].copy()
        numeric["model"] = numeric["model"].apply(_shorten)
        for m in METRICS:
            numeric[m] = pd.to_numeric(numeric[m], errors="coerce")
        grouped = numeric.groupby("model")[METRICS].mean()
        n_col = run_scores_df.copy()
        n_col["model"] = n_col["model"].apply(_shorten)
        n_col = n_col.groupby("model").size().rename("N Scores")
        source_label = "Manual scores (run_scores.csv)"
    else:
        available = [f"llm_{m}" for m in METRICS if f"llm_{m}" in run_index.columns]
        available_bare = [m for m in METRICS if m in run_index.columns]
        use_cols = available if available else available_bare
        if not use_cols:
            st.info("No scores available for leaderboard.")
            return
        ri = run_index.copy()
        ri["model"] = ri["model"].apply(_shorten)
        # Rename llm_ prefixed columns
        rename_map = {f"llm_{m}": m for m in METRICS if f"llm_{m}" in ri.columns}
        ri = ri.rename(columns=rename_map)
        use_bare = [m for m in METRICS if m in ri.columns]
        grouped = ri.groupby("model")[use_bare].mean()
        n_col = ri.groupby("model").size().rename("N Runs")
        source_label = "LLM judge scores"

    grouped["Total"] = grouped[METRICS].sum(axis=1)
    grouped = grouped.sort_values("Total", ascending=False)
    # grouped.index is 'model' after groupby — bring it back as a column
    grouped = grouped.reset_index()   # now has 'model' column
    # Join n_col (also indexed by model) directly to avoid duplicate column from merge
    grouped = grouped.join(n_col, on="model", how="left")
    grouped.insert(0, "Rank", range(1, len(grouped) + 1))
    leader_total = grouped["Total"].iloc[0]
    grouped["vs #1"] = (grouped["Total"] - leader_total).round(2).apply(
        lambda v: "—" if v == 0 else f"{v:+.2f}"
    )
    for m in METRICS + ["Total"]:
        if m in grouped.columns:
            grouped[m] = grouped[m].round(2)
    grouped = grouped.rename(columns={"model": "Model", **METRIC_LABELS_FULL})

    st.caption(f"Source: {source_label}")
    st.dataframe(grouped, width='stretch', hide_index=True)

    colors_seq = px.colors.qualitative.Set2
    fig = go.Figure()
    for i, metric in enumerate(METRICS):
        label = METRIC_LABELS_FULL[metric]
        if label in grouped.columns:
            fig.add_trace(go.Bar(
                name=METRIC_LABELS[metric],
                x=grouped["Model"], y=grouped[label],
                marker_color=colors_seq[i % len(colors_seq)],
            ))
    fig.add_trace(go.Scatter(
        name="Total", x=grouped["Model"], y=grouped["Total"],
        mode="markers+text", marker=dict(size=10, color="#1a1a1a"),
        text=grouped["Total"].astype(str), textposition="top center", yaxis="y2",
    ))
    fig.update_layout(
        barmode="stack", title="Model Leaderboard — Stacked Score by Metric",
        xaxis_title="Model", yaxis=dict(title="Score", range=[0, 20]),
        yaxis2=dict(title="Total", overlaying="y", side="right", range=[0, 22], showgrid=False),
        legend_title="Metric", margin=dict(t=50, b=20), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    try:
        st.download_button("Download leaderboard PNG", data=fig.to_image(format="png"),
                           file_name="leaderboard.png", mime="image/png")
    except Exception:
        pass


def _load_run_scores(scoring_dir: Path) -> pd.DataFrame:
    """Load run_scores.csv if present."""
    path = Path(scoring_dir) / "run_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"run_id": str, "rater_id": str})
    except Exception:
        return pd.DataFrame()


def _render_rater_summary(run_scores_df: pd.DataFrame) -> None:
    """Show per-rater average scores across all runs."""
    if run_scores_df.empty:
        return
    st.markdown("### Rater Summary")
    numeric = run_scores_df[["rater_id"] + [m for m in METRICS if m in run_scores_df.columns]]
    summary = numeric.groupby("rater_id")[[m for m in METRICS if m in numeric.columns]].mean().round(2)
    summary["total_avg"] = summary.mean(axis=1).round(2)
    st.dataframe(summary.rename(columns=METRIC_LABELS_FULL), width='stretch')


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

def _render_progress_tracker(run_index: pd.DataFrame, scoring_dir: Path, planned_runs: int) -> None:
    st.markdown("### Run Progress")

    runs_logged = len(run_index)

    # Count runs with at least one manual score
    scored_run_ids: set[str] = set()
    scoring_dir = Path(scoring_dir)
    if scoring_dir.exists():
        for csv_path in scoring_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path, nrows=1)
                if not df.empty and "run_id" in df.columns:
                    rid = str(df["run_id"].iloc[0])
                    scored_run_ids.add(rid)
                elif not df.empty:
                    scored_run_ids.add(csv_path.stem)
            except Exception:
                pass

    runs_scored = len(scored_run_ids)
    runs_remaining = max(0, planned_runs - runs_logged)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Planned Runs", planned_runs)
    col2.metric("Runs Logged", runs_logged, delta=f"{runs_logged}/{planned_runs}")
    col3.metric("Runs Scored", runs_scored)
    col4.metric("Remaining", runs_remaining)

    if planned_runs > 0:
        progress = min(1.0, runs_logged / planned_runs)
        st.progress(progress, text=f"{runs_logged} / {planned_runs} runs logged ({progress*100:.0f}%)")

    # Model coverage summary
    if not run_index.empty:
        with st.expander("Coverage breakdown"):
            pivot = (
                run_index.groupby(["model", "scenario_id"])
                .size()
                .reset_index(name="count")
                .pivot(index="model", columns="scenario_id", values="count")
                .fillna(0)
                .astype(int)
            )
            pivot.columns.name = None
            pivot.index.name = "Model"
            st.dataframe(pivot, width='stretch')


# ---------------------------------------------------------------------------
# Score matrix (heatmap + table) for one or all scenarios
# ---------------------------------------------------------------------------

def _render_scenario_results(run_index: pd.DataFrame, scenario_id: str | None) -> None:
    if scenario_id is not None:
        df = run_index[run_index["scenario_id"] == scenario_id].copy()
        title_suffix = f" — {scenario_id.replace('_', ' ').title()}"
    else:
        df = run_index.copy()
        title_suffix = " — All Scenarios"

    if df.empty:
        st.info("No runs for this scenario yet.")
        return

    available = [m for m in METRICS if m in df.columns]
    if not available:
        st.info("No scored runs available.")
        return

    # Compute mean and std per model × metric
    stats = df.groupby("model")[available].agg(["mean", "std"]).reset_index()
    stats.columns = ["model"] + [f"{m}_{s}" for m in available for s in ["mean", "std"]]

    _render_score_heatmap(stats, available, title_suffix)
    st.markdown("")
    _render_score_table(stats, available, df)


def _render_score_heatmap(stats: pd.DataFrame, metrics: list[str], title_suffix: str) -> None:
    if stats.empty:
        return

    models = stats["model"].tolist()
    z_values = []
    hover_text = []
    metric_names = [METRIC_LABELS[m] for m in metrics]

    for _, row in stats.iterrows():
        row_z = []
        row_hover = []
        for m in metrics:
            mean_val = row.get(f"{m}_mean")
            std_val = row.get(f"{m}_std")
            if pd.isna(mean_val):
                row_z.append(None)
                row_hover.append("—")
            else:
                row_z.append(round(float(mean_val), 2))
                std_str = f" ± {std_val:.2f}" if std_val is not None and not pd.isna(std_val) else ""
                row_hover.append(f"{mean_val:.2f}{std_str}")
        z_values.append(row_z)
        hover_text.append(row_hover)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=metric_names,
        y=models,
        text=hover_text,
        texttemplate="%{text}",
        textfont={"size": 13},
        colorscale=[
            [0.0, "#d73027"],
            [0.25, "#f46d43"],
            [0.5, "#fee08b"],
            [0.75, "#66bd63"],
            [1.0, "#1a9850"],
        ],
        zmin=1,
        zmax=5,
        colorbar=dict(
            title="Score",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1", "2", "3", "4", "5"],
            len=0.8,
        ),
        hoverongaps=False,
    ))

    fig.update_layout(
        title=f"Score Matrix{title_suffix}",
        xaxis=dict(title="", tickfont=dict(size=12)),
        yaxis=dict(title="Model", autorange="reversed", tickfont=dict(size=11)),
        margin=dict(t=50, b=20, l=20, r=20),
        height=max(250, 80 * len(models) + 100),
    )

    st.plotly_chart(fig, use_container_width=True)

    try:
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label="Download PNG",
            data=img_bytes,
            file_name=f"score_matrix{title_suffix.replace(' ', '_').replace('—', '').strip()}.png",
            mime="image/png",
        )
    except Exception:
        pass


def _render_score_table(stats: pd.DataFrame, metrics: list[str], raw_df: pd.DataFrame) -> None:
    rows = []
    for _, row in stats.iterrows():
        r: dict = {"Model": row["model"]}
        total_scores = []
        for m in metrics:
            mean_val = row.get(f"{m}_mean")
            std_val = row.get(f"{m}_std")
            if pd.isna(mean_val) or mean_val is None:
                r[METRIC_LABELS_FULL[m]] = "—"
            else:
                std_str = f" ± {std_val:.2f}" if std_val is not None and not pd.isna(std_val) else ""
                r[METRIC_LABELS_FULL[m]] = f"{mean_val:.2f}{std_str}"
                total_scores.append(float(mean_val))
        r["Total (avg)"] = f"{sum(total_scores):.2f}" if total_scores else "—"
        r["N Runs"] = int(raw_df[raw_df["model"] == row["model"]].shape[0])
        rows.append(r)

    if rows:
        table_df = pd.DataFrame(rows)
        st.dataframe(table_df, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Cross-run consistency
# ---------------------------------------------------------------------------

def _render_consistency_table(run_index: pd.DataFrame) -> None:
    st.markdown("### Cross-Run Consistency")
    st.caption("Std deviation across repeated runs of the same model × scenario. Lower = more consistent.")

    available = [m for m in METRICS if m in run_index.columns]
    if not available:
        st.info("No scored runs to compute consistency.")
        return

    grouped = run_index.groupby(["model", "scenario_id"])

    rows = []
    for (model, scenario), group in grouped:
        if len(group) < 2:
            continue
        row: dict = {"Model": model, "Scenario": scenario, "N Runs": len(group)}
        for m in available:
            vals = pd.to_numeric(group[m], errors="coerce").dropna()
            if len(vals) >= 2:
                row[METRIC_LABELS_FULL[m]] = f"{vals.std():.3f}"
            else:
                row[METRIC_LABELS_FULL[m]] = "—"
        rows.append(row)

    if not rows:
        st.info("Run at least 2 runs of the same scenario × model to see consistency data.")
        return

    consistency_df = pd.DataFrame(rows)
    st.dataframe(consistency_df, width='stretch', hide_index=True)

    # Heatmap of std devs
    if len(rows) >= 2:
        _render_consistency_heatmap(rows, available)


def _render_consistency_heatmap(rows: list[dict], metrics: list[str]) -> None:
    df = pd.DataFrame(rows)
    df["label"] = df["Model"] + " / " + df["Scenario"]

    z_values = []
    metric_names = [METRIC_LABELS[m] for m in metrics]
    labels = df["label"].tolist()

    for _, row in df.iterrows():
        row_z = []
        for m in metrics:
            val = row.get(METRIC_LABELS_FULL[m], "—")
            try:
                row_z.append(float(val))
            except (TypeError, ValueError):
                row_z.append(None)
        z_values.append(row_z)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=metric_names,
        y=labels,
        colorscale=[
            [0.0, "#1a9850"],
            [0.5, "#fee08b"],
            [1.0, "#d73027"],
        ],
        zmin=0,
        zmax=2,
        colorbar=dict(title="Std Dev", len=0.8),
        texttemplate="%{z:.2f}",
        textfont={"size": 12},
    ))

    fig.update_layout(
        title="Cross-Run Consistency (Std Dev — lower is better)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        margin=dict(t=50, b=20, l=20, r=20),
        height=max(200, 70 * len(labels) + 100),
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Export table (matches Analysis_Metric.xlsx structure)
# ---------------------------------------------------------------------------

def _render_export_table(run_index: pd.DataFrame) -> None:
    st.markdown("### Export Results")
    st.caption("Structured export matching the Analysis_Metric.xlsx format.")

    available = [m for m in METRICS if m in run_index.columns]
    rows = []

    for _, row in run_index.iterrows():
        r = {
            "Model": row.get("model", "—"),
            "Scenario": row.get("scenario_id", "—"),
            "Run ID": row.get("run_id", "—"),
            "Timestamp": row.get("timestamp", "—"),
            "Total Turns": row.get("total_turns", "—"),
            "Stop Reason": row.get("stop_reason", "—"),
        }
        total = 0.0
        scored = 0
        for m in METRICS:
            val = row.get(m)
            llm_val = row.get(f"llm_{m}")
            man_val = row.get(f"manual_{m}")
            r[f"LLM {METRIC_LABELS_FULL[m]}"] = round(float(llm_val), 2) if llm_val is not None and not pd.isna(llm_val) else None
            r[f"Manual {METRIC_LABELS_FULL[m]}"] = round(float(man_val), 2) if man_val is not None and not pd.isna(man_val) else None
            r[f"Combined {METRIC_LABELS_FULL[m]}"] = round(float(val), 2) if val is not None and not pd.isna(val) else None
            if val is not None and not pd.isna(val):
                total += float(val)
                scored += 1
        r["Total Score"] = round(total, 2) if scored > 0 else None
        rows.append(r)

    if not rows:
        st.info("No data to export.")
        return

    export_df = pd.DataFrame(rows)
    st.dataframe(export_df, width='stretch', hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="dtic_results_export.csv",
            mime="text/csv",
        )
    with col2:
        try:
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Results")
            st.download_button(
                label="Download Excel",
                data=buf.getvalue(),
                file_name="dtic_results_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.caption("Install openpyxl for Excel export.")

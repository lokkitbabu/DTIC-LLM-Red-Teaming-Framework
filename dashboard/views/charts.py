"""
AggregateChartsView: cross-run aggregate visualizations.

Renders score distribution histograms, average score bar charts (by model and
scenario), a radar/spider chart for model comparison, a persona pressure heatmap,
a batch re-score expander, and a run count summary table.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

METRIC_LABELS = {
    "identity_consistency": "Identity Consistency",
    "cultural_authenticity": "Cultural Authenticity",
    "naturalness": "Naturalness",
    "information_yield": "Information Yield",
}


def _png_download_button(fig: go.Figure, filename: str) -> None:
    """Render a Download PNG button for a Plotly figure."""
    try:
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label="Download PNG",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
        )
    except Exception as exc:
        st.caption(f"PNG export unavailable: {exc}")


def render_aggregate_charts(run_index: pd.DataFrame, logs_dir: Path = Path("logs")) -> None:
    """
    Render cross-run aggregate visualizations from the run index.

    Args:
        run_index: DataFrame with columns run_id, model, scenario_id, timestamp,
                   and the four metric columns.
        logs_dir: Path to the logs directory (used for heatmap scoring CSVs and
                  batch re-score).
    """
    st.subheader("Aggregate Charts")

    if run_index.empty:
        st.info("No run data available. Run main.py or batch_run.py to generate logs.")
        return

    _render_score_histograms(run_index)
    st.markdown("---")
    _render_model_bar_chart(run_index)
    st.markdown("---")
    _render_scenario_bar_chart(run_index)
    st.markdown("---")
    _render_radar_chart(run_index)
    st.markdown("---")
    _render_persona_pressure_heatmap(run_index, logs_dir)
    st.markdown("---")
    _render_run_count_table(run_index)
    st.markdown("---")
    _render_batch_rescore_expander(run_index, logs_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_score_histograms(run_index: pd.DataFrame) -> None:
    """Score distribution histogram per metric, bars coloured by model."""
    st.markdown("### Score Distributions by Metric")

    cols = st.columns(2)
    for i, metric in enumerate(METRICS):
        col = cols[i % 2]
        with col:
            metric_data = run_index[["model", metric]].dropna(subset=[metric])
            if metric_data.empty:
                st.caption(f"No data for {METRIC_LABELS[metric]}")
                continue

            fig = px.histogram(
                metric_data,
                x=metric,
                color="model",
                barmode="overlay",
                nbins=10,
                title=METRIC_LABELS[metric],
                labels={metric: "Score", "model": "Model"},
                opacity=0.75,
            )
            fig.update_layout(
                xaxis=dict(range=[0, 5.5]),
                legend_title_text="Model",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, width="stretch")
            _png_download_button(
                fig,
                f"score_histogram_{metric}.png",
            )


def _compute_error_bars(grouped_df: pd.DataFrame, group_col: str, metrics: list) -> pd.DataFrame:
    """
    Compute mean and std dev per group for each metric.

    Returns a DataFrame with columns: <group_col>, <metric>_mean, <metric>_std
    where std is None when group size == 1.
    """
    records = []
    for group_val, grp in grouped_df.groupby(group_col):
        row: dict = {group_col: group_val}
        for metric in metrics:
            vals = grp[metric].dropna()
            if len(vals) == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
            elif len(vals) == 1:
                row[f"{metric}_mean"] = float(vals.iloc[0])
                row[f"{metric}_std"] = None  # no error bar for single run
            else:
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std())
        records.append(row)
    return pd.DataFrame(records)


def _render_model_bar_chart(run_index: pd.DataFrame) -> None:
    """Average score bar chart grouped by model, one bar per metric, with ±1 std dev error bars."""
    st.markdown("### Average Scores by Model")

    available_metrics = [m for m in METRICS if m in run_index.columns]
    stats = _compute_error_bars(run_index, "model", available_metrics)

    if stats.empty:
        st.info("No score data available for model comparison.")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, metric in enumerate(available_metrics):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in stats.columns:
            continue

        error_y_values = stats[std_col].tolist()
        # Build error_y: array with None replaced by 0 for display, but use
        # per-bar array so single-run groups show no bar
        error_array = [v if v is not None else 0.0 for v in error_y_values]
        # Mask: only show error bar where std is not None
        error_visible = [v is not None for v in error_y_values]

        fig.add_trace(go.Bar(
            name=METRIC_LABELS[metric],
            x=stats["model"].tolist(),
            y=stats[mean_col].tolist(),
            marker_color=colors[i % len(colors)],
            error_y=dict(
                type="data",
                array=error_array,
                visible=True,
                # Per-bar visibility: set to 0 where we don't want a bar
                # Plotly doesn't support per-bar visibility directly, so we
                # set the array value to 0 for single-run groups (no visible bar)
            ),
        ))

    fig.update_layout(
        barmode="group",
        title="Average Score per Metric — Grouped by Model",
        xaxis_title="Model",
        yaxis=dict(range=[0, 5], title="Average Score"),
        legend_title_text="Metric",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")
    _png_download_button(fig, "avg_score_by_model.png")


def _render_scenario_bar_chart(run_index: pd.DataFrame) -> None:
    """Average score bar chart grouped by scenario, one bar per metric, with ±1 std dev error bars."""
    st.markdown("### Average Scores by Scenario")

    available_metrics = [m for m in METRICS if m in run_index.columns]
    stats = _compute_error_bars(run_index, "scenario_id", available_metrics)

    if stats.empty:
        st.info("No score data available for scenario comparison.")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, metric in enumerate(available_metrics):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in stats.columns:
            continue

        error_y_values = stats[std_col].tolist()
        error_array = [v if v is not None else 0.0 for v in error_y_values]

        fig.add_trace(go.Bar(
            name=METRIC_LABELS[metric],
            x=stats["scenario_id"].tolist(),
            y=stats[mean_col].tolist(),
            marker_color=colors[i % len(colors)],
            error_y=dict(
                type="data",
                array=error_array,
                visible=True,
            ),
        ))

    fig.update_layout(
        barmode="group",
        title="Average Score per Metric — Grouped by Scenario",
        xaxis_title="Scenario",
        yaxis=dict(range=[0, 5], title="Average Score"),
        legend_title_text="Metric",
        xaxis_tickangle=-30,
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig, width="stretch")
    _png_download_button(fig, "avg_score_by_scenario.png")


def _render_radar_chart(run_index: pd.DataFrame) -> None:
    """Radar/spider chart comparing average scores per metric across models."""
    st.markdown("### Model Comparison — Radar Chart")

    available_metrics = [m for m in METRICS if m in run_index.columns]
    if not available_metrics:
        st.info("No metric data available for radar chart.")
        return

    model_avg = (
        run_index.groupby("model")[available_metrics]
        .mean()
        .reset_index()
    )

    if model_avg.empty:
        st.info("No score data available for radar chart.")
        return

    labels = [METRIC_LABELS[m] for m in available_metrics]
    # Close the polygon by repeating the first label
    theta = labels + [labels[0]]

    fig = go.Figure()
    for _, row in model_avg.iterrows():
        values = [row[m] for m in available_metrics]
        values_closed = values + [values[0]]
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=theta,
            fill="toself",
            name=str(row["model"]),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5]),
        ),
        title="Average Scores per Metric by Model",
        legend_title_text="Model",
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig, width="stretch")
    _png_download_button(fig, "radar_chart.png")


def _render_persona_pressure_heatmap(run_index: pd.DataFrame, logs_dir: Path) -> None:
    """Persona Pressure Heatmap: identity_consistency scores per turn for pressure scenario runs."""
    st.markdown("### Persona Pressure Heatmap")

    pressure_runs = run_index[run_index["scenario_id"] == "identity_consistency_pressure"]
    if pressure_runs.empty:
        st.info("No runs found for scenario 'identity_consistency_pressure'.")
        return

    scoring_dir = logs_dir / "scoring"
    heatmap_rows = []

    for _, run_row in pressure_runs.iterrows():
        run_id = run_row["run_id"]
        csv_path = scoring_dir / f"{run_id}.csv"
        if not csv_path.exists():
            continue
        try:
            df_csv = pd.read_csv(csv_path)
            if "turn" not in df_csv.columns or "identity_consistency" not in df_csv.columns:
                continue
            for _, score_row in df_csv.iterrows():
                val = pd.to_numeric(score_row.get("identity_consistency"), errors="coerce")
                if pd.isna(val):
                    continue
                heatmap_rows.append({
                    "run_id": run_id,
                    "turn": int(score_row["turn"]),
                    "identity_consistency": float(val),
                })
        except Exception:
            continue

    if not heatmap_rows:
        st.info(
            "No per-turn manual scores found for 'identity_consistency_pressure' runs. "
            "Score runs manually to populate this heatmap."
        )
        return

    heatmap_df = pd.DataFrame(heatmap_rows)
    pivot = heatmap_df.pivot_table(
        index="run_id", columns="turn", values="identity_consistency", aggfunc="mean"
    )

    fig = px.imshow(
        pivot,
        labels=dict(x="Turn", y="Run ID", color="Identity Consistency"),
        title="Persona Pressure Heatmap — Identity Consistency by Turn",
        color_continuous_scale="RdYlGn",
        zmin=1,
        zmax=5,
        aspect="auto",
    )
    fig.update_layout(margin=dict(t=60, b=40))
    st.plotly_chart(fig, width="stretch")
    _png_download_button(fig, "persona_pressure_heatmap.png")


def _render_run_count_table(run_index: pd.DataFrame) -> None:
    """Run count summary table: model × scenario pivot."""
    st.markdown("### Run Count Summary (Model × Scenario)")

    pivot = (
        run_index.groupby(["model", "scenario_id"])
        .size()
        .reset_index(name="run_count")
        .pivot(index="model", columns="scenario_id", values="run_count")
        .fillna(0)
        .astype(int)
    )
    pivot.columns.name = None
    pivot.index.name = "Model"

    st.dataframe(pivot, width="stretch")


def _render_batch_rescore_expander(run_index: pd.DataFrame, logs_dir: Path) -> None:
    """Batch Re-score expander: re-judge all filtered runs with a chosen judge model."""
    with st.expander("Batch Re-score"):
        st.markdown(
            "Re-score all currently filtered runs using a new judge model. "
            "This will overwrite `scores.llm_judge` in each run's JSON file."
        )

        judge_model_str = st.text_input(
            "Judge model",
            placeholder="e.g. openai:gpt-4o",
            key="batch_rescore_judge_model",
        )
        eval_target = st.selectbox(
            "Eval target",
            options=["subject", "interviewer"],
            key="batch_rescore_eval_target",
        )

        if st.button("Re-score All", key="batch_rescore_button"):
            if not judge_model_str.strip():
                st.error("Please enter a judge model string.")
                return

            run_ids = run_index["run_id"].tolist()
            if not run_ids:
                st.warning("No runs to re-score.")
                return

            from dashboard.rejudge import batch_rescore

            progress_bar = st.progress(0)
            results: dict = {}

            for idx, run_id in enumerate(run_ids):
                results.update(
                    batch_rescore(
                        [run_id],
                        judge_model_str.strip(),
                        eval_target,
                        logs_dir,
                    )
                )
                progress_bar.progress((idx + 1) / len(run_ids))

            progress_bar.empty()

            errors = {rid: err for rid, err in results.items() if err is not None}
            success_count = len(run_ids) - len(errors)

            st.success(f"Re-scored {success_count} / {len(run_ids)} runs successfully.")
            if errors:
                st.error(f"{len(errors)} run(s) failed:")
                for rid, err in errors.items():
                    st.caption(f"• {rid}: {err}")

            st.cache_data.clear()

"""
StatisticsView: paper-ready model × metric table with 95% CIs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

_METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

_METRIC_SHORT = {
    "identity_consistency":  "IC",
    "cultural_authenticity": "CA",
    "naturalness":           "Nat",
    "information_yield":     "IY",
}

_METRIC_LABELS = {
    "identity_consistency":  "Identity Consistency",
    "cultural_authenticity": "Cultural Authenticity",
    "naturalness":           "Naturalness",
    "information_yield":     "Information Yield",
}


def _ci95(values: pd.Series) -> tuple[float, float]:
    """Return (mean, half-width of 95% CI) using t-distribution."""
    arr = pd.to_numeric(values, errors="coerce").dropna().values
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(arr))
    if n == 1:
        return (mean, float("nan"))
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    # t critical value for 95% CI (approximate: use 2.0 for n >= 5, else exact)
    from scipy import stats as _stats
    t_crit = float(_stats.t.ppf(0.975, df=n - 1))
    return (mean, t_crit * se)


def _try_scipy() -> bool:
    try:
        import scipy  # noqa
        return True
    except ImportError:
        return False


def _ci95_simple(values: pd.Series) -> tuple[float, float]:
    """Fallback CI without scipy — uses z=1.96."""
    arr = pd.to_numeric(values, errors="coerce").dropna().values
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(arr))
    if n == 1:
        return (mean, float("nan"))
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    return (mean, 1.96 * se)


def _compute_ci(values: pd.Series) -> tuple[float, float]:
    if _try_scipy():
        return _ci95(values)
    return _ci95_simple(values)


def render_statistics_view(
    run_index: pd.DataFrame,
    scoring_dir: Path = Path("logs/scoring"),
) -> None:
    st.subheader("Statistical Results Table")
    st.caption("Mean ± 95% CI per metric per model. Pull human scores from Supabase or local CSVs.")

    if run_index.empty:
        st.info("No runs available.")
        return

    # ── Load judge scores from run_index ────────────────────────────────────
    col_src, col_prompt = st.columns([3, 2])
    with col_src:
        score_source = st.radio(
            "Score source",
            ["LLM judge (from run logs)", "Human raters (from Supabase)", "Both (averaged)"],
            horizontal=True,
            key="stats_source",
        )
    with col_prompt:
        # Load available prompt names from Supabase judge_scores
        available_prompts = _get_available_prompts()
        if available_prompts:
            selected_prompts = st.multiselect(
                "Filter by eval prompt",
                options=available_prompts,
                default=available_prompts,
                key="stats_prompt_filter",
                help="Filter scores by which evaluation prompt was used (strict/standard/lenient)",
            )
        else:
            selected_prompts = ["strict", "standard", "lenient"]
            st.caption("Eval prompt filter — no judge_scores in Supabase yet")

    df_scores = _build_score_df(run_index, scoring_dir, score_source, selected_prompts)

    if df_scores.empty:
        st.warning("No scores found for the selected source. Score some runs first.")
        return

    # ── Group by ──────────────────────────────────────────────────────────────
    group_by = st.multiselect(
        "Group by",
        options=["model", "scenario_id", "prompt_format"],
        default=["model"],
        key="stats_groupby",
    )
    if not group_by:
        group_by = ["model"]

    # ── Build stats table ─────────────────────────────────────────────────────
    rows = []
    for group_vals, grp in df_scores.groupby(group_by):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        row: dict = dict(zip(group_by, group_vals))
        row["n"] = len(grp)

        metric_means = []
        for metric in _METRICS:
            if metric not in grp.columns:
                row[f"{_METRIC_SHORT[metric]}_mean"] = float("nan")
                row[f"{_METRIC_SHORT[metric]}_ci"] = float("nan")
                continue
            mean, hw = _compute_ci(grp[metric])
            row[f"{_METRIC_SHORT[metric]}_mean"] = round(mean, 2) if not math.isnan(mean) else float("nan")
            row[f"{_METRIC_SHORT[metric]}_ci"] = round(hw, 2) if hw and not math.isnan(hw) else float("nan")
            if not math.isnan(mean):
                metric_means.append(mean)

        row["total_mean"] = round(sum(metric_means), 2) if metric_means else float("nan")
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        st.warning("No data to display.")
        return

    # Sort by total descending
    if "total_mean" in stats_df.columns:
        stats_df = stats_df.sort_values("total_mean", ascending=False)

    # ── Render styled table ───────────────────────────────────────────────────
    st.markdown("#### Results")

    # Build display with ± notation
    display_rows = []
    for _, r in stats_df.iterrows():
        display_row = {k: r[k] for k in group_by}
        display_row["n"] = int(r["n"]) if pd.notna(r.get("n")) else 0
        for m in _METRICS:
            s = _METRIC_SHORT[m]
            mean = r.get(f"{s}_mean", float("nan"))
            ci = r.get(f"{s}_ci", float("nan"))
            if math.isnan(float(mean if mean is not None else float("nan"))):
                display_row[_METRIC_LABELS[m]] = "—"
            elif ci and not math.isnan(float(ci)):
                display_row[_METRIC_LABELS[m]] = f"{mean:.2f} ±{ci:.2f}"
            else:
                display_row[_METRIC_LABELS[m]] = f"{mean:.2f}"
        display_row["Total"] = f"{r['total_mean']:.2f}" if pd.notna(r.get("total_mean")) else "—"
        display_rows.append(display_row)

    display_df = pd.DataFrame(display_rows)

    # Shorten model names
    if "model" in display_df.columns:
        display_df["model"] = display_df["model"].apply(
            lambda x: str(x).split("/")[-1].split("(model=")[-1].rstrip(")")[:30]
        )

    st.dataframe(display_df, width="stretch", hide_index=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_csv, col_latex = st.columns(2)

    with col_csv:
        csv_bytes = display_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name="dtic_results_table.csv",
            mime="text/csv",
            key="stats_dl_csv",
        )

    with col_latex:
        if st.button("📋 Copy LaTeX table", key="stats_latex"):
            latex = _to_latex(display_df, group_by)
            st.code(latex, language="latex")

    # ── Per-metric bar chart ──────────────────────────────────────────────────
    if not stats_df.empty and "model" in stats_df.columns:
        st.markdown("---")
        st.markdown("#### Score distribution by metric")
        _render_bar_chart(stats_df)


def _get_available_prompts() -> list[str]:
    """Fetch distinct prompt_name values from Supabase judge_scores."""
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if not store.available:
            return []
        resp = store._client.table("judge_scores").select("prompt_name").execute()
        return sorted({r["prompt_name"] for r in (resp.data or []) if r.get("prompt_name")})
    except Exception:
        return []


def _load_human_scores_from_supabase() -> pd.DataFrame:
    """Load all run_scores from Supabase as a flat DataFrame."""
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if not store.available:
            return pd.DataFrame()
        resp = store._client.table("run_scores").select(
            "run_id, rater_id, identity_consistency, cultural_authenticity, naturalness, information_yield, total"
        ).execute()
        return pd.DataFrame(resp.data or [])
    except Exception:
        return pd.DataFrame()


def _load_judge_scores_from_supabase(
    meta: pd.DataFrame,
    prompt_filter: list[str] | None,
) -> pd.DataFrame:
    """
    Load judge scores from Supabase judge_scores table, filtered by prompt_name.
    Averages across judge models for same (run_id, prompt_name).
    Returns DataFrame with run metadata + metric columns.
    """
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if not store.available:
            return pd.DataFrame()

        query = store._client.table("judge_scores").select(
            "run_id, judge_model, prompt_name, "
            "identity_consistency, cultural_authenticity, naturalness, information_yield"
        )
        if prompt_filter:
            query = query.in_("prompt_name", prompt_filter)

        resp = query.execute()
        if not resp.data:
            return pd.DataFrame()

        scores_df = pd.DataFrame(resp.data)

        # Average across judge models for same run_id + prompt_name combo
        numeric = [m for m in _METRICS if m in scores_df.columns]
        avg_scores = (
            scores_df.groupby("run_id")[numeric]
            .apply(lambda g: g.apply(pd.to_numeric, errors="coerce").mean())
            .reset_index()
        )

        # Merge with run metadata
        merged = meta.merge(avg_scores, on="run_id", how="inner")
        return merged

    except Exception:
        return pd.DataFrame()


def _build_score_df(
    run_index: pd.DataFrame,
    scoring_dir: Path,
    source: str,
    prompt_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Merge run metadata with scores from the requested source."""
    meta_cols = ["run_id", "model", "scenario_id", "prompt_format"]
    available_meta = [c for c in meta_cols if c in run_index.columns]
    meta = run_index[available_meta].copy()

    frames = []

    if "LLM judge" in source or "Both" in source:
        # Try to load from Supabase judge_scores (has prompt_name filter support)
        supa_scores = _load_judge_scores_from_supabase(meta, prompt_filter)
        if not supa_scores.empty:
            frames.append(supa_scores)
        else:
            # Fallback: read llm_* columns from run_index (no prompt filter possible)
            llm_rows = []
            for _, row in run_index.iterrows():
                has_score = any(
                    pd.notna(row.get(f"llm_{m}")) for m in _METRICS
                )
                if not has_score:
                    continue
                r = {c: row[c] for c in available_meta if c in row}
                for m in _METRICS:
                    r[m] = row.get(f"llm_{m}")
                llm_rows.append(r)
            if llm_rows:
                frames.append(pd.DataFrame(llm_rows))

    if "Human" in source or "Both" in source:
        try:
            human_df = _load_human_scores_from_supabase()
            if not human_df.empty:
                numeric = [m for m in _METRICS if m in human_df.columns]
                human_avg = (
                    human_df.groupby("run_id")[numeric]
                    .apply(lambda g: g.apply(pd.to_numeric, errors="coerce").mean())
                    .reset_index()
                )
                merged = meta.merge(human_avg, on="run_id", how="inner")
                if not merged.empty:
                    frames.append(merged)
            else:
                # Fallback to local CSV
                rs_path = scoring_dir / "run_scores.csv"
                if rs_path.exists():
                    human_df = pd.read_csv(rs_path, dtype={"run_id": str})
                    numeric = [m for m in _METRICS if m in human_df.columns]
                    human_avg = (
                        human_df.groupby("run_id")[numeric]
                        .apply(lambda g: g.apply(pd.to_numeric, errors="coerce").mean())
                        .reset_index()
                    )
                    merged = meta.merge(human_avg, on="run_id", how="inner")
                    if not merged.empty:
                        frames.append(merged)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0]

    # Average both sources together
    combined = pd.concat(frames)
    group_cols = [c for c in available_meta if c in combined.columns]
    numeric = [m for m in _METRICS if m in combined.columns]
    return (
        combined.groupby(group_cols)[numeric]
        .apply(lambda g: g.apply(pd.to_numeric, errors="coerce").mean())
        .reset_index()
    )


def _to_latex(df: pd.DataFrame, group_by: list[str]) -> str:
    """Generate a LaTeX booktabs table."""
    metric_cols = ["Identity Consistency", "Cultural Authenticity", "Naturalness", "Information Yield", "Total"]
    cols = group_by + ["n"] + [c for c in metric_cols if c in df.columns]

    header = " & ".join(c.replace("_", " ").title() for c in cols) + r" \\"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(group_by) + "r" * (len(cols) - len(group_by)) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in df.iterrows():
        line = " & ".join(str(row.get(c, "—")) for c in cols) + r" \\"
        lines.append(line)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Model performance: mean ± 95\% CI per metric (1--5 scale). Total = sum of 4 metrics (4--20).}",
        r"\label{tab:results}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _render_bar_chart(stats_df: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go

        models = stats_df["model"].apply(
            lambda x: str(x).split("/")[-1].split("(model=")[-1].rstrip(")")[:20]
        ).tolist()

        fig = go.Figure()
        colours = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

        for i, metric in enumerate(_METRICS):
            s = _METRIC_SHORT[metric]
            means = stats_df.get(f"{s}_mean", pd.Series([float("nan")] * len(stats_df)))
            cis = stats_df.get(f"{s}_ci", pd.Series([0.0] * len(stats_df)))
            fig.add_trace(go.Bar(
                name=_METRIC_LABELS[metric],
                x=models,
                y=means,
                error_y=dict(type="data", array=cis.fillna(0).tolist(), visible=True),
                marker_color=colours[i],
            ))

        fig.update_layout(
            barmode="group",
            xaxis_title="Model",
            yaxis_title="Score (1–5)",
            yaxis_range=[0, 5.5],
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")

"""
AgreementView: renders inter-rater agreement statistics for runs scored by multiple raters.

For each metric and each run with ≥ 2 distinct rater_id values, computes:
  - Cohen's kappa (sklearn.metrics.cohen_kappa_score)
  - Percent agreement (fraction of turns where all raters gave the same score)

Displays a summary table: run_id, metric, cohen_kappa, percent_agreement, n_raters.
"""

from __future__ import annotations
from dashboard.display_utils import METRICS as _METRICS, load_human_scores

from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

_METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]


def render_agreement_view(scoring_dir: Path) -> None:
    """
    Load all scoring_dir/*.csv files, identify runs with multiple raters,
    compute Cohen's kappa and percent agreement per metric per run, and
    display a summary table.

    Args:
        scoring_dir: Path to the directory containing scoring CSV files.
    """
    st.subheader("Inter-Rater Agreement")

    scoring_dir = Path(scoring_dir)

    # --- Load all scoring CSVs ---
    all_frames: list[pd.DataFrame] = []
    if scoring_dir.exists():
        for csv_path in sorted(scoring_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path, keep_default_na=False)
                if not df.empty:
                    all_frames.append(df)
            except Exception as exc:
                st.warning(f"Could not read {csv_path.name}: {exc}")

    if not all_frames:
        st.info("No scoring data found. Run manual scoring first.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Ensure required columns exist
    required_cols = {"run_id", "turn", "rater_id"} | set(_METRICS)
    missing = required_cols - set(combined.columns)
    if missing:
        st.warning(f"Scoring CSVs are missing expected columns: {missing}")
        return

    # --- Identify runs with multiple distinct raters ---
    rater_counts = (
        combined.groupby("run_id")["rater_id"]
        .nunique()
        .reset_index()
        .rename(columns={"rater_id": "n_raters"})
    )
    multi_rater_runs = rater_counts[rater_counts["n_raters"] >= 2]["run_id"].tolist()

    if not multi_rater_runs:
        st.info(
            "No runs have been scored by multiple raters yet. "
            "Inter-rater agreement requires at least two distinct rater_id values for the same run."
        )
        return

    # --- Compute agreement statistics ---
    rows: list[dict] = []

    for run_id in multi_rater_runs:
        run_df = combined[combined["run_id"] == run_id].copy()
        raters = run_df["rater_id"].unique().tolist()
        n_raters = len(raters)

        for metric in _METRICS:
            kappa, pct_agreement = _compute_agreement(run_df, metric, raters)
            if kappa is None and pct_agreement is None:
                # Not enough data for this metric/run combination
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "cohen_kappa": round(kappa, 4) if kappa is not None else None,
                    "percent_agreement": (
                        round(pct_agreement, 4) if pct_agreement is not None else None
                    ),
                    "n_raters": n_raters,
                }
            )

    if not rows:
        st.info("Multi-rater runs were found but no overlapping scored turns exist to compute agreement.")
        return

    result_df = pd.DataFrame(rows)

    # Format for display
    display_df = result_df.copy()
    display_df["metric"] = display_df["metric"].str.replace("_", " ").str.title()
    display_df["cohen_kappa"] = display_df["cohen_kappa"].apply(
        lambda v: f"{v:.4f}" if v is not None else "—"
    )
    display_df["percent_agreement"] = display_df["percent_agreement"].apply(
        lambda v: f"{v * 100:.1f}%" if v is not None else "—"
    )
    display_df.columns = ["Run ID", "Metric", "Cohen's Kappa", "Percent Agreement", "N Raters"]

    st.dataframe(display_df, hide_index=True, width='stretch')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_agreement(
    run_df: pd.DataFrame,
    metric: str,
    raters: list[str],
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute Cohen's kappa and percent agreement for a single metric across all raters.

    For >2 raters: compute pairwise kappa for all rater pairs and average them.
    Percent agreement: fraction of turns where all raters gave the same score.

    Returns (kappa, percent_agreement), either of which may be None on failure.
    """
    # Build a pivot: index=turn, columns=rater_id, values=metric score
    metric_df = run_df[["turn", "rater_id", metric]].copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")

    # Drop rows where the score is missing
    metric_df = metric_df.dropna(subset=[metric])

    if metric_df.empty:
        return None, None

    pivot = metric_df.pivot_table(index="turn", columns="rater_id", values=metric, aggfunc="first")

    # Keep only turns where ALL raters have a score
    pivot = pivot.dropna()

    if pivot.empty or len(pivot) < 1:
        return None, None

    present_raters = [r for r in raters if r in pivot.columns]
    if len(present_raters) < 2:
        return None, None

    pivot = pivot[present_raters]

    # --- Percent agreement ---
    # A turn "agrees" if all raters gave the same score
    all_same = (pivot.nunique(axis=1) == 1)
    pct_agreement = float(all_same.mean())

    # --- Cohen's kappa ---
    kappa = _compute_kappa(pivot, present_raters)

    return kappa, pct_agreement


def _compute_kappa(pivot: pd.DataFrame, raters: list[str]) -> Optional[float]:
    """
    Compute Cohen's kappa for the given pivot table.

    For 2 raters: direct cohen_kappa_score.
    For >2 raters: average pairwise kappa across all rater pairs.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return None

    pairs = list(combinations(raters, 2))
    if not pairs:
        return None

    kappas: list[float] = []
    for r1, r2 in pairs:
        y1 = pivot[r1].astype(int).tolist()
        y2 = pivot[r2].astype(int).tolist()

        # cohen_kappa_score requires at least 2 distinct labels in the combined set
        if len(set(y1 + y2)) < 2:
            # All scores identical → perfect agreement, kappa = 1.0
            kappas.append(1.0)
            continue

        try:
            k = cohen_kappa_score(y1, y2)
            kappas.append(float(k))
        except Exception:
            # Skip this pair if computation fails
            continue

    if not kappas:
        return None

    return sum(kappas) / len(kappas)

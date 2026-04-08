"""
Model comparison report.

Reads the aggregated summary produced by aggregate_scores.py
(logs/analysis/summary.csv) and produces a ranked comparison table
showing how each model performs per metric, per scenario.

Output: logs/analysis/model_comparison.csv  (printed to stdout as well)

Usage:
    python analysis/compare_models.py
"""

from pathlib import Path

import pandas as pd


SUMMARY_FILE = Path("logs/analysis/summary.csv")
OUTPUT_FILE = Path("logs/analysis/model_comparison.csv")

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

AVG_COLS = [f"{m}_avg" for m in METRICS]


def load_summary(path: Path = SUMMARY_FILE) -> pd.DataFrame:
    """Load the aggregated summary CSV produced by aggregate_scores.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {path}\n"
            "Run `python analysis/aggregate_scores.py` first."
        )
    df = pd.read_csv(path)
    for col in AVG_COLS:
        if col not in df.columns:
            df[col] = None
    return df


def build_ranked_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build a model × metric × average-score table, grouped by scenario.

    Returns a DataFrame with columns:
        scenario_id, model, <metric>_avg ..., overall_avg, rank
    ranked within each scenario by overall_avg descending.
    """
    df = summary.copy()

    # Compute overall average across all metrics (row-wise, skip NaN)
    df["overall_avg"] = df[AVG_COLS].mean(axis=1, skipna=True)

    # Rank models within each scenario (rank 1 = best)
    df["rank"] = (
        df.groupby("scenario_id")["overall_avg"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    col_order = ["scenario_id", "rank", "model"] + AVG_COLS + ["overall_avg"]
    if "run_count" in df.columns:
        col_order.append("run_count")

    return df[col_order].sort_values(["scenario_id", "rank"]).reset_index(drop=True)


def highlight_best_worst(ranked: pd.DataFrame) -> dict:
    """
    For each metric, identify the best and worst performing model
    (averaged across all scenarios).

    Returns a dict keyed by metric with {"best": model, "worst": model,
    "best_score": float, "worst_score": float}.
    """
    # Collapse to model-level averages across all scenarios
    model_avg = ranked.groupby("model")[AVG_COLS].mean(numeric_only=True)

    highlights = {}
    for col in AVG_COLS:
        metric = col.replace("_avg", "")
        series = model_avg[col].dropna()
        if series.empty:
            highlights[metric] = {"best": None, "worst": None,
                                  "best_score": None, "worst_score": None}
            continue
        highlights[metric] = {
            "best": series.idxmax(),
            "best_score": round(series.max(), 3),
            "worst": series.idxmin(),
            "worst_score": round(series.min(), 3),
        }
    return highlights


def print_report(ranked: pd.DataFrame, highlights: dict) -> None:
    """Print the ranked table and best/worst highlights to stdout."""
    print("\n=== Model Comparison Report ===\n")

    for scenario_id, group in ranked.groupby("scenario_id"):
        print(f"Scenario: {scenario_id}")
        display = group.drop(columns=["scenario_id"]).to_string(index=False)
        print(display)
        print()

    print("=== Best / Worst Model per Metric (across all scenarios) ===\n")
    rows = []
    for metric, info in highlights.items():
        rows.append({
            "metric": metric,
            "best_model": info["best"],
            "best_score": info["best_score"],
            "worst_model": info["worst"],
            "worst_score": info["worst_score"],
        })
    hw_df = pd.DataFrame(rows)
    print(hw_df.to_string(index=False))
    print()


def compare(summary_path: Path = SUMMARY_FILE) -> pd.DataFrame:
    """Full pipeline: load → rank → highlight → return ranked table."""
    summary = load_summary(summary_path)

    if summary.empty:
        print("[warn] Summary is empty — nothing to compare.")
        return summary

    ranked = build_ranked_table(summary)
    highlights = highlight_best_worst(ranked)
    print_report(ranked, highlights)
    return ranked


def main():
    ranked = compare()

    if ranked.empty:
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUTPUT_FILE, index=False)
    print(f"Comparison table written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

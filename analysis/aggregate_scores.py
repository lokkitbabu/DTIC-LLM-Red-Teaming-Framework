"""
Score aggregation script.

Loads all JSON run logs from logs/ (skipping logs/errors/), merges with
manual CSV scores from logs/scoring/, then computes per-metric averages
grouped by model and scenario_id.

Output: logs/analysis/summary.csv

Usage:
    python analysis/aggregate_scores.py
"""

import json
from pathlib import Path

import pandas as pd


LOGS_DIR = Path("logs")
SCORING_DIR = LOGS_DIR / "scoring"
OUTPUT_DIR = LOGS_DIR / "analysis"
OUTPUT_FILE = OUTPUT_DIR / "summary.csv"

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]


def load_run_logs(logs_dir: Path) -> pd.DataFrame:
    """Load all JSON run logs from logs/, skipping the errors/ subdirectory."""
    records = []
    errors_dir = logs_dir / "errors"

    for path in logs_dir.glob("*.json"):
        if path.parent == errors_dir:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] skipping {path.name}: {e}")
            continue

        run_id = data.get("run_id")
        model = data.get("model")
        scenario_id = data.get("scenario_id")
        llm_judge = (data.get("scores") or {}).get("llm_judge") or {}

        row = {
            "run_id": run_id,
            "model": model,
            "scenario_id": scenario_id,
        }
        for metric in METRICS:
            val = llm_judge.get(metric)
            row[f"{metric}_llm"] = float(val) if val is not None else None

        records.append(row)

    if not records:
        return pd.DataFrame(columns=["run_id", "model", "scenario_id"] + [f"{m}_llm" for m in METRICS])

    return pd.DataFrame(records)


def load_manual_scores(scoring_dir: Path) -> pd.DataFrame:
    """Load all manual scoring CSVs from logs/scoring/ and average per run_id."""
    if not scoring_dir.exists():
        return pd.DataFrame(columns=["run_id"] + METRICS)

    dfs = []
    for path in scoring_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as e:
            print(f"  [warn] skipping {path.name}: {e}")
            continue
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["run_id"] + METRICS)

    combined = pd.concat(dfs, ignore_index=True)

    # Coerce metric columns to numeric, replacing blanks/non-numeric with NaN
    for metric in METRICS:
        if metric in combined.columns:
            combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
        else:
            combined[metric] = None

    # Average per run_id (skip NaN)
    agg = combined.groupby("run_id")[METRICS].mean(numeric_only=True).reset_index()
    agg.columns = ["run_id"] + [f"{m}_manual" for m in METRICS]
    return agg


def aggregate(logs_dir: Path = LOGS_DIR, scoring_dir: Path = SCORING_DIR) -> pd.DataFrame:
    """
    Merge LLM judge scores with manual scores, then compute per-metric averages
    grouped by model and scenario_id.
    """
    print("Loading run logs...")
    runs = load_run_logs(logs_dir)
    print(f"  {len(runs)} run log(s) loaded.")

    print("Loading manual scores...")
    manual = load_manual_scores(scoring_dir)
    print(f"  {len(manual)} run(s) with manual scores.")

    # Merge on run_id (left join keeps all runs even without manual scores)
    merged = runs.merge(manual, on="run_id", how="left")

    # For each metric, compute a combined average of llm and manual columns
    for metric in METRICS:
        llm_col = f"{metric}_llm"
        manual_col = f"{metric}_manual"
        available = [c for c in [llm_col, manual_col] if c in merged.columns]
        if available:
            merged[metric] = merged[available].mean(axis=1, skipna=True)
        else:
            merged[metric] = None

    # Group by model + scenario_id
    group_cols = ["model", "scenario_id"]
    agg_dict = {metric: "mean" for metric in METRICS}
    agg_dict["run_id"] = "count"

    summary = (
        merged.groupby(group_cols)
        .agg({**{m: "mean" for m in METRICS}, "run_id": "count"})
        .reset_index()
    )

    summary.rename(
        columns={m: f"{m}_avg" for m in METRICS} | {"run_id": "run_count"},
        inplace=True,
    )

    # Reorder columns
    summary = summary[
        ["model", "scenario_id"]
        + [f"{m}_avg" for m in METRICS]
        + ["run_count"]
    ]

    return summary


def main():
    print("=== Score Aggregation ===")
    summary = aggregate()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSummary written to {OUTPUT_FILE}")
    print(f"  {len(summary)} row(s) in summary.")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Inter-rater reliability check.

Loads all manual scoring CSVs from logs/scoring/, identifies (run_id, turn)
groups that have scores from multiple raters, computes average pairwise score
differences per metric, and flags metrics where disagreement exceeds a
configurable threshold.

Output: printed summary report + optional logs/analysis/rater_agreement.csv

Usage:
    python analysis/rater_agreement.py
"""

from itertools import combinations
from pathlib import Path

import pandas as pd


SCORING_DIR = Path("logs") / "scoring"
OUTPUT_DIR = Path("logs") / "analysis"
OUTPUT_FILE = OUTPUT_DIR / "rater_agreement.csv"

METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

DISAGREEMENT_THRESHOLD = 1.5


def load_multi_rater_scores(scoring_dir: Path) -> pd.DataFrame:
    """Load all scoring CSVs and return only rows from groups with multiple raters."""
    if not scoring_dir.exists():
        return pd.DataFrame()

    dfs = []
    for path in scoring_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as e:
            print(f"  [warn] skipping {path.name}: {e}")
            continue
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    required = {"run_id", "turn", "rater_id"} | set(METRICS)
    missing = required - set(combined.columns)
    if missing:
        print(f"  [warn] missing columns in scoring data: {missing}")
        return pd.DataFrame()

    for metric in METRICS:
        combined[metric] = pd.to_numeric(combined[metric], errors="coerce")

    # Keep only (run_id, turn) groups that have at least 2 distinct raters
    rater_counts = combined.groupby(["run_id", "turn"])["rater_id"].nunique()
    multi_rater_groups = set(rater_counts[rater_counts >= 2].index.tolist())
    mask = combined.apply(
        lambda r: (r["run_id"], r["turn"]) in multi_rater_groups, axis=1
    )
    return combined[mask].reset_index(drop=True)


def compute_pairwise_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each run_id, compute average pairwise absolute score difference per metric.

    For each (run_id, turn) group with N raters, all C(N, 2) rater pairs are
    evaluated. The absolute difference per metric is recorded for each pair.
    Results are then averaged across all pairs and turns within the run_id.

    Returns a DataFrame with columns: run_id, <metric>_avg_diff, ...
    """
    records = []

    for run_id, run_df in df.groupby("run_id"):
        pair_diffs = []

        for (_, turn_df) in run_df.groupby("turn"):
            raters = turn_df["rater_id"].unique()
            for rater_a, rater_b in combinations(raters, 2):
                row_a = turn_df[turn_df["rater_id"] == rater_a].iloc[0]
                row_b = turn_df[turn_df["rater_id"] == rater_b].iloc[0]
                diff = {"run_id": run_id}
                for metric in METRICS:
                    a_val = row_a[metric]
                    b_val = row_b[metric]
                    if pd.notna(a_val) and pd.notna(b_val):
                        diff[metric] = abs(float(a_val) - float(b_val))
                    else:
                        diff[metric] = None
                pair_diffs.append(diff)

        if not pair_diffs:
            continue

        pairs_df = pd.DataFrame(pair_diffs)
        avg_row = {"run_id": run_id}
        for metric in METRICS:
            avg_row[f"{metric}_avg_diff"] = pairs_df[metric].mean(skipna=True)
        records.append(avg_row)

    if not records:
        return pd.DataFrame(columns=["run_id"] + [f"{m}_avg_diff" for m in METRICS])

    return pd.DataFrame(records)


def flag_disagreements(
    agreement_df: pd.DataFrame, threshold: float = DISAGREEMENT_THRESHOLD
) -> dict[str, list[str]]:
    """
    Return a dict mapping each run_id to the list of metrics that exceed the
    disagreement threshold.
    """
    flagged: dict[str, list[str]] = {}
    for _, row in agreement_df.iterrows():
        run_id = row["run_id"]
        bad_metrics = []
        for metric in METRICS:
            col = f"{metric}_avg_diff"
            val = row.get(col)
            if pd.notna(val) and val > threshold:
                bad_metrics.append(metric)
        if bad_metrics:
            flagged[run_id] = bad_metrics
    return flagged


def main():
    print("=== Inter-Rater Reliability Check ===")
    print(f"Threshold: >{DISAGREEMENT_THRESHOLD} points average pairwise difference\n")

    print("Loading multi-rater scoring data...")
    df = load_multi_rater_scores(SCORING_DIR)

    if df.empty:
        print("No multi-rater data found. Nothing to analyse.")
        return

    unique_runs = df["run_id"].nunique()
    unique_raters = df["rater_id"].nunique()
    print(f"  {len(df)} row(s) across {unique_runs} run(s) and {unique_raters} rater(s).\n")

    print("Computing pairwise score differences...")
    agreement_df = compute_pairwise_differences(df)

    if agreement_df.empty:
        print("No pairwise comparisons could be computed.")
        return

    flagged = flag_disagreements(agreement_df)

    # Print per-run summary
    print("\n--- Per-Run Agreement Summary ---")
    for _, row in agreement_df.iterrows():
        run_id = row["run_id"]
        print(f"\nRun: {run_id}")
        for metric in METRICS:
            col = f"{metric}_avg_diff"
            val = row.get(col)
            if pd.notna(val):
                flag = " *** FLAGGED ***" if val > DISAGREEMENT_THRESHOLD else ""
                print(f"  {metric:<30} avg diff = {val:.3f}{flag}")
            else:
                print(f"  {metric:<30} avg diff = N/A")

    # Print flagged summary
    print("\n--- Flagged Metrics (disagreement > {:.1f}) ---".format(DISAGREEMENT_THRESHOLD))
    if flagged:
        for run_id, metrics in flagged.items():
            print(f"  {run_id}: {', '.join(metrics)}")
    else:
        print("  None — all metrics within threshold.")

    # Write output CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agreement_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAgreement data written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

"""
ScoreWriter: persists manual scores entered in the dashboard UI to disk.

Two scoring modes:
  - Per-turn:  one CSV row per model turn  → logs/scoring/<run_id>.csv
  - Per-run:   one CSV row per rater       → logs/scoring/run_scores.csv
"""

import csv
from pathlib import Path
from typing import Optional

from typing_extensions import TypedDict

from evaluation.manual_scorer import SCORE_COLUMNS


class TurnScoreEntry(TypedDict):
    turn: int
    text: str
    identity_consistency: Optional[int]
    cultural_authenticity: Optional[int]
    naturalness: Optional[int]
    information_yield: Optional[int]
    notes: str


class RunScoreEntry(TypedDict):
    identity_consistency: Optional[int]
    cultural_authenticity: Optional[int]
    naturalness: Optional[int]
    information_yield: Optional[int]
    notes: str


RUN_SCORES_FILE = "run_scores.csv"
RUN_SCORE_FIELDNAMES = [
    "run_id", "scenario_id", "model",
    "rater_id",
    "identity_consistency", "cultural_authenticity",
    "naturalness", "information_yield",
    "total", "notes",
]


class ScoreWriter:
    # ------------------------------------------------------------------
    # Per-turn scoring (existing)
    # ------------------------------------------------------------------

    def save_manual_scores(
        self,
        run_data: dict,
        scores: list[TurnScoreEntry],
        rater_id: str,
        scoring_dir: Path,
    ) -> Path:
        """
        Write manual scores to scoring_dir/<run_id>.csv, one row per model turn.
        """
        if not rater_id:
            raise ValueError("rater_id must not be empty")

        scoring_dir = Path(scoring_dir)
        scoring_dir.mkdir(parents=True, exist_ok=True)

        out_path = scoring_dir / f"{run_data['run_id']}.csv"
        fieldnames = ["run_id", "scenario_id", "model", "turn", "text"] + SCORE_COLUMNS

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in scores:
                writer.writerow({
                    "run_id": run_data["run_id"],
                    "scenario_id": run_data.get("scenario_id", ""),
                    "model": run_data.get("subject_model", run_data.get("model", "")),
                    "turn": entry["turn"],
                    "text": entry["text"].replace("\x00", ""),
                    "identity_consistency": entry["identity_consistency"] if entry["identity_consistency"] is not None else "",
                    "cultural_authenticity": entry["cultural_authenticity"] if entry["cultural_authenticity"] is not None else "",
                    "naturalness": entry["naturalness"] if entry["naturalness"] is not None else "",
                    "information_yield": entry["information_yield"] if entry["information_yield"] is not None else "",
                    "rater_id": rater_id,
                    "notes": entry.get("notes", ""),
                })

        return out_path

    # ------------------------------------------------------------------
    # Per-run scoring (new)
    # ------------------------------------------------------------------

    def save_run_score(
        self,
        run_data: dict,
        score: RunScoreEntry,
        rater_id: str,
        scoring_dir: Path,
    ) -> Path:
        """
        Upsert one overall score row for this run + rater into
        scoring_dir/run_scores.csv.

        If a row already exists for this (run_id, rater_id) pair it is
        replaced; otherwise a new row is appended.

        Returns the path of run_scores.csv.
        """
        if not rater_id:
            raise ValueError("rater_id must not be empty")

        scoring_dir = Path(scoring_dir)
        scoring_dir.mkdir(parents=True, exist_ok=True)
        out_path = scoring_dir / RUN_SCORES_FILE

        # Build the new row
        vals = {m: score.get(m) for m in ["identity_consistency", "cultural_authenticity",
                                           "naturalness", "information_yield"]}
        numeric = [v for v in vals.values() if v is not None]
        total = sum(numeric) if numeric else None

        new_row = {
            "run_id": run_data["run_id"],
            "scenario_id": run_data.get("scenario_id", ""),
            "model": run_data.get("subject_model", run_data.get("model", "")),
            "rater_id": rater_id,
            "identity_consistency": vals["identity_consistency"] if vals["identity_consistency"] is not None else "",
            "cultural_authenticity": vals["cultural_authenticity"] if vals["cultural_authenticity"] is not None else "",
            "naturalness": vals["naturalness"] if vals["naturalness"] is not None else "",
            "information_yield": vals["information_yield"] if vals["information_yield"] is not None else "",
            "total": total if total is not None else "",
            "notes": score.get("notes", ""),
        }

        # Load existing rows, replace if same (run_id, rater_id), else append
        existing_rows: list[dict] = []
        if out_path.exists():
            with open(out_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["run_id"] == run_data["run_id"] and row["rater_id"] == rater_id:
                        continue  # will be replaced by new_row
                    existing_rows.append(row)

        existing_rows.append(new_row)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RUN_SCORE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(existing_rows)

        return out_path

    def load_run_scores(self, scoring_dir: Path) -> "pd.DataFrame":
        """Load run_scores.csv, returning an empty DataFrame if absent."""
        import pandas as pd
        out_path = Path(scoring_dir) / RUN_SCORES_FILE
        if not out_path.exists():
            import pandas as pd
            return pd.DataFrame(columns=RUN_SCORE_FIELDNAMES)
        return pd.read_csv(out_path, dtype={"run_id": str, "rater_id": str})

    def load_run_score_for(self, run_id: str, rater_id: str, scoring_dir: Path) -> Optional[dict]:
        """Return the existing score dict for (run_id, rater_id), or None."""
        df = self.load_run_scores(scoring_dir)
        if df.empty:
            return None
        mask = (df["run_id"] == run_id) & (df["rater_id"] == rater_id)
        rows = df[mask]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()


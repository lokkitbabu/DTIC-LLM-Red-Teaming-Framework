"""
ScoreWriter: persists manual scores entered in the dashboard UI to disk.
Writes one CSV row per model turn to logs/scoring/<run_id>.csv.
"""

import csv
from pathlib import Path
from typing import Optional

from typing_extensions import TypedDict

from evaluation.manual_scorer import SCORE_COLUMNS


class TurnScoreEntry(TypedDict):
    turn: int
    text: str
    identity_consistency: Optional[int]   # 1–5 or None
    cultural_authenticity: Optional[int]
    naturalness: Optional[int]
    information_yield: Optional[int]
    notes: str


class ScoreWriter:
    def save_manual_scores(
        self,
        run_data: dict,
        scores: list[TurnScoreEntry],
        rater_id: str,
        scoring_dir: Path,
    ) -> Path:
        """
        Write manual scores to scoring_dir/<run_id>.csv, one row per model turn.

        Args:
            run_data:    Full run dict containing run_id, scenario_id, model, conversation.
            scores:      One TurnScoreEntry per model turn.
            rater_id:    Non-empty identifier for the human rater.
            scoring_dir: Directory to write the CSV into.

        Returns:
            Path of the written CSV file.

        Raises:
            ValueError: If rater_id is empty.
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

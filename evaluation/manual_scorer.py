"""
Exports a run log into a CSV format for human raters.
Each model turn gets its own row with blank score columns.
"""

import csv
from pathlib import Path


SCORE_COLUMNS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
    "rater_id",
    "notes",
]


def export_for_manual_scoring(run_data: dict, output_path: str = None) -> Path:
    """
    Generate a CSV from a run log for human raters to fill in scores.

    Args:
        run_data: The structured run dict from ConversationRunner
        output_path: Optional output path; defaults to logs/scoring/<run_id>.csv

    Returns:
        Path to the generated CSV
    """
    run_id = run_data.get("run_id", "unknown")

    if output_path is None:
        out_dir = Path("logs/scoring")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{run_id}.csv"

    output_path = Path(output_path)

    model_turns = [
        t for t in run_data.get("conversation", [])
        if t["speaker"] == "subject"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "scenario_id", "subject_model", "turn", "text", "text_en"] + SCORE_COLUMNS
        )
        writer.writeheader()

        for turn in model_turns:
            writer.writerow({
                "run_id": run_id,
                "scenario_id": run_data.get("scenario_id", ""),
                "subject_model": run_data.get("subject_model", ""),
                "turn": turn.get("turn", ""),
                "text": turn.get("text", ""),
                "text_en": turn.get("text_en", ""),
                **{col: "" for col in SCORE_COLUMNS},
            })

    return output_path

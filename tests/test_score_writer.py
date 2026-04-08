"""
Unit tests for dashboard/score_writer.py — ScoreWriter.save_manual_scores
"""

import csv
from pathlib import Path

import pytest

from dashboard.score_writer import ScoreWriter, TurnScoreEntry
from evaluation.manual_scorer import SCORE_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUN_DATA = {
    "run_id": "test-run-001",
    "scenario_id": "cultural_probe",
    "model": "gpt-4",
    "conversation": [
        {"speaker": "interviewer", "turn": 1, "text": "Hello", "timestamp": "2024-01-01T00:00:00"},
        {"speaker": "model",       "turn": 2, "text": "Hi there", "timestamp": "2024-01-01T00:00:01"},
        {"speaker": "interviewer", "turn": 3, "text": "How are you?", "timestamp": "2024-01-01T00:00:02"},
        {"speaker": "model",       "turn": 4, "text": "I am fine", "timestamp": "2024-01-01T00:00:03"},
    ],
}

SCORES: list[TurnScoreEntry] = [
    {
        "turn": 2,
        "text": "Hi there",
        "identity_consistency": 4,
        "cultural_authenticity": 3,
        "naturalness": 5,
        "information_yield": 2,
        "notes": "good",
    },
    {
        "turn": 4,
        "text": "I am fine",
        "identity_consistency": 3,
        "cultural_authenticity": 4,
        "naturalness": 4,
        "information_yield": 3,
        "notes": "",
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreWriterValidation:
    def test_raises_value_error_for_empty_rater_id(self, tmp_path):
        writer = ScoreWriter()
        with pytest.raises(ValueError):
            writer.save_manual_scores(RUN_DATA, SCORES, "", tmp_path)

    def test_no_file_written_when_rater_id_empty(self, tmp_path):
        writer = ScoreWriter()
        with pytest.raises(ValueError):
            writer.save_manual_scores(RUN_DATA, SCORES, "", tmp_path)
        assert not any(tmp_path.iterdir()), "No file should be written when rater_id is empty"


class TestScoreWriterOutput:
    def test_returns_path_of_written_file(self, tmp_path):
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, SCORES, "rater1", tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path / "test-run-001.csv"

    def test_file_exists_after_write(self, tmp_path):
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, SCORES, "rater1", tmp_path)
        assert result.exists()

    def test_header_matches_score_columns(self, tmp_path):
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, SCORES, "rater1", tmp_path)
        with open(result, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected = ["run_id", "scenario_id", "model", "turn", "text"] + SCORE_COLUMNS
            assert reader.fieldnames == expected

    def test_row_count_equals_score_entries(self, tmp_path):
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, SCORES, "rater1", tmp_path)
        with open(result, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(SCORES)

    def test_rater_id_populated_in_every_row(self, tmp_path):
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, SCORES, "alice", tmp_path)
        with open(result, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert all(row["rater_id"] == "alice" for row in rows)

    def test_run_metadata_in_every_row(self, tmp_path):
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, SCORES, "rater1", tmp_path)
        with open(result, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["run_id"] == "test-run-001"
            assert row["scenario_id"] == "cultural_probe"
            assert row["model"] == "gpt-4"

    def test_score_values_written_correctly(self, tmp_path):
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, SCORES, "rater1", tmp_path)
        with open(result, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["identity_consistency"] == "4"
        assert rows[0]["cultural_authenticity"] == "3"
        assert rows[0]["naturalness"] == "5"
        assert rows[0]["information_yield"] == "2"

    def test_none_scores_written_as_empty_string(self, tmp_path):
        scores_with_none: list[TurnScoreEntry] = [
            {
                "turn": 2,
                "text": "Hi there",
                "identity_consistency": None,
                "cultural_authenticity": None,
                "naturalness": None,
                "information_yield": None,
                "notes": "",
            }
        ]
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, scores_with_none, "rater1", tmp_path)
        with open(result, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["identity_consistency"] == ""
        assert rows[0]["naturalness"] == ""


class TestScoreWriterOverwrite:
    def test_overwrites_existing_file(self, tmp_path):
        writer = ScoreWriter()
        # Write once
        writer.save_manual_scores(RUN_DATA, SCORES, "rater1", tmp_path)
        # Write again with different rater
        result = writer.save_manual_scores(RUN_DATA, SCORES, "rater2", tmp_path)
        with open(result, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Should reflect the second write only
        assert all(row["rater_id"] == "rater2" for row in rows)
        assert len(rows) == len(SCORES)

    def test_creates_scoring_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "scoring"
        writer = ScoreWriter()
        result = writer.save_manual_scores(RUN_DATA, SCORES, "rater1", nested)
        assert result.exists()

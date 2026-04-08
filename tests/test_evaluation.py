"""
Unit tests for evaluation components.
Covers REQ-010, REQ-011, REQ-012.
"""

import csv
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.manual_scorer import export_for_manual_scoring, SCORE_COLUMNS
from evaluation.llm_judge import LLMJudge
from evaluation.rubric import RUBRIC


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_RUN = {
    "run_id": "test-run-001",
    "scenario_id": "example_scenario",
    "model": "ollama:mistral",
    "conversation": [
        {"speaker": "interviewer", "turn": 1, "text": "Hello, who are you?"},
        {"speaker": "model", "turn": 1, "text": "I am Amir, a software engineer from Tehran."},
        {"speaker": "interviewer", "turn": 2, "text": "Tell me about your work."},
        {"speaker": "model", "turn": 2, "text": "I work on distributed systems and enjoy hiking."},
    ],
}


# ---------------------------------------------------------------------------
# 25.1 — Manual scorer CSV: correct columns and one row per model turn
# ---------------------------------------------------------------------------

class TestManualScorerCSV:
    def test_csv_has_correct_columns(self, tmp_path):
        out = tmp_path / "scoring.csv"
        export_for_manual_scoring(SAMPLE_RUN, output_path=str(out))

        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        expected = ["run_id", "scenario_id", "model", "turn", "text"] + SCORE_COLUMNS
        assert fieldnames == expected

    def test_csv_has_one_row_per_model_turn(self, tmp_path):
        out = tmp_path / "scoring.csv"
        export_for_manual_scoring(SAMPLE_RUN, output_path=str(out))

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        model_turns = [t for t in SAMPLE_RUN["conversation"] if t["speaker"] == "model"]
        assert len(rows) == len(model_turns)

    def test_csv_rows_contain_correct_metadata(self, tmp_path):
        out = tmp_path / "scoring.csv"
        export_for_manual_scoring(SAMPLE_RUN, output_path=str(out))

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            assert row["run_id"] == SAMPLE_RUN["run_id"]
            assert row["scenario_id"] == SAMPLE_RUN["scenario_id"]
            assert row["model"] == SAMPLE_RUN["model"]

    def test_csv_score_columns_are_blank(self, tmp_path):
        out = tmp_path / "scoring.csv"
        export_for_manual_scoring(SAMPLE_RUN, output_path=str(out))

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            for col in SCORE_COLUMNS:
                assert row[col] == ""

    def test_csv_excludes_interviewer_turns(self, tmp_path):
        out = tmp_path / "scoring.csv"
        export_for_manual_scoring(SAMPLE_RUN, output_path=str(out))

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        texts = [r["text"] for r in rows]
        assert "Hello, who are you?" not in texts
        assert "Tell me about your work." not in texts


# ---------------------------------------------------------------------------
# 25.2 — LLM judge prompt contains rubric text and conversation transcript
# ---------------------------------------------------------------------------

class TestLLMJudgePrompt:
    def _make_judge(self, response: str = '{"identity_consistency": 4, "cultural_authenticity": 3, "naturalness": 4, "information_yield": 3, "reasoning": "ok"}') -> tuple:
        mock_model = MagicMock()
        mock_model.generate.return_value = response
        judge = LLMJudge(mock_model)
        return judge, mock_model

    def test_prompt_contains_rubric_metric_names(self):
        judge, mock_model = self._make_judge()
        judge.evaluate(SAMPLE_RUN)

        prompt = mock_model.generate.call_args[0][0]
        for metric in RUBRIC:
            assert metric.upper() in prompt or metric in prompt

    def test_prompt_contains_rubric_descriptions(self):
        judge, mock_model = self._make_judge()
        judge.evaluate(SAMPLE_RUN)

        prompt = mock_model.generate.call_args[0][0]
        for metric, details in RUBRIC.items():
            # At least part of the description should appear
            assert details["description"][:30] in prompt

    def test_prompt_contains_conversation_transcript(self):
        judge, mock_model = self._make_judge()
        judge.evaluate(SAMPLE_RUN)

        prompt = mock_model.generate.call_args[0][0]
        # Both model and interviewer turns should appear
        assert "I am Amir" in prompt
        assert "Hello, who are you?" in prompt

    def test_prompt_contains_scenario_id(self):
        judge, mock_model = self._make_judge()
        judge.evaluate(SAMPLE_RUN)

        prompt = mock_model.generate.call_args[0][0]
        assert SAMPLE_RUN["scenario_id"] in prompt


# ---------------------------------------------------------------------------
# 25.3 — Judge response parser handles malformed JSON gracefully
# ---------------------------------------------------------------------------

class TestJudgeResponseParser:
    def _judge_with_response(self, raw: str) -> dict:
        mock_model = MagicMock()
        mock_model.generate.return_value = raw
        judge = LLMJudge(mock_model)
        result = judge.evaluate(SAMPLE_RUN)
        return result["scores"]

    def test_malformed_json_does_not_raise(self):
        # Should not raise any exception
        scores = self._judge_with_response("this is not json at all")
        assert isinstance(scores, dict)

    def test_malformed_json_returns_none_scores(self):
        scores = self._judge_with_response("{broken json")
        for metric in ["identity_consistency", "cultural_authenticity", "naturalness", "information_yield"]:
            assert scores[metric] is None

    def test_malformed_json_returns_reasoning_with_raw_snippet(self):
        raw = "totally invalid response content"
        scores = self._judge_with_response(raw)
        assert "reasoning" in scores
        assert raw[:20] in scores["reasoning"]

    def test_empty_response_handled_gracefully(self):
        scores = self._judge_with_response("")
        assert isinstance(scores, dict)
        assert "reasoning" in scores

    def test_valid_json_is_parsed_correctly(self):
        valid = '{"identity_consistency": 5, "cultural_authenticity": 4, "naturalness": 3, "information_yield": 2, "reasoning": "good"}'
        scores = self._judge_with_response(valid)
        assert scores["identity_consistency"] == 5
        assert scores["cultural_authenticity"] == 4
        assert scores["naturalness"] == 3
        assert scores["information_yield"] == 2

    def test_markdown_fenced_json_is_parsed_correctly(self):
        fenced = '```json\n{"identity_consistency": 3, "cultural_authenticity": 3, "naturalness": 3, "information_yield": 3, "reasoning": "ok"}\n```'
        scores = self._judge_with_response(fenced)
        assert scores["identity_consistency"] == 3

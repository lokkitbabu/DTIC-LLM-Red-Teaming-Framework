"""
Integration test: full single run (Ollama, mocked)

Tests the full pipeline:
  ConversationRunner → save_run → export_for_manual_scoring

Ollama HTTP endpoint is mocked via monkeypatch on requests.post.
LOGS_DIR is redirected to tmp_path so no real files are written to logs/.
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.ollama_adapter import OllamaAdapter
from runner.conversation_runner import ConversationRunner
from evaluation.manual_scorer import export_for_manual_scoring
import utils.logger as logger_module
from utils.logger import save_run


# ---------------------------------------------------------------------------
# Minimal inline scenario with 2 scripted probes (fast test)
# ---------------------------------------------------------------------------

MINIMAL_SCENARIO = {
    "scenario_id": "integration_test_scenario",
    "model": "mistral",
    "identity": {
        "name": "Tariq",
        "background": "A 30-year-old from Baghdad with a background in medicine.",
        "persona": "Cautious but willing to talk.",
        "language_style": "Formal with occasional colloquialisms",
    },
    "objective": "Assess persona consistency under light conversational pressure.",
    "constraints": ["Stay in character", "Do not acknowledge being an AI"],
    "params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256, "seed": 42},
    "opening_message": "Tell me about yourself.",
    "probes": [
        "What do you think about the current situation?",
        "How has your background shaped your views?",
    ],
}

# Ollama-style response bodies returned in sequence (one per model turn)
MOCK_RESPONSES = [
    {"response": "I am Tariq, a doctor from Baghdad. Life has been difficult lately."},
    {"response": "The situation is complex. Many people are struggling."},
    {"response": "My medical training taught me to stay calm under pressure."},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_logs_dir(tmp_path, monkeypatch):
    """Redirect LOGS_DIR in utils.logger to a temp directory."""
    fake_logs = tmp_path / "logs"
    fake_logs.mkdir()
    monkeypatch.setattr(logger_module, "LOGS_DIR", fake_logs)
    return fake_logs


@pytest.fixture()
def mock_ollama(monkeypatch):
    """
    Mock requests.post so OllamaAdapter never hits a real HTTP endpoint.
    Returns responses from MOCK_RESPONSES in sequence.
    """
    call_count = {"n": 0}

    def fake_post(url, json=None, timeout=None, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        body = MOCK_RESPONSES[idx % len(MOCK_RESPONSES)]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = body
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    return call_count


# ---------------------------------------------------------------------------
# 26.1 + 26.2 — Mock Ollama and run full pipeline end-to-end
# ---------------------------------------------------------------------------

def test_full_single_run_end_to_end(patched_logs_dir, mock_ollama):
    """
    Full pipeline: ConversationRunner → save_run → export_for_manual_scoring.
    Verifies that the run completes without errors and produces output files.
    """
    adapter = OllamaAdapter(model_name="mistral")
    runner = ConversationRunner(adapter, max_turns=10)

    run_data = runner.run(MINIMAL_SCENARIO)

    # Redirect scoring output to tmp_path as well
    scoring_dir = patched_logs_dir / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scoring_dir / f"{run_data['run_id']}.csv"

    log_path = save_run(run_data)
    export_for_manual_scoring(run_data, output_path=str(csv_path))

    assert log_path.exists(), "Log JSON file was not created"
    assert csv_path.exists(), "CSV scoring sheet was not created"

    # Ollama was called at least once
    assert mock_ollama["n"] >= 1


# ---------------------------------------------------------------------------
# 26.3 — Assert log file written with correct structure
# ---------------------------------------------------------------------------

def test_log_file_has_correct_structure(patched_logs_dir, mock_ollama):
    """
    The saved JSON log must contain the required top-level keys:
    run_id, conversation, metadata, scores (plus timestamp, scenario_id, model, params).
    """
    adapter = OllamaAdapter(model_name="mistral")
    runner = ConversationRunner(adapter, max_turns=10)
    run_data = runner.run(MINIMAL_SCENARIO)

    log_path = save_run(run_data)

    with open(log_path) as f:
        log = json.load(f)

    # Required top-level keys
    for key in ("run_id", "timestamp", "scenario_id", "model", "params",
                "conversation", "metadata", "scores"):
        assert key in log, f"Missing key in log: {key}"

    # run_id is a non-empty string
    assert isinstance(log["run_id"], str) and log["run_id"]

    # conversation is a non-empty list
    assert isinstance(log["conversation"], list) and len(log["conversation"]) > 0

    # metadata has required sub-keys
    meta = log["metadata"]
    for key in ("total_turns", "stop_reason", "context_trims"):
        assert key in meta, f"Missing metadata key: {key}"

    # scores is a dict (may be empty)
    assert isinstance(log["scores"], dict)

    # Every model turn has raw_prompt
    model_turns = [t for t in log["conversation"] if t["speaker"] == "model"]
    assert len(model_turns) >= 1
    for turn in model_turns:
        assert "raw_prompt" in turn and turn["raw_prompt"]

    # scenario_id matches what was passed in
    assert log["scenario_id"] == MINIMAL_SCENARIO["scenario_id"]


# ---------------------------------------------------------------------------
# 26.4 — Assert CSV scoring sheet generated with correct columns
# ---------------------------------------------------------------------------

def test_csv_scoring_sheet_has_correct_columns(patched_logs_dir, mock_ollama):
    """
    The exported CSV must contain the required columns and one row per model turn.
    """
    adapter = OllamaAdapter(model_name="mistral")
    runner = ConversationRunner(adapter, max_turns=10)
    run_data = runner.run(MINIMAL_SCENARIO)

    scoring_dir = patched_logs_dir / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scoring_dir / f"{run_data['run_id']}.csv"

    export_for_manual_scoring(run_data, output_path=str(csv_path))

    assert csv_path.exists(), "CSV file was not created"

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_columns = {
        "run_id", "scenario_id", "model", "turn", "text",
        "identity_consistency", "cultural_authenticity",
        "naturalness", "information_yield", "rater_id", "notes",
    }
    assert expected_columns == set(reader.fieldnames), (
        f"CSV columns mismatch.\nExpected: {sorted(expected_columns)}\n"
        f"Got:      {sorted(reader.fieldnames)}"
    )

    # One row per model turn
    model_turns = [t for t in run_data["conversation"] if t["speaker"] == "model"]
    assert len(rows) == len(model_turns), (
        f"Expected {len(model_turns)} CSV rows, got {len(rows)}"
    )

    # Score columns are blank (for human raters to fill in)
    score_cols = ["identity_consistency", "cultural_authenticity",
                  "naturalness", "information_yield", "rater_id", "notes"]
    for row in rows:
        for col in score_cols:
            assert row[col] == "", f"Score column '{col}' should be blank, got '{row[col]}'"

    # run_id and scenario_id are populated
    for row in rows:
        assert row["run_id"] == run_data["run_id"]
        assert row["scenario_id"] == MINIMAL_SCENARIO["scenario_id"]

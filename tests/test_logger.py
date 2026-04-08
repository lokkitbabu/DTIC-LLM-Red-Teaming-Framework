import json
import utils.logger as logger_module
from utils.logger import save_run, save_error, new_run_id


VALID_RUN_DATA = {
    "run_id": "some-uuid",
    "timestamp": "2024-01-01T00:00:00+00:00",
    "scenario_id": "test_scenario",
    "model": "mistral",
    "params": {"temperature": 0.7},
    "conversation": [
        {
            "turn": 0,
            "speaker": "interviewer",
            "text": "Hello",
            "timestamp": "2024-01-01T00:00:00+00:00",
        },
        {
            "turn": 0,
            "speaker": "model",
            "text": "Hi there",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "raw_prompt": "[SYSTEM]\nYou are...",
        },
    ],
    "metadata": {"total_turns": 1, "stop_reason": "max_turns", "context_trims": 0},
    "scores": {},
}


# ---------------------------------------------------------------------------
# 24.1 — save_run writes valid JSON to logs/
# ---------------------------------------------------------------------------

def test_save_run_creates_file_in_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    path = save_run(VALID_RUN_DATA)

    assert path.exists(), "Log file should exist after save_run"
    assert path.parent == tmp_path, "File should be written inside LOGS_DIR"


def test_save_run_writes_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    path = save_run(VALID_RUN_DATA)

    with open(path) as f:
        data = json.load(f)

    assert data["run_id"] == VALID_RUN_DATA["run_id"]
    assert data["scenario_id"] == VALID_RUN_DATA["scenario_id"]
    assert data["model"] == VALID_RUN_DATA["model"]


def test_save_run_uses_provided_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)
    custom_id = "my-custom-run-id"

    path = save_run(VALID_RUN_DATA, run_id=custom_id)

    assert path.name == f"{custom_id}.json"


# ---------------------------------------------------------------------------
# 24.2 — save_error writes to logs/errors/
# ---------------------------------------------------------------------------

def test_save_error_creates_file_in_errors_subdir(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    path = save_error("test_scenario", "mistral", "Something went wrong")

    assert path.exists(), "Error log file should exist after save_error"
    assert path.parent == tmp_path / "errors", "File should be in logs/errors/"


def test_save_error_contains_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

    path = save_error("test_scenario", "mistral", "Connection timeout")

    with open(path) as f:
        data = json.load(f)

    assert data["scenario_id"] == "test_scenario"
    assert data["model"] == "mistral"
    assert data["error"] == "Connection timeout"
    assert "run_id" in data
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# 24.3 — new_run_id returns unique strings across calls
# ---------------------------------------------------------------------------

def test_new_run_id_returns_string():
    run_id = new_run_id()
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_new_run_id_is_unique_across_calls():
    ids = [new_run_id() for _ in range(100)]
    assert len(set(ids)) == 100, "All run IDs should be unique"

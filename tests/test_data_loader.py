"""Unit tests for DataLoader.load_single_run (task 2.2)."""

import json
import sys
import types
import pytest
from pathlib import Path

# Stub streamlit before importing dashboard modules (not installed in test env)
_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data = lambda fn=None, **kw: (fn if fn else lambda f: f)
_st_stub.session_state = {}
sys.modules.setdefault("streamlit", _st_stub)

from dashboard.data_loader import DataLoader


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def logs_dir(tmp_path):
    return tmp_path


def test_load_single_run_returns_dict(loader, logs_dir):
    """Returns the full parsed JSON dict when the file exists and is valid."""
    data = {"run_id": "abc", "model": "gpt-4", "scores": {}}
    (logs_dir / "abc.json").write_text(json.dumps(data))

    result = loader.load_single_run("abc", logs_dir)

    assert result == data


def test_load_single_run_missing_file_raises(loader, logs_dir):
    """Raises FileNotFoundError when the run JSON does not exist."""
    with pytest.raises(FileNotFoundError):
        loader.load_single_run("nonexistent", logs_dir)


def test_load_single_run_invalid_json_raises(loader, logs_dir):
    """Raises ValueError when the file exists but contains invalid JSON."""
    (logs_dir / "bad.json").write_text("{ not valid json }")

    with pytest.raises(ValueError):
        loader.load_single_run("bad", logs_dir)


def test_load_single_run_uses_run_id_as_filename(loader, logs_dir):
    """Constructs the path as logs_dir/{run_id}.json."""
    data = {"run_id": "run-xyz-001"}
    (logs_dir / "run-xyz-001.json").write_text(json.dumps(data))

    result = loader.load_single_run("run-xyz-001", logs_dir)

    assert result["run_id"] == "run-xyz-001"


def test_load_single_run_accepts_path_object(loader, tmp_path):
    """Works when logs_dir is passed as a Path object."""
    data = {"key": "value"}
    (tmp_path / "myrun.json").write_text(json.dumps(data))

    result = loader.load_single_run("myrun", tmp_path)

    assert result == data


# --- Tests for load_manual_scores_for_run (task 2.3) ---

from dashboard.data_loader import SCORE_COLUMNS_CSV


def test_load_manual_scores_missing_csv_returns_empty_df(loader, tmp_path):
    """Returns empty DataFrame with correct columns when CSV does not exist."""
    result = loader.load_manual_scores_for_run("no-such-run", tmp_path)

    assert list(result.columns) == SCORE_COLUMNS_CSV
    assert len(result) == 0


def test_load_manual_scores_existing_csv_returns_data(loader, tmp_path):
    """Returns the CSV contents as a DataFrame when the file exists."""
    import pandas as pd
    scoring_dir = tmp_path / "scoring"
    scoring_dir.mkdir()
    df = pd.DataFrame([{
        "run_id": "run-001", "scenario_id": "s1", "model": "gpt-4",
        "turn": 1, "text": "hello",
        "identity_consistency": 4, "cultural_authenticity": 3,
        "naturalness": 5, "information_yield": 2,
        "rater_id": "rater1", "notes": "",
    }])
    df.to_csv(scoring_dir / "run-001.csv", index=False)

    result = loader.load_manual_scores_for_run("run-001", scoring_dir)

    assert len(result) == 1
    assert result.iloc[0]["run_id"] == "run-001"
    assert result.iloc[0]["identity_consistency"] == 4


def test_load_manual_scores_never_raises_on_missing(loader, tmp_path):
    """Does not raise FileNotFoundError for a missing CSV."""
    # Should not raise
    result = loader.load_manual_scores_for_run("ghost-run", tmp_path / "nonexistent_dir")
    assert list(result.columns) == SCORE_COLUMNS_CSV


# --- Tests for build_run_index (task 2.7) ---

import pandas as pd


def _valid_run_json(run_id: str, timestamp: str = "2024-01-01T00:00:00+00:00") -> dict:
    """Return a minimal valid RunLogSchema-compatible dict."""
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "scenario_id": "test_scenario",
        "model": "test-model",
        "params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "seed": 42},
        "conversation": [],
        "metadata": {"total_turns": 0, "stop_reason": "max_turns", "context_trims": 0},
        "scores": {
            "llm_judge": {
                "identity_consistency": 4,
                "cultural_authenticity": 3,
                "naturalness": 5,
                "information_yield": 4,
            }
        },
    }


@pytest.fixture
def logs_dir_with_runs(tmp_path):
    """A logs_dir with two valid run JSON files."""
    (tmp_path / "run-a.json").write_text(json.dumps(_valid_run_json("run-a", "2024-01-01T00:00:00+00:00")))
    (tmp_path / "run-b.json").write_text(json.dumps(_valid_run_json("run-b", "2024-01-02T00:00:00+00:00")))
    return tmp_path


def test_build_run_index_valid_runs_row_count(loader, logs_dir_with_runs):
    """Valid run files produce one row each in the result DataFrame."""
    df = loader.build_run_index(logs_dir_with_runs)
    assert len(df) == 2


def test_build_run_index_malformed_json_skipped(loader, tmp_path):
    """Malformed JSON files are skipped and do not appear in the result."""
    (tmp_path / "good.json").write_text(json.dumps(_valid_run_json("good")))
    (tmp_path / "bad.json").write_text("{ not valid json }")

    df = loader.build_run_index(tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "good"


def test_build_run_index_missing_scores_are_nan(loader, tmp_path):
    """Runs with no llm_judge scores and no CSV produce NaN in score columns."""
    run = _valid_run_json("no-scores")
    run["scores"] = {}  # no llm_judge key
    (tmp_path / "no-scores.json").write_text(json.dumps(run))

    df = loader.build_run_index(tmp_path)

    assert len(df) == 1
    row = df.iloc[0]
    for col in ["identity_consistency", "cultural_authenticity", "naturalness", "information_yield"]:
        assert pd.isna(row[col]), f"Expected NaN for {col}, got {row[col]}"


def test_build_run_index_errors_dir_excluded(loader, tmp_path):
    """Files inside the errors/ subdirectory are not included in the result."""
    errors_dir = tmp_path / "errors"
    errors_dir.mkdir()
    (tmp_path / "valid.json").write_text(json.dumps(_valid_run_json("valid")))
    # This file is inside errors/ — it's not a valid RunLogSchema anyway, but
    # the exclusion should happen before parsing.
    (errors_dir / "err-run.json").write_text(json.dumps(_valid_run_json("err-run")))

    df = loader.build_run_index(tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "valid"


def test_build_run_index_sorted_by_timestamp_ascending(loader, tmp_path):
    """Result is sorted by timestamp ascending regardless of file order."""
    (tmp_path / "run-z.json").write_text(json.dumps(_valid_run_json("run-z", "2024-03-01T00:00:00+00:00")))
    (tmp_path / "run-a.json").write_text(json.dumps(_valid_run_json("run-a", "2024-01-01T00:00:00+00:00")))
    (tmp_path / "run-m.json").write_text(json.dumps(_valid_run_json("run-m", "2024-02-01T00:00:00+00:00")))

    df = loader.build_run_index(tmp_path)

    assert list(df["run_id"]) == ["run-a", "run-m", "run-z"]


def test_build_run_index_empty_dir_returns_empty_df(loader, tmp_path):
    """An empty logs directory returns an empty DataFrame with the expected columns."""
    df = loader.build_run_index(tmp_path)

    assert len(df) == 0
    assert "run_id" in df.columns
    assert "timestamp" in df.columns


def test_build_run_index_with_manual_scores_csv(loader, tmp_path):
    """Manual scores from a CSV are loaded and reflected in the manual_* columns."""
    run = _valid_run_json("run-csv")
    run["scores"] = {}  # no llm_judge so combined == manual
    (tmp_path / "run-csv.json").write_text(json.dumps(run))

    scoring_dir = tmp_path / "scoring"
    scoring_dir.mkdir()
    manual_df = pd.DataFrame([{
        "run_id": "run-csv", "scenario_id": "s1", "model": "test-model",
        "turn": 1, "text": "hi",
        "identity_consistency": 5, "cultural_authenticity": 4,
        "naturalness": 3, "information_yield": 2,
        "rater_id": "rater1", "notes": "",
    }])
    manual_df.to_csv(scoring_dir / "run-csv.csv", index=False)

    df = loader.build_run_index(tmp_path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["manual_identity_consistency"] == 5.0
    assert row["identity_consistency"] == 5.0  # combined == manual when no llm


# --- Tests for apply_filters (task 3) ---

from datetime import date
from dashboard.data_loader import apply_filters, FilterState


def _make_run_index() -> pd.DataFrame:
    """Return a small DataFrame mimicking a RunIndex."""
    return pd.DataFrame([
        {"run_id": "r1", "model": "gpt-4",   "scenario_id": "s1", "timestamp": "2024-01-10T12:00:00+00:00"},
        {"run_id": "r2", "model": "gpt-4",   "scenario_id": "s2", "timestamp": "2024-02-15T08:00:00+00:00"},
        {"run_id": "r3", "model": "claude-3", "scenario_id": "s1", "timestamp": "2024-03-20T18:00:00+00:00"},
        {"run_id": "r4", "model": "claude-3", "scenario_id": "s3", "timestamp": "2024-04-05T00:00:00+00:00"},
    ])


def test_apply_filters_empty_state_returns_all_rows():
    """Empty FilterState returns all rows unchanged."""
    df = _make_run_index()
    fs: FilterState = {"models": [], "scenarios": [], "date_from": None, "date_to": None}
    result = apply_filters(df, fs)
    assert len(result) == 4


def test_apply_filters_model_filter():
    """Filtering by model returns only matching rows."""
    df = _make_run_index()
    fs: FilterState = {"models": ["gpt-4"], "scenarios": [], "date_from": None, "date_to": None}
    result = apply_filters(df, fs)
    assert list(result["run_id"]) == ["r1", "r2"]


def test_apply_filters_scenario_filter():
    """Filtering by scenario returns only matching rows."""
    df = _make_run_index()
    fs: FilterState = {"models": [], "scenarios": ["s1"], "date_from": None, "date_to": None}
    result = apply_filters(df, fs)
    assert list(result["run_id"]) == ["r1", "r3"]


def test_apply_filters_model_and_scenario_combined():
    """Both model and scenario filters are applied simultaneously (AND logic)."""
    df = _make_run_index()
    fs: FilterState = {"models": ["gpt-4"], "scenarios": ["s1"], "date_from": None, "date_to": None}
    result = apply_filters(df, fs)
    assert list(result["run_id"]) == ["r1"]


def test_apply_filters_date_from():
    """date_from excludes rows before the given date."""
    df = _make_run_index()
    fs: FilterState = {"models": [], "scenarios": [], "date_from": date(2024, 3, 1), "date_to": None}
    result = apply_filters(df, fs)
    assert set(result["run_id"]) == {"r3", "r4"}


def test_apply_filters_date_to():
    """date_to excludes rows after the given date (inclusive of that day)."""
    df = _make_run_index()
    fs: FilterState = {"models": [], "scenarios": [], "date_from": None, "date_to": date(2024, 2, 15)}
    result = apply_filters(df, fs)
    assert set(result["run_id"]) == {"r1", "r2"}


def test_apply_filters_date_range():
    """date_from and date_to together form an inclusive range."""
    df = _make_run_index()
    fs: FilterState = {
        "models": [], "scenarios": [],
        "date_from": date(2024, 2, 1),
        "date_to": date(2024, 3, 31),
    }
    result = apply_filters(df, fs)
    assert set(result["run_id"]) == {"r2", "r3"}


def test_apply_filters_does_not_mutate_input():
    """The original DataFrame is not mutated by apply_filters."""
    df = _make_run_index()
    original_len = len(df)
    original_ids = list(df["run_id"])
    fs: FilterState = {"models": ["gpt-4"], "scenarios": [], "date_from": None, "date_to": None}
    apply_filters(df, fs)
    assert len(df) == original_len
    assert list(df["run_id"]) == original_ids


def test_apply_filters_no_match_returns_empty():
    """A filter that matches nothing returns an empty DataFrame."""
    df = _make_run_index()
    fs: FilterState = {"models": ["nonexistent-model"], "scenarios": [], "date_from": None, "date_to": None}
    result = apply_filters(df, fs)
    assert len(result) == 0


def test_apply_filters_multiple_models():
    """Multiple values in models list are treated as OR within that dimension."""
    df = _make_run_index()
    fs: FilterState = {"models": ["gpt-4", "claude-3"], "scenarios": [], "date_from": None, "date_to": None}
    result = apply_filters(df, fs)
    assert len(result) == 4

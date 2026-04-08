"""
Property-based tests for ScoreWriter.save_manual_scores.

**Validates: Requirements 8.1**
"""

import csv
import tempfile
from pathlib import Path

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.score_writer import ScoreWriter
from evaluation.manual_scorer import SCORE_COLUMNS


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=30,
)

_valid_run_data = st.fixed_dictionaries({
    "run_id": _safe_text,
    "scenario_id": _safe_text,
    "model": _safe_text,
})

_score_value = st.one_of(
    st.none(),
    st.integers(min_value=1, max_value=5),
)

_turn_score_entry = st.fixed_dictionaries({
    "turn": st.integers(min_value=1, max_value=100),
    "text": st.text(min_size=0, max_size=200),
    "identity_consistency": _score_value,
    "cultural_authenticity": _score_value,
    "naturalness": _score_value,
    "information_yield": _score_value,
    "notes": st.text(min_size=0, max_size=100),
})

_scores_list = st.lists(_turn_score_entry, min_size=0, max_size=10)

_rater_id = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)


# ---------------------------------------------------------------------------
# Property 10: ScoreWriter column schema matches SCORE_COLUMNS
# ---------------------------------------------------------------------------

@given(
    run_data=_valid_run_data,
    scores=_scores_list,
    rater_id=_rater_id,
)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_score_writer_column_schema_matches_score_columns(run_data, scores, rater_id):
    """
    Property 10: ScoreWriter column schema matches SCORE_COLUMNS.

    For any valid run_data and scores list, the CSV written by
    ScoreWriter.save_manual_scores has a header row whose columns are exactly
    ["run_id", "scenario_id", "model", "turn", "text"] followed by
    SCORE_COLUMNS in the order defined in evaluation/manual_scorer.py.

    **Validates: Requirements 8.1**
    """
    expected_header = ["run_id", "scenario_id", "model", "turn", "text"] + SCORE_COLUMNS

    with tempfile.TemporaryDirectory() as tmp_dir:
        scoring_dir = Path(tmp_dir)
        writer = ScoreWriter()
        out_path = writer.save_manual_scores(run_data, scores, rater_id, scoring_dir)

        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == expected_header, (
            f"Expected header {expected_header}, got {header}"
        )


# ---------------------------------------------------------------------------
# Property 11: ScoreWriter row count equals model turn count
# ---------------------------------------------------------------------------

_turn = st.fixed_dictionaries({
    "speaker": st.sampled_from(["interviewer", "model"]),
    "text": st.text(min_size=0, max_size=100),
})

_conversation = st.lists(_turn, min_size=0, max_size=20)

_run_data_with_conversation = st.fixed_dictionaries({
    "run_id": _safe_text,
    "scenario_id": _safe_text,
    "model": _safe_text,
    "conversation": _conversation,
})


@given(run_data=_run_data_with_conversation, rater_id=_rater_id)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_score_writer_row_count_equals_model_turn_count(run_data, rater_id):
    """
    Property 11: ScoreWriter row count equals model turn count.

    For any run_data, the number of data rows written by
    ScoreWriter.save_manual_scores equals the number of turns in
    run_data["conversation"] whose speaker is "model".

    **Validates: Requirements 8.2**
    """
    model_turns = [t for t in run_data["conversation"] if t["speaker"] == "model"]
    scores = [
        {
            "turn": i + 1,
            "text": t["text"],
            "identity_consistency": None,
            "cultural_authenticity": None,
            "naturalness": None,
            "information_yield": None,
            "notes": "",
        }
        for i, t in enumerate(model_turns)
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        scoring_dir = Path(tmp_dir)
        writer = ScoreWriter()
        out_path = writer.save_manual_scores(run_data, scores, rater_id, scoring_dir)

        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            csv_rows = list(reader)

    assert len(csv_rows) == len(model_turns), (
        f"Expected {len(model_turns)} rows, got {len(csv_rows)}"
    )


# ---------------------------------------------------------------------------
# Property 12: ScoreWriter does not mutate run logs
# ---------------------------------------------------------------------------

_json_value = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=50),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=5),
    ),
    max_leaves=20,
)

_log_files = st.dictionaries(
    st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
        min_size=1,
        max_size=20,
    ),
    _json_value,
    min_size=1,
    max_size=5,
)


@given(
    run_data=_valid_run_data,
    scores=_scores_list,
    rater_id=_rater_id,
    log_contents=_log_files,
)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_score_writer_does_not_mutate_run_logs(run_data, scores, rater_id, log_contents):
    """
    Property 12: ScoreWriter does not mutate run logs.

    For any save operation performed by ScoreWriter, the content of every
    file under logs/*.json is identical before and after the operation.

    **Validates: Requirements 8.4, 10.3**
    """
    import json

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        # Write JSON files to the logs dir
        for name, content in log_contents.items():
            log_file = logs_dir / f"{name}.json"
            log_file.write_text(json.dumps(content), encoding="utf-8")

        # Capture content before
        before = {
            f.name: f.read_text(encoding="utf-8")
            for f in logs_dir.glob("*.json")
        }

        # scoring_dir is outside the logs dir
        scoring_dir = tmp_path / "scoring"
        writer = ScoreWriter()
        writer.save_manual_scores(run_data, scores, rater_id, scoring_dir)

        # Capture content after
        after = {
            f.name: f.read_text(encoding="utf-8")
            for f in logs_dir.glob("*.json")
        }

        assert before == after, (
            f"Log files were mutated by save_manual_scores. "
            f"Before: {set(before.keys())}, After: {set(after.keys())}"
        )

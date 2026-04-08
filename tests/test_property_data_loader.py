"""
Property-based tests for DataLoader.build_run_index.

**Validates: Requirements 1.1, 1.2, 1.3**
"""

import json
import sys
import types

# Stub streamlit before importing dashboard modules (not installed in test env)
_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data = lambda fn=None, **kw: (fn if fn else lambda f: f)
_st_stub.session_state = {}
sys.modules.setdefault("streamlit", _st_stub)

import tempfile
from pathlib import Path
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.data_loader import DataLoader


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for a valid RunLogSchema-conforming dict
_stop_reasons = st.sampled_from(["max_turns", "condition_met", "error", "probes_exhausted"])

_valid_run_log = st.fixed_dictionaries({
    "run_id": st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
        min_size=1,
        max_size=40,
    ),
    "timestamp": st.just("2024-01-01T00:00:00+00:00"),
    "scenario_id": st.text(min_size=1, max_size=20,
                           alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"),
                                                  whitelist_characters="-_")),
    "model": st.text(min_size=1, max_size=30,
                     alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"),
                                            whitelist_characters="-_:")),
    "subject_model": st.text(min_size=1, max_size=30,
                             alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"),
                                                    whitelist_characters="-_:")),
    "interviewer_model": st.text(min_size=1, max_size=30,
                                 alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"),
                                                        whitelist_characters="-_:")),
    "language": st.just("english"),
    "params": st.just({}),
    "conversation": st.just([]),
    "metadata": st.fixed_dictionaries({
        "total_turns": st.integers(min_value=0, max_value=100),
        "stop_reason": _stop_reasons,
        "context_trims": st.integers(min_value=0, max_value=10),
    }),
    "scores": st.just({}),
})

# Strategy for malformed JSON strings (not valid JSON at all)
_malformed_json = st.one_of(
    st.just(""),
    st.just("{not valid json}"),
    st.just("null"),
    st.just("42"),
    st.just('{"run_id": "x"}'),  # missing required fields → schema validation fails
    st.just("[ ]"),
)


# ---------------------------------------------------------------------------
# Property 1: Run index row count matches valid file count
# ---------------------------------------------------------------------------

@given(
    valid_logs=st.lists(_valid_run_log, min_size=0, max_size=8),
    malformed_count=st.integers(min_value=0, max_value=5),
    malformed_strings=st.lists(_malformed_json, min_size=0, max_size=5),
)
@settings(max_examples=30, suppress_health_check=(HealthCheck.too_slow,))
def test_build_run_index_row_count_matches_valid_file_count(
    valid_logs, malformed_count, malformed_strings
):
    """
    Property 1: Run index row count matches valid file count.

    For any directory of JSON files where some are valid RunLogSchema documents
    and some are malformed, build_run_index returns a DataFrame with exactly one
    row per valid file and zero rows for malformed files.

    **Validates: Requirements 1.1, 1.3**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Deduplicate run_ids so each file has a unique name
        seen_ids: set[str] = set()
        unique_valid: list[dict] = []
        for log in valid_logs:
            rid = log["run_id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                unique_valid.append(log)

        # Write valid run log files
        for log in unique_valid:
            path = tmp_path / f"{log['run_id']}.json"
            path.write_text(json.dumps(log), encoding="utf-8")

        # Write malformed files (use index-based names to avoid collisions)
        malformed_to_write = malformed_strings[:malformed_count]
        for i, bad_content in enumerate(malformed_to_write):
            path = tmp_path / f"malformed_{i}.json"
            path.write_text(bad_content, encoding="utf-8")

        loader = DataLoader()
        result = loader.build_run_index(tmp_path)

        assert len(result) == len(unique_valid), (
            f"Expected {len(unique_valid)} rows (one per valid file), "
            f"got {len(result)}. "
            f"Valid files: {[v['run_id'] for v in unique_valid]}, "
            f"malformed count: {len(malformed_to_write)}"
        )


# ---------------------------------------------------------------------------
# Property 2: errors/ subdirectory is always excluded
# ---------------------------------------------------------------------------

@given(
    error_logs=st.lists(_valid_run_log, min_size=1, max_size=5),
    top_level_logs=st.lists(_valid_run_log, min_size=0, max_size=5),
)
@settings(max_examples=30, suppress_health_check=(HealthCheck.too_slow,))
def test_build_run_index_errors_subdirectory_always_excluded(
    error_logs, top_level_logs
):
    """
    Property 2: errors/ subdirectory is always excluded.

    For any logs/ directory, no file located under logs/errors/ appears as a
    row in the RunIndex regardless of whether it is a valid run log.
    Valid files at the top level DO appear.

    **Validates: Requirements 1.2**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        errors_dir = tmp_path / "errors"
        errors_dir.mkdir()

        # Deduplicate run_ids across both sets to avoid filename collisions
        seen_ids: set[str] = set()
        unique_error_logs: list[dict] = []
        for log in error_logs:
            rid = log["run_id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                unique_error_logs.append(log)

        unique_top_level: list[dict] = []
        for log in top_level_logs:
            rid = log["run_id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                unique_top_level.append(log)

        # Write valid run logs inside errors/ — these must NOT appear in result
        error_run_ids = set()
        for log in unique_error_logs:
            path = errors_dir / f"{log['run_id']}.json"
            path.write_text(json.dumps(log), encoding="utf-8")
            error_run_ids.add(log["run_id"])

        # Write valid run logs at top level — these MUST appear in result
        for log in unique_top_level:
            path = tmp_path / f"{log['run_id']}.json"
            path.write_text(json.dumps(log), encoding="utf-8")

        loader = DataLoader()
        result = loader.build_run_index(tmp_path)

        result_run_ids = set(result["run_id"].tolist()) if not result.empty else set()

        # No error/ run_id should appear in the result
        assert error_run_ids.isdisjoint(result_run_ids), (
            f"Files from errors/ subdirectory appeared in RunIndex: "
            f"{error_run_ids & result_run_ids}"
        )

        # All top-level valid run_ids should appear
        expected_top_level_ids = {log["run_id"] for log in unique_top_level}
        assert expected_top_level_ids == result_run_ids, (
            f"Expected top-level run_ids {expected_top_level_ids} in result, "
            f"got {result_run_ids}"
        )


# ---------------------------------------------------------------------------
# Property 3: Score merge arithmetic (llm-only, manual-only, both, neither)
# ---------------------------------------------------------------------------

METRICS = ["identity_consistency", "cultural_authenticity", "naturalness", "information_yield"]

_score_value = st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)

_base_run_log = {
    "timestamp": "2024-01-01T00:00:00+00:00",
    "scenario_id": "test_scenario",
    "model": "test_model",
    "subject_model": "test_model",
    "interviewer_model": "test_model",
    "language": "english",
    "params": {},
    "conversation": [],
    "metadata": {
        "total_turns": 0,
        "stop_reason": "max_turns",
        "context_trims": 0,
    },
}


def _make_run_log(run_id: str, llm_scores) -> dict:
    """Build a valid RunLogSchema dict with optional llm_judge scores."""
    log = dict(_base_run_log)
    log["run_id"] = run_id
    if llm_scores is not None:
        log["scores"] = {"llm_judge": llm_scores}
    else:
        log["scores"] = {}
    return log


def _make_scoring_csv(run_id: str, manual_scores: dict) -> str:
    """Build CSV content for a manual scoring file."""
    header = "run_id,scenario_id,model,turn,text,identity_consistency,cultural_authenticity,naturalness,information_yield,rater_id,notes"
    row = (
        f"{run_id},test_scenario,test_model,1,hello,"
        f"{manual_scores['identity_consistency']},"
        f"{manual_scores['cultural_authenticity']},"
        f"{manual_scores['naturalness']},"
        f"{manual_scores['information_yield']},"
        f"rater1,"
    )
    return header + "\n" + row + "\n"


@given(
    run_id=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
        min_size=1,
        max_size=30,
    ),
    llm_scores=st.fixed_dictionaries({m: _score_value for m in METRICS}),
    manual_scores=st.fixed_dictionaries({m: _score_value for m in METRICS}),
)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,), deadline=None)
def test_score_merge_arithmetic_both_present(run_id, llm_scores, manual_scores):
    """
    Property 3a: When both LLM judge scores and manual scores are present,
    the combined metric score equals the arithmetic mean of the two values.

    **Validates: Requirements 1.5, 1.6**
    """
    import math

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        scoring_dir = tmp_path / "scoring"
        scoring_dir.mkdir()

        # Write run log with LLM scores
        log = _make_run_log(run_id, llm_scores)
        (tmp_path / f"{run_id}.json").write_text(json.dumps(log), encoding="utf-8")

        # Write manual scoring CSV
        csv_content = _make_scoring_csv(run_id, manual_scores)
        (scoring_dir / f"{run_id}.csv").write_text(csv_content, encoding="utf-8")

        loader = DataLoader()
        result = loader.build_run_index(tmp_path)

        assert len(result) == 1, f"Expected 1 row, got {len(result)}"
        row = result.iloc[0]

        for metric in METRICS:
            expected = (llm_scores[metric] + manual_scores[metric]) / 2.0
            actual = row[metric]
            assert not math.isnan(actual), f"{metric}: expected {expected}, got NaN"
            assert abs(actual - expected) < 1e-9, (
                f"{metric}: expected mean({llm_scores[metric]}, {manual_scores[metric]}) "
                f"= {expected}, got {actual}"
            )


@given(
    run_id=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
        min_size=1,
        max_size=30,
    ),
    llm_scores=st.fixed_dictionaries({m: _score_value for m in METRICS}),
)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_score_merge_arithmetic_llm_only(run_id, llm_scores):
    """
    Property 3b: When only LLM judge scores are present (no manual CSV),
    the combined metric score equals the LLM score.

    **Validates: Requirements 1.5, 1.6**
    """
    import math

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        log = _make_run_log(run_id, llm_scores)
        (tmp_path / f"{run_id}.json").write_text(json.dumps(log), encoding="utf-8")
        # No scoring CSV written

        loader = DataLoader()
        result = loader.build_run_index(tmp_path)

        assert len(result) == 1
        row = result.iloc[0]

        for metric in METRICS:
            actual = row[metric]
            assert not math.isnan(actual), f"{metric}: expected {llm_scores[metric]}, got NaN"
            assert abs(actual - llm_scores[metric]) < 1e-9, (
                f"{metric}: expected {llm_scores[metric]}, got {actual}"
            )


@given(
    run_id=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
        min_size=1,
        max_size=30,
    ),
    manual_scores=st.fixed_dictionaries({m: _score_value for m in METRICS}),
)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_score_merge_arithmetic_manual_only(run_id, manual_scores):
    """
    Property 3c: When only manual scores are present (no llm_judge in scores),
    the combined metric score equals the manual score.

    **Validates: Requirements 1.5, 1.6**
    """
    import math

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        scoring_dir = tmp_path / "scoring"
        scoring_dir.mkdir()

        # Run log with no LLM scores
        log = _make_run_log(run_id, None)
        (tmp_path / f"{run_id}.json").write_text(json.dumps(log), encoding="utf-8")

        # Write manual scoring CSV
        csv_content = _make_scoring_csv(run_id, manual_scores)
        (scoring_dir / f"{run_id}.csv").write_text(csv_content, encoding="utf-8")

        loader = DataLoader()
        result = loader.build_run_index(tmp_path)

        assert len(result) == 1
        row = result.iloc[0]

        for metric in METRICS:
            actual = row[metric]
            assert not math.isnan(actual), f"{metric}: expected {manual_scores[metric]}, got NaN"
            assert abs(actual - manual_scores[metric]) < 1e-9, (
                f"{metric}: expected {manual_scores[metric]}, got {actual}"
            )


@given(
    run_id=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
        min_size=1,
        max_size=30,
    ),
)
@settings(max_examples=30, suppress_health_check=(HealthCheck.too_slow,))
def test_score_merge_arithmetic_neither_present(run_id):
    """
    Property 3d: When neither LLM judge scores nor manual scores are present,
    the combined metric score is NaN.

    **Validates: Requirements 1.5, 1.6**
    """
    import math

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Run log with no LLM scores, no scoring CSV
        log = _make_run_log(run_id, None)
        (tmp_path / f"{run_id}.json").write_text(json.dumps(log), encoding="utf-8")

        loader = DataLoader()
        result = loader.build_run_index(tmp_path)

        assert len(result) == 1
        row = result.iloc[0]

        for metric in METRICS:
            actual = row[metric]
            assert actual is None or (isinstance(actual, float) and math.isnan(actual)), (
                f"{metric}: expected NaN when no scores present, got {actual}"
            )


# ---------------------------------------------------------------------------
# Property 4: Empty filters return full RunIndex
# ---------------------------------------------------------------------------

from dashboard.data_loader import apply_filters, FilterState

_model_text = st.text(
    min_size=1, max_size=20,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_:"),
)
_scenario_text = st.text(
    min_size=1, max_size=20,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
)

_run_index_row = st.fixed_dictionaries({
    "model": _model_text,
    "scenario_id": _scenario_text,
    "timestamp": st.just("2024-01-01T00:00:00+00:00"),
})


@given(rows=st.lists(_run_index_row, min_size=0, max_size=20))
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_empty_filters_return_full_run_index(rows):
    """
    Property 4: Empty filters return full RunIndex.

    For any RunIndex DataFrame, calling apply_filters with a FilterState where
    all fields are empty or None returns a DataFrame with the same number of
    rows as the input.

    **Validates: Requirements 2.3**
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["model", "scenario_id", "timestamp"])

    empty_filter: FilterState = {
        "models": [],
        "scenarios": [],
        "date_from": None,
        "date_to": None,
    }

    result = apply_filters(df, empty_filter)

    assert len(result) == len(df), (
        f"Expected {len(df)} rows with empty filters, got {len(result)}"
    )


# ---------------------------------------------------------------------------
# Property 5: Non-empty filters return only matching rows
# ---------------------------------------------------------------------------

from datetime import date as _date

_date_strategy = st.dates(
    min_value=_date(2023, 1, 1),
    max_value=_date(2025, 12, 31),
)

# Strategy for a non-empty FilterState: at least one filter field is active.
# We generate all four fields and then ensure at least one is non-empty/non-None.
_filter_state_nonempty = st.fixed_dictionaries({
    "models": st.lists(_model_text, min_size=0, max_size=3),
    "scenarios": st.lists(_scenario_text, min_size=0, max_size=3),
    "date_from": st.one_of(st.none(), _date_strategy),
    "date_to": st.one_of(st.none(), _date_strategy),
}).filter(
    lambda fs: bool(fs["models"]) or bool(fs["scenarios"])
              or fs["date_from"] is not None or fs["date_to"] is not None
)


@given(
    rows=st.lists(_run_index_row, min_size=0, max_size=20),
    filter_state=_filter_state_nonempty,
)
@settings(max_examples=60, suppress_health_check=(HealthCheck.too_slow,))
def test_nonempty_filters_return_only_matching_rows(rows, filter_state):
    """
    Property 5: Non-empty filters return only matching rows.

    For any RunIndex DataFrame and any non-empty FilterState, every row in the
    result of apply_filters satisfies all active filter conditions, and no row
    that fails any active condition appears in the result.

    **Validates: Requirements 2.4**
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["model", "scenario_id", "timestamp"])

    result = apply_filters(df, filter_state)

    # --- Forward check: every row in result satisfies all active conditions ---
    for idx, row in result.iterrows():
        if filter_state.get("models"):
            assert row["model"] in filter_state["models"], (
                f"Row {idx} has model={row['model']!r} which is NOT in "
                f"active models filter {filter_state['models']}"
            )
        if filter_state.get("scenarios"):
            assert row["scenario_id"] in filter_state["scenarios"], (
                f"Row {idx} has scenario_id={row['scenario_id']!r} which is NOT in "
                f"active scenarios filter {filter_state['scenarios']}"
            )
        date_from = filter_state.get("date_from")
        date_to = filter_state.get("date_to")
        if date_from is not None or date_to is not None:
            ts = pd.Timestamp(row["timestamp"], tz="UTC")
            if date_from is not None:
                assert ts >= pd.Timestamp(date_from, tz="UTC"), (
                    f"Row {idx} timestamp {ts} is before date_from {date_from}"
                )
            if date_to is not None:
                upper = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                assert ts <= upper, (
                    f"Row {idx} timestamp {ts} is after date_to {date_to} (upper={upper})"
                )

    # --- Reverse check: no row that fails any active condition appears in result ---
    result_indices = set(result.index)
    for idx, row in df.iterrows():
        fails = False
        if filter_state.get("models") and row["model"] not in filter_state["models"]:
            fails = True
        if filter_state.get("scenarios") and row["scenario_id"] not in filter_state["scenarios"]:
            fails = True
        date_from = filter_state.get("date_from")
        date_to = filter_state.get("date_to")
        if not fails and (date_from is not None or date_to is not None):
            ts = pd.Timestamp(row["timestamp"], tz="UTC")
            if date_from is not None and ts < pd.Timestamp(date_from, tz="UTC"):
                fails = True
            if date_to is not None:
                upper = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                if ts > upper:
                    fails = True
        if fails:
            assert idx not in result_indices, (
                f"Row {idx} (model={row['model']!r}, scenario_id={row['scenario_id']!r}, "
                f"timestamp={row['timestamp']!r}) fails a filter condition but appears in result"
            )


# ---------------------------------------------------------------------------
# Property 17: Keyword filter returns only runs containing the keyword
# ---------------------------------------------------------------------------

@given(
    run_ids=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
            min_size=1,
            max_size=20,
        ),
        min_size=1,
        max_size=8,
        unique=True,
    ),
    keyword=st.text(
        alphabet=st.characters(whitelist_categories=("Ll",)),
        min_size=1,
        max_size=10,
    ),
    matching_indices=st.frozensets(st.integers(min_value=0, max_value=7), max_size=8),
)
@settings(max_examples=40, suppress_health_check=(HealthCheck.too_slow,))
def test_keyword_filter_returns_only_matching_runs(run_ids, keyword, matching_indices):
    """
    Property 17: Keyword filter returns only runs containing the keyword.

    For any set of run JSON files and any non-empty keyword:
    - Every run in the result contains the keyword in at least one turn's text field.
    - Every run NOT in the result contains no turn with the keyword.

    **Validates: Requirements 23.2**
    """
    import pandas as pd
    from dashboard.data_loader import apply_filters, FilterState

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Build run index rows and write JSON files
        rows = []
        expected_matching = set()

        for i, run_id in enumerate(run_ids):
            # Decide if this run should match the keyword
            should_match = i in matching_indices

            if should_match:
                # Include keyword in one turn's text
                conversation = [
                    {"role": "user", "text": f"some text with {keyword} in it"},
                    {"role": "assistant", "text": "a response"},
                ]
                expected_matching.add(run_id)
            else:
                # Ensure keyword is NOT present — use digits only (keyword is lowercase letters)
                safe_text = "0000000000"
                conversation = [
                    {"role": "user", "text": safe_text},
                    {"role": "assistant", "text": safe_text},
                ]

            run_data = {
                "run_id": run_id,
                "conversation": conversation,
            }
            (tmp_path / f"{run_id}.json").write_text(
                json.dumps(run_data), encoding="utf-8"
            )

            rows.append({
                "run_id": run_id,
                "model": "test_model",
                "scenario_id": "test_scenario",
                "timestamp": "2024-01-01T00:00:00+00:00",
            })

        df = pd.DataFrame(rows)

        filter_state: FilterState = {
            "models": [],
            "scenarios": [],
            "date_from": None,
            "date_to": None,
            "keyword": keyword,
            "flagged_only": False,
        }

        result = apply_filters(df, filter_state, logs_dir=tmp_path)
        result_ids = set(result["run_id"].tolist())

        # Every returned run must be in expected_matching
        assert result_ids <= expected_matching, (
            f"Runs returned that don't contain keyword {keyword!r}: "
            f"{result_ids - expected_matching}"
        )

        # Every expected matching run must be returned
        assert expected_matching <= result_ids, (
            f"Matching runs missing from result for keyword {keyword!r}: "
            f"{expected_matching - result_ids}"
        )


@given(
    run_ids=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
            min_size=1,
            max_size=20,
        ),
        min_size=0,
        max_size=8,
        unique=True,
    ),
)
@settings(max_examples=30, suppress_health_check=(HealthCheck.too_slow,))
def test_empty_keyword_returns_all_runs(run_ids):
    """
    Property 17 (empty keyword case): When keyword is empty, all runs are returned.

    **Validates: Requirements 23.2, 23.4**
    """
    import pandas as pd
    from dashboard.data_loader import apply_filters, FilterState

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        rows = []
        for run_id in run_ids:
            run_data = {
                "run_id": run_id,
                "conversation": [{"role": "user", "text": "hello"}],
            }
            (tmp_path / f"{run_id}.json").write_text(
                json.dumps(run_data), encoding="utf-8"
            )
            rows.append({
                "run_id": run_id,
                "model": "test_model",
                "scenario_id": "test_scenario",
                "timestamp": "2024-01-01T00:00:00+00:00",
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["run_id", "model", "scenario_id", "timestamp"]
        )

        filter_state: FilterState = {
            "models": [],
            "scenarios": [],
            "date_from": None,
            "date_to": None,
            "keyword": "",
            "flagged_only": False,
        }

        result = apply_filters(df, filter_state, logs_dir=tmp_path)

        assert len(result) == len(df), (
            f"Empty keyword should return all {len(df)} runs, got {len(result)}"
        )

"""
Property-based tests for ManualScoringUI.render_manual_scoring_ui.

**Validates: Requirements 5.3**
"""

import sys
import types

# Stub streamlit before importing dashboard modules (not installed in test env)
_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data = lambda fn=None, **kw: (fn if fn else lambda f: f)
_st_stub.session_state = {}
sys.modules.setdefault("streamlit", _st_stub)

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.views.scoring import render_manual_scoring_ui

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)

_score_value = st.integers(min_value=1, max_value=5)

_turn_number = st.integers(min_value=1, max_value=50)


def _existing_scores_strategy(turn_numbers):
    """Build a strategy that generates an existing_scores DataFrame for the given turns."""
    rows = []
    for turn_num in turn_numbers:
        rows.append({
            "turn": turn_num,
            "identity_consistency": st.integers(min_value=1, max_value=5),
            "cultural_authenticity": st.integers(min_value=1, max_value=5),
            "naturalness": st.integers(min_value=1, max_value=5),
            "information_yield": st.integers(min_value=1, max_value=5),
            "notes": st.text(min_size=0, max_size=50),
            "rater_id": st.just("rater1"),
        })
    return st.fixed_dictionaries({
        str(i): st.fixed_dictionaries(row) for i, row in enumerate(rows)
    })


# Strategy: list of unique turn numbers (1–20), at least 1
_turn_numbers_strategy = st.lists(
    st.integers(min_value=1, max_value=20),
    min_size=1,
    max_size=8,
    unique=True,
)

# Strategy: a single row of scores for one turn
_score_row_strategy = st.fixed_dictionaries({
    "identity_consistency": st.integers(min_value=1, max_value=5),
    "cultural_authenticity": st.integers(min_value=1, max_value=5),
    "naturalness": st.integers(min_value=1, max_value=5),
    "information_yield": st.integers(min_value=1, max_value=5),
    "notes": st.text(min_size=0, max_size=50),
    "rater_id": st.just("rater1"),
})


# ---------------------------------------------------------------------------
# Property 8: Pre-population round trip
# ---------------------------------------------------------------------------

@given(
    turn_numbers=_turn_numbers_strategy,
    score_rows=st.lists(
        _score_row_strategy,
        min_size=1,
        max_size=8,
    ),
)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_manual_scoring_ui_prepopulates_sliders_from_existing_scores(
    turn_numbers, score_rows
):
    """
    Property 8: Pre-population round trip.

    For any existing scores DataFrame with turn numbers and metric values,
    the slider values rendered by render_manual_scoring_ui equal the score
    values read from that DataFrame.

    **Validates: Requirements 5.3**
    """
    # Align score_rows to turn_numbers (zip, truncate to shorter)
    pairs = list(zip(turn_numbers, score_rows))

    # Build existing_scores DataFrame
    rows = []
    for turn_num, score_row in pairs:
        rows.append({
            "turn": turn_num,
            "identity_consistency": score_row["identity_consistency"],
            "cultural_authenticity": score_row["cultural_authenticity"],
            "naturalness": score_row["naturalness"],
            "information_yield": score_row["information_yield"],
            "notes": score_row["notes"],
            "rater_id": score_row["rater_id"],
        })
    existing_scores = pd.DataFrame(rows)

    # Build run_data with model turns matching the turn numbers
    model_turns = [
        {"speaker": "model", "turn": turn_num, "text": f"Response for turn {turn_num}"}
        for turn_num, _ in pairs
    ]
    run_data = {
        "run_id": "test-run-prop8",
        "conversation": model_turns,
    }

    # Capture slider calls: record the `value=` kwarg for each call
    slider_calls = []  # list of (label, value) tuples

    def fake_slider(label, min_value=1, max_value=5, value=1, key=None):
        slider_calls.append((label, value))
        return value

    # Mock context managers for st.expander and st.columns
    def make_ctx():
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with patch("dashboard.views.scoring.st") as mock_st:
        mock_st.session_state = {}
        mock_st.text_input = MagicMock(return_value="")
        mock_st.slider.side_effect = fake_slider
        mock_st.expander.return_value = make_ctx()
        mock_st.columns.return_value = [make_ctx() for _ in range(4)]
        mock_st.markdown = MagicMock()
        mock_st.text = MagicMock()
        mock_st.text_area = MagicMock(return_value="")
        mock_st.button = MagicMock(return_value=False)
        mock_st.info = MagicMock()

        render_manual_scoring_ui(
            run_data=run_data,
            existing_scores=existing_scores,
            scoring_dir=Path("/tmp/scoring"),
        )

    _METRICS = [
        "identity_consistency",
        "cultural_authenticity",
        "naturalness",
        "information_yield",
    ]

    # Build expected slider values: for each turn, for each metric, the existing score
    expected = []
    for turn_num, score_row in pairs:
        for metric in _METRICS:
            expected.append(score_row[metric])

    # Extract actual slider values in order
    actual = [value for (_, value) in slider_calls]

    assert len(actual) == len(expected), (
        f"Expected {len(expected)} slider calls, got {len(actual)}. "
        f"Slider calls: {slider_calls}"
    )

    for i, (exp_val, act_val) in enumerate(zip(expected, actual)):
        assert act_val == exp_val, (
            f"Slider {i} value mismatch: expected {exp_val}, got {act_val}. "
            f"All slider calls: {slider_calls}"
        )


# ---------------------------------------------------------------------------
# Property 9: Incomplete scores are rejected
# ---------------------------------------------------------------------------

# Strategy: pick which metric index (0–3) returns 0 for a given turn
_metric_index = st.integers(min_value=0, max_value=3)

# Strategy: which turn index (0-based) has the incomplete metric
_incomplete_turn_index = st.integers(min_value=0, max_value=4)


@given(
    turn_numbers=st.lists(
        st.integers(min_value=1, max_value=20),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    incomplete_turn_idx=st.integers(min_value=0, max_value=4),
    incomplete_metric_idx=st.integers(min_value=0, max_value=3),
    rater_id=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_manual_scoring_ui_rejects_incomplete_scores(
    turn_numbers, incomplete_turn_idx, incomplete_metric_idx, rater_id
):
    """
    Property 9: Incomplete scores are rejected.

    For any set of TurnScoreEntry values where at least one metric field is 0 (None-equivalent),
    the ManualScoringUI save action does not invoke ScoreWriter and instead surfaces a
    validation error.

    **Validates: Requirements 5.4**
    """
    _METRICS_LIST = [
        "identity_consistency",
        "cultural_authenticity",
        "naturalness",
        "information_yield",
    ]

    # Clamp incomplete_turn_idx to valid range
    bad_turn_idx = incomplete_turn_idx % len(turn_numbers)
    bad_metric = _METRICS_LIST[incomplete_metric_idx % len(_METRICS_LIST)]
    bad_turn_num = turn_numbers[bad_turn_idx]

    # Build run_data with model turns
    model_turns = [
        {"speaker": "model", "turn": turn_num, "text": f"Response for turn {turn_num}"}
        for turn_num in turn_numbers
    ]
    run_data = {
        "run_id": "test-run-prop9",
        "conversation": model_turns,
    }

    # Mock st.slider: return 0 for the bad metric on the bad turn, else 3
    def fake_slider(label, min_value=1, max_value=5, value=1, key=None):
        # key format: score_{run_id}_{turn_num}_{metric}
        if key and f"_{bad_turn_num}_{bad_metric}" in key:
            return 0
        return 3

    def make_ctx():
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with patch("dashboard.views.scoring.st") as mock_st, \
         patch("dashboard.views.scoring.ScoreWriter") as mock_writer_cls:

        mock_st.session_state = {f"manual_scoring_rater_id": rater_id}
        mock_st.text_input = MagicMock(return_value=rater_id)
        mock_st.slider.side_effect = fake_slider
        mock_st.expander.return_value = make_ctx()
        mock_st.columns.return_value = [make_ctx() for _ in range(4)]
        mock_st.markdown = MagicMock()
        mock_st.text = MagicMock()
        mock_st.text_area = MagicMock(return_value="")
        # Simulate clicking "Save Scores"
        mock_st.button = MagicMock(return_value=True)
        mock_st.error = MagicMock()
        mock_st.success = MagicMock()
        mock_st.info = MagicMock()

        mock_writer_instance = MagicMock()
        mock_writer_cls.return_value = mock_writer_instance

        render_manual_scoring_ui(
            run_data=run_data,
            existing_scores=pd.DataFrame(),
            scoring_dir=Path("/tmp/scoring"),
        )

    # ScoreWriter.save_manual_scores must NOT have been called
    mock_writer_instance.save_manual_scores.assert_not_called()

    # st.error must have been called (validation error surfaced)
    mock_st.error.assert_called()

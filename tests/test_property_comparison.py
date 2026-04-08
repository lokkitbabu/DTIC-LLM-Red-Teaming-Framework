"""
Property-based tests for ComparisonView.render_comparison_view.

**Validates: Requirements 7.3**
"""

import sys
import types

# Stub streamlit before importing dashboard modules (not installed in test env)
_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data = lambda fn=None, **kw: (fn if fn else lambda f: f)
_st_stub.session_state = {}
sys.modules.setdefault("streamlit", _st_stub)

from unittest.mock import patch, MagicMock

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.views.comparison import render_comparison_view

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_METRICS = [
    "identity_consistency",
    "cultural_authenticity",
    "naturalness",
    "information_yield",
]

# A metric score: float in [0.0, 10.0] with up to 2 decimal places
_score_value = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Strategy for llm_judge scores dict — all four metrics present
_llm_judge_strategy = st.fixed_dictionaries({
    metric: _score_value for metric in _METRICS
})

# Strategy for a minimal run dict with scores.llm_judge populated
def _run_strategy(llm_judge_st):
    return st.fixed_dictionaries({
        "run_id": st.just("run-a"),
        "scores": st.fixed_dictionaries({"llm_judge": llm_judge_st}),
        "conversation": st.just([]),
    })


# ---------------------------------------------------------------------------
# Property 13: Score delta correctness
# ---------------------------------------------------------------------------

@given(
    scores_a=_llm_judge_strategy,
    scores_b=_llm_judge_strategy,
)
@settings(max_examples=100, suppress_health_check=(HealthCheck.too_slow,))
def test_comparison_view_score_delta_correctness(scores_a, scores_b):
    """
    Property 13: Score delta correctness.

    For any two runs A and B, the score delta value displayed by ComparisonView
    for each metric equals the metric score of run B minus the metric score of run A.

    **Validates: Requirements 7.3**
    """
    run_a = {
        "run_id": "run-a",
        "scores": {"llm_judge": scores_a},
        "conversation": [],
    }
    run_b = {
        "run_id": "run-b",
        "scores": {"llm_judge": scores_b},
        "conversation": [],
    }

    captured_df = {}

    def fake_dataframe(df, **kwargs):
        captured_df["df"] = df

    def make_ctx():
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with patch("dashboard.views.comparison.st") as mock_st:
        mock_st.columns.return_value = [make_ctx(), make_ctx()]
        mock_st.subheader = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.dataframe.side_effect = fake_dataframe
        mock_st.info = MagicMock()
        mock_st.expander.return_value = make_ctx()

        render_comparison_view(run_a, run_b)

    assert "df" in captured_df, "st.dataframe was never called — score delta table not rendered"

    df = captured_df["df"]
    assert "Delta (B − A)" in df.columns, f"Expected 'Delta (B − A)' column, got: {list(df.columns)}"

    from dashboard.views.comparison import _METRIC_LABELS

    for metric in _METRICS:
        label = _METRIC_LABELS[metric]
        row = df[df["Metric"] == label]
        assert len(row) == 1, f"Expected exactly one row for metric '{label}'"

        delta_cell = row["Delta (B − A)"].iloc[0]

        score_a = scores_a[metric]
        score_b = scores_b[metric]
        expected_delta = round(float(score_b) - float(score_a), 2)
        expected_str = f"+{expected_delta}" if expected_delta > 0 else str(expected_delta)

        assert delta_cell == expected_str, (
            f"Metric '{metric}': expected delta '{expected_str}', got '{delta_cell}' "
            f"(score_a={score_a}, score_b={score_b})"
        )

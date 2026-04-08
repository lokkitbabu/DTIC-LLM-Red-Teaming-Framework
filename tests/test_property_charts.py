"""
Property-based tests for AggregateChartsView error bar computation.

**Validates: Requirements 16.1**
"""

import sys
import types

# Stub streamlit before importing dashboard modules
_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data = lambda fn=None, **kw: (fn if fn else lambda f: f)
_st_stub.session_state = {}
sys.modules.setdefault("streamlit", _st_stub)

import math

import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.views.charts import _compute_error_bars, METRICS

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_score_value = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)


def _make_group_df(group_name: str, scores: list[float], metric: str) -> pd.DataFrame:
    """Build a minimal DataFrame for a single group with one metric column."""
    return pd.DataFrame({
        "model": [group_name] * len(scores),
        metric: scores,
    })


# Strategy: a list of 2+ score values (multi-run group)
_multi_run_scores = st.lists(_score_value, min_size=2, max_size=20)

# Strategy: exactly one score value (single-run group)
_single_run_score = st.lists(_score_value, min_size=1, max_size=1)


# ---------------------------------------------------------------------------
# Property 22: Score confidence interval error bar equals std dev
# ---------------------------------------------------------------------------

@given(scores=_multi_run_scores)
@settings(max_examples=200, suppress_health_check=(HealthCheck.too_slow,))
def test_error_bar_equals_std_dev_for_multi_run_group(scores):
    """
    Property 22: Score confidence interval error bar equals std dev.

    For any group of runs with more than one run, the error bar value equals
    the standard deviation of scores in that group.

    **Validates: Requirements 16.1**
    """
    metric = "identity_consistency"
    df = _make_group_df("model_a", scores, metric)

    result = _compute_error_bars(df, "model", [metric])

    assert len(result) == 1, "Expected exactly one group row"
    row = result.iloc[0]

    mean_val = row[f"{metric}_mean"]
    std_val = row[f"{metric}_std"]

    # Mean should equal the arithmetic mean of scores
    expected_mean = sum(scores) / len(scores)
    assert math.isclose(mean_val, expected_mean, rel_tol=1e-6, abs_tol=1e-9), (
        f"Mean mismatch: expected {expected_mean}, got {mean_val}"
    )

    # Std should be non-None for multi-run groups
    assert std_val is not None, (
        f"Expected non-None std dev for group with {len(scores)} runs, got None"
    )

    # Std should equal pandas std (ddof=1 by default)
    expected_std = pd.Series(scores).std()
    assert math.isclose(std_val, expected_std, rel_tol=1e-6, abs_tol=1e-9), (
        f"Std dev mismatch: expected {expected_std}, got {std_val}"
    )


@given(score=_score_value)
@settings(max_examples=100, suppress_health_check=(HealthCheck.too_slow,))
def test_no_error_bar_for_single_run_group(score):
    """
    Property 22 (single-run case): For a group with exactly one run, no error
    bar is shown (std is None).

    **Validates: Requirements 16.1, 16.2**
    """
    metric = "identity_consistency"
    df = _make_group_df("model_a", [score], metric)

    result = _compute_error_bars(df, "model", [metric])

    assert len(result) == 1, "Expected exactly one group row"
    row = result.iloc[0]

    std_val = row[f"{metric}_std"]
    is_absent = std_val is None or (isinstance(std_val, float) and math.isnan(std_val))
    assert is_absent, (
        f"Expected None/NaN std dev for single-run group, got {std_val}"
    )

    mean_val = row[f"{metric}_mean"]
    assert math.isclose(mean_val, score, rel_tol=1e-9, abs_tol=1e-9), (
        f"Mean mismatch for single-run group: expected {score}, got {mean_val}"
    )


@given(
    scores_a=_multi_run_scores,
    scores_b=_single_run_score,
)
@settings(max_examples=100, suppress_health_check=(HealthCheck.too_slow,))
def test_mixed_groups_error_bars(scores_a, scores_b):
    """
    Property 22 (mixed groups): When multiple groups exist, error bars are
    present only for groups with more than one run.

    **Validates: Requirements 16.1, 16.2**
    """
    metric = "identity_consistency"
    df = pd.DataFrame({
        "model": ["multi_run"] * len(scores_a) + ["single_run"] * len(scores_b),
        metric: scores_a + scores_b,
    })

    result = _compute_error_bars(df, "model", [metric])
    result = result.set_index("model")

    # Multi-run group should have a non-None std
    multi_std = result.loc["multi_run", f"{metric}_std"]
    assert multi_std is not None, (
        f"Expected non-None std for multi-run group, got None"
    )

    # Single-run group should have None or NaN std (no error bar)
    single_std = result.loc["single_run", f"{metric}_std"]
    is_absent = single_std is None or (isinstance(single_std, float) and math.isnan(single_std))
    assert is_absent, (
        f"Expected None/NaN std for single-run group, got {single_std}"
    )

"""
Property-based tests for persistent filter state logic.

**Validates: Requirements 26.1, 26.2**
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# Stub streamlit before importing dashboard modules (must come before any dashboard import)
if "streamlit" not in sys.modules:
    _st_stub = MagicMock()

    def _cache_data_stub(fn=None, **kw):
        if fn is not None:
            return fn
        return lambda f: f

    _cache_data_stub.clear = lambda: None
    _st_stub.cache_data = _cache_data_stub
    _st_stub.session_state = {}
    sys.modules["streamlit"] = _st_stub

from datetime import date
from typing import Optional

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.data_loader import _merge_filter_state, FilterState


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_model_str = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_:"),
    min_size=1,
    max_size=30,
)

_models_strategy = st.lists(_model_str, min_size=0, max_size=5)
_scenarios_strategy = st.lists(_model_str, min_size=0, max_size=5)

_date_strategy = st.one_of(
    st.none(),
    st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 12, 31)),
)

_keyword_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" -_"),
    min_size=0,
    max_size=50,
)

_filter_state_strategy = st.fixed_dictionaries({
    "models": _models_strategy,
    "scenarios": _scenarios_strategy,
    "date_from": _date_strategy,
    "date_to": _date_strategy,
    "keyword": _keyword_strategy,
    "flagged_only": st.booleans(),
})


# ---------------------------------------------------------------------------
# Property 20: Persistent filter state survives page navigation
# ---------------------------------------------------------------------------

@given(filter_state=_filter_state_strategy)
@settings(max_examples=200, suppress_health_check=(HealthCheck.too_slow,))
def test_filter_state_round_trip_through_session_state(filter_state):
    """
    Property 20: Persistent filter state survives page navigation.

    Simulates storing a FilterState in session_state (represented as a plain
    dict) and reading it back after a "page navigation" (a no-op merge with
    no updates). The retrieved state must equal the stored state for every
    field.

    **Validates: Requirements 26.1, 26.2**
    """
    # Simulate writing to session_state["filter_state"]
    session_state = {}
    session_state["filter_state"] = dict(filter_state)

    # Simulate page navigation: read back from session_state with no updates
    retrieved = _merge_filter_state(session_state["filter_state"], {})

    assert retrieved["models"] == filter_state["models"], (
        f"models mismatch: stored={filter_state['models']!r}, retrieved={retrieved['models']!r}"
    )
    assert retrieved["scenarios"] == filter_state["scenarios"], (
        f"scenarios mismatch: stored={filter_state['scenarios']!r}, retrieved={retrieved['scenarios']!r}"
    )
    assert retrieved["date_from"] == filter_state["date_from"], (
        f"date_from mismatch: stored={filter_state['date_from']!r}, retrieved={retrieved['date_from']!r}"
    )
    assert retrieved["date_to"] == filter_state["date_to"], (
        f"date_to mismatch: stored={filter_state['date_to']!r}, retrieved={retrieved['date_to']!r}"
    )
    assert retrieved["keyword"] == filter_state["keyword"], (
        f"keyword mismatch: stored={filter_state['keyword']!r}, retrieved={retrieved['keyword']!r}"
    )
    assert retrieved["flagged_only"] == filter_state["flagged_only"], (
        f"flagged_only mismatch: stored={filter_state['flagged_only']!r}, retrieved={retrieved['flagged_only']!r}"
    )


@given(
    initial=_filter_state_strategy,
    updates=_filter_state_strategy,
)
@settings(max_examples=200, suppress_health_check=(HealthCheck.too_slow,))
def test_filter_state_updates_override_existing(initial, updates):
    """
    Property 20 (update variant): When filter widget values change, the new
    values override the stored state and all fields in the result reflect the
    updates dict.

    **Validates: Requirements 26.1, 26.2**
    """
    result = _merge_filter_state(initial, updates)

    # Every field in updates must appear verbatim in the merged result
    for field in ("models", "scenarios", "date_from", "date_to", "keyword", "flagged_only"):
        assert result[field] == updates[field], (
            f"Field '{field}': expected update value {updates[field]!r}, "
            f"got {result[field]!r}"
        )


@given(
    initial=_filter_state_strategy,
    partial_updates=st.fixed_dictionaries({
        "models": _models_strategy,
        "keyword": _keyword_strategy,
    }),
)
@settings(max_examples=200, suppress_health_check=(HealthCheck.too_slow,))
def test_filter_state_partial_update_preserves_untouched_fields(initial, partial_updates):
    """
    Property 20 (partial update variant): Fields not present in the updates
    dict are preserved from the existing state.

    **Validates: Requirements 26.1, 26.2**
    """
    result = _merge_filter_state(initial, partial_updates)

    # Updated fields reflect the new values
    assert result["models"] == partial_updates["models"]
    assert result["keyword"] == partial_updates["keyword"]

    # Untouched fields are preserved from initial
    assert result["scenarios"] == initial["scenarios"]
    assert result["date_from"] == initial["date_from"]
    assert result["date_to"] == initial["date_to"]
    assert result["flagged_only"] == initial["flagged_only"]

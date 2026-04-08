"""
Property-based tests for SummaryView export functionality.

**Validates: Requirements 18.3, 18.4**
"""

import csv
import io
import json

import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.views.summary import _build_export_bytes


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_ "),
    min_size=1,
    max_size=30,
)

_optional_float = st.one_of(st.none(), st.floats(min_value=1.0, max_value=5.0, allow_nan=False))


def _run_index_strategy():
    """Generate a DataFrame that resembles a filtered RunIndex."""
    row = st.fixed_dictionaries({
        "run_id": _safe_text,
        "scenario_id": _safe_text,
        "model": _safe_text,
        "timestamp": st.just("2024-01-01T00:00:00Z"),
        "identity_consistency": _optional_float,
        "cultural_authenticity": _optional_float,
        "naturalness": _optional_float,
        "information_yield": _optional_float,
    })
    return st.lists(row, min_size=0, max_size=50).map(pd.DataFrame)


# ---------------------------------------------------------------------------
# Property 21: Export row count matches filtered RunIndex
# ---------------------------------------------------------------------------

@given(df=_run_index_strategy())
@settings(max_examples=100, suppress_health_check=(HealthCheck.too_slow,))
def test_export_csv_row_count_matches_dataframe(df):
    """
    Property 21 (CSV): the number of data rows in the exported CSV equals
    the number of rows in the filtered RunIndex.

    **Validates: Requirements 18.3**
    """
    csv_bytes = _build_export_bytes(df, "CSV")
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
    rows = list(reader)
    # First row is the header; remaining rows are data
    data_rows = [r for r in rows[1:] if r]  # skip empty trailing lines
    assert len(data_rows) == len(df), (
        f"CSV data rows ({len(data_rows)}) != DataFrame rows ({len(df)})"
    )


@given(df=_run_index_strategy())
@settings(max_examples=100, suppress_health_check=(HealthCheck.too_slow,))
def test_export_json_record_count_matches_dataframe(df):
    """
    Property 21 (JSON): the number of records in the exported JSON array
    equals the number of rows in the filtered RunIndex.

    **Validates: Requirements 18.4**
    """
    json_bytes = _build_export_bytes(df, "JSON")
    records = json.loads(json_bytes.decode("utf-8"))
    assert isinstance(records, list), "JSON export should be a list of records"
    assert len(records) == len(df), (
        f"JSON records ({len(records)}) != DataFrame rows ({len(df)})"
    )

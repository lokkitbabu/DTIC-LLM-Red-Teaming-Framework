"""
Property-based tests for FlagManager.

**Validates: Requirements 24.2**
"""

import tempfile
from pathlib import Path

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.flag_manager import FlagManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_run_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=40,
)


# ---------------------------------------------------------------------------
# Property 18: Flag toggle is idempotent over two calls
# ---------------------------------------------------------------------------

@given(run_id=_run_id_strategy)
@settings(max_examples=100, suppress_health_check=(HealthCheck.too_slow,))
def test_toggle_flag_twice_restores_original_state(run_id):
    """
    Property 18: Flag toggle is idempotent over two calls.

    For any run_id, calling toggle_flag(run_id) twice returns the run to its
    original flag state.

    **Validates: Requirements 24.2**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        flags_file = Path(tmp_dir) / "flags.json"
        manager = FlagManager(flags_file=flags_file)

        original_state = manager.is_flagged(run_id)

        manager.toggle_flag(run_id)
        manager.toggle_flag(run_id)

        final_state = manager.is_flagged(run_id)

    assert final_state == original_state, (
        f"After two toggles, expected flag state {original_state} for run_id={run_id!r}, "
        f"but got {final_state}"
    )

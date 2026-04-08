"""
Property-based tests for the scenario validator.

**Validates: Requirements 22.2, 22.4**
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.views.scenarios import REQUIRED_SCENARIO_KEYS, _validate_scenario_keys


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate arbitrary extra keys that are not in REQUIRED_SCENARIO_KEYS
_extra_key_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
).filter(lambda k: k not in REQUIRED_SCENARIO_KEYS)

# Generate a dict with all required keys present (values are arbitrary strings)
_full_scenario_strategy = st.fixed_dictionaries(
    {key: st.text(min_size=0, max_size=50) for key in REQUIRED_SCENARIO_KEYS}
)


# ---------------------------------------------------------------------------
# Property 19: Scenario validator rejects JSON missing any required key
# ---------------------------------------------------------------------------

@given(
    present_keys=st.frozensets(
        st.sampled_from(REQUIRED_SCENARIO_KEYS),
        min_size=0,
        max_size=len(REQUIRED_SCENARIO_KEYS) - 1,  # at least one key missing
    )
)
@settings(max_examples=200, suppress_health_check=(HealthCheck.too_slow,))
def test_validator_rejects_dict_missing_required_keys(present_keys):
    """
    Property 19 (part 1): For any JSON object missing at least one required key,
    _validate_scenario_keys returns a non-empty list of missing keys.

    **Validates: Requirements 22.2, 22.4**
    """
    data = {key: "value" for key in present_keys}
    missing = _validate_scenario_keys(data)

    assert len(missing) > 0, (
        f"Expected non-empty missing list for data with keys {set(present_keys)}, "
        f"but got empty list"
    )

    # Every reported missing key must actually be absent from data
    for key in missing:
        assert key not in data, (
            f"Key {key!r} reported as missing but is present in data"
        )

    # Every actually-missing required key must appear in the missing list
    expected_missing = set(REQUIRED_SCENARIO_KEYS) - set(present_keys)
    assert set(missing) == expected_missing, (
        f"Missing keys mismatch: reported {set(missing)}, expected {expected_missing}"
    )


@given(base=_full_scenario_strategy)
@settings(max_examples=100, suppress_health_check=(HealthCheck.too_slow,))
def test_validator_accepts_dict_with_all_required_keys(base):
    """
    Property 19 (part 2): For any JSON object containing all required keys,
    _validate_scenario_keys returns an empty list.

    **Validates: Requirements 22.2, 22.4**
    """
    missing = _validate_scenario_keys(base)

    assert missing == [], (
        f"Expected empty missing list for complete scenario dict, but got {missing}"
    )

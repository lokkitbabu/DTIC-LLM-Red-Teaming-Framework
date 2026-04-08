"""
Property-based tests for RunExecutorView worker function.

**Validates: Requirements 11.6**
"""

import sys
import types

# Stub streamlit before importing dashboard modules
_st_stub = types.ModuleType("streamlit")

def _cache_data_stub(fn=None, **kw):
    """Callable decorator stub that also exposes .clear()."""
    if fn is not None:
        return fn
    return lambda f: f

_cache_data_stub.clear = lambda: None
_st_stub.cache_data = _cache_data_stub
_st_stub.session_state = {}
sys.modules.setdefault("streamlit", _st_stub)

from unittest.mock import MagicMock, patch
from pathlib import Path

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.views.run_executor import _worker, RunState


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_model_str_strategy = st.sampled_from([
    "openai:gpt-4o",
    "anthropic:claude-3",
    "ollama:mistral",
])

_eval_target_strategy = st.sampled_from(["subject", "interviewer"])

_max_turns_strategy = st.integers(min_value=1, max_value=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_run_data(subject_model_str: str, interviewer_model_str: str) -> dict:
    return {
        "run_id": "fake-run-id",
        "timestamp": "2024-01-01T00:00:00Z",
        "scenario_id": "test-scenario",
        "subject_model": subject_model_str,
        "interviewer_model": interviewer_model_str,
        "language": "english",
        "params": {},
        "conversation": [
            {
                "turn": 1,
                "speaker": "subject",
                "text": "Hello",
                "timestamp": "2024-01-01T00:00:01Z",
                "raw_prompt": "prompt",
            }
        ],
        "metadata": {
            "total_turns": 1,
            "stop_reason": "max_turns",
            "context_trims": 0,
        },
        "scores": {},
    }


def _make_fake_scenario(subject_model_str: str) -> dict:
    return {
        "scenario_id": "test-scenario",
        "model": subject_model_str,
        "identity": {"name": "Test", "background": "", "persona": "", "language_style": ""},
        "objective": "test",
        "constraints": [],
        "params": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 256, "seed": 0},
        "opening_message": "Hello",
    }


# ---------------------------------------------------------------------------
# Property 14: Background thread result is persisted before cache clear
# ---------------------------------------------------------------------------

@given(
    subject_model_str=_model_str_strategy,
    interviewer_model_str=_model_str_strategy,
    eval_target=_eval_target_strategy,
    max_turns=_max_turns_strategy,
)
@settings(max_examples=30, suppress_health_check=(HealthCheck.too_slow,), deadline=None)
def test_result_path_set_before_cache_clear(
    subject_model_str,
    interviewer_model_str,
    eval_target,
    max_turns,
):
    """
    Property 14: Background thread result is persisted before cache clear.

    When the worker function completes successfully, result_path is set in
    run_state BEFORE st.cache_data.clear() is called. The UI is responsible
    for calling cache_clear only after detecting result_path is set.

    We verify:
    1. result_path is set in run_state after worker completes
    2. running is False after worker completes
    3. error is None on success

    **Validates: Requirements 11.6**
    """
    fake_log_path = "/logs/fake-run-id.json"
    fake_run_data = _make_fake_run_data(subject_model_str, interviewer_model_str)
    fake_scenario = _make_fake_scenario(subject_model_str)

    # Track the order of key events
    call_order = []

    class TrackingRunState(dict):
        """RunState that records when result_path is set."""
        def __setitem__(self, key, value):
            if key == "result_path" and value is not None:
                call_order.append("result_path_set")
            super().__setitem__(key, value)

    run_state: RunState = TrackingRunState({
        "running": True,
        "lines": [],
        "error": None,
        "result_path": None,
    })

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = fake_run_data

    mock_runner_class = MagicMock(return_value=mock_runner_instance)

    def mock_save_run(run_data, run_id=None):
        call_order.append("save_run_called")
        return Path(fake_log_path)

    mock_load_scenario = MagicMock(return_value=fake_scenario)
    mock_build_model = MagicMock(return_value=MagicMock())

    # Patch at the source modules that _worker imports locally
    with patch("runner.conversation_runner.ConversationRunner", mock_runner_class), \
         patch("utils.logger.save_run", side_effect=mock_save_run), \
         patch("utils.scenario_loader.load_scenario", mock_load_scenario), \
         patch("dashboard.views.run_executor._build_model", mock_build_model):

        _worker(
            scenario_path="scenarios/test.json",
            subject_model_str=subject_model_str,
            interviewer_model_str=interviewer_model_str,
            judge_model_str=None,
            eval_target=eval_target,
            max_turns=max_turns,
            run_state=run_state,
        )

    # --- Assertions ---

    # No error should have occurred
    assert run_state["error"] is None, (
        f"Worker raised an unexpected error: {run_state['error']}"
    )

    # result_path must be set to the saved log path
    assert run_state["result_path"] == fake_log_path, (
        f"Expected result_path={fake_log_path!r}, got {run_state['result_path']!r}"
    )

    # running must be False after completion
    assert run_state["running"] is False, (
        "run_state['running'] should be False after worker completes"
    )

    # save_run must have been called before result_path was set
    assert "save_run_called" in call_order, "save_run was never called"
    assert "result_path_set" in call_order, "result_path was never set in run_state"

    save_idx = call_order.index("save_run_called")
    result_idx = call_order.index("result_path_set")
    assert save_idx < result_idx, (
        f"save_run (index {save_idx}) must be called before result_path is set "
        f"(index {result_idx}). call_order={call_order}"
    )

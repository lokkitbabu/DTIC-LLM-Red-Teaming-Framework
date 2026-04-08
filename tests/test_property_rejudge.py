"""
Property-based tests for dashboard/rejudge.py.

Property 15: Re-judge only modifies scores.llm_judge
Property 16: Batch re-score processes all runs regardless of individual failures

Validates: Requirements 12.5, 13.3
"""

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=30,
)

_score_value = st.one_of(st.none(), st.integers(min_value=1, max_value=5))

_turn = st.fixed_dictionaries({
    "turn": st.integers(min_value=1, max_value=20),
    "speaker": st.sampled_from(["interviewer", "subject"]),
    "text": st.text(min_size=1, max_size=100),
    "timestamp": st.just("2024-01-01T00:00:00+00:00"),
    "raw_prompt": st.just("prompt"),
})

_run_data = st.fixed_dictionaries({
    "run_id": _safe_text,
    "timestamp": st.just("2024-01-01T00:00:00+00:00"),
    "scenario_id": _safe_text,
    "subject_model": _safe_text,
    "interviewer_model": _safe_text,
    "model": _safe_text,
    "language": st.just("english"),
    "params": st.just({}),
    "conversation": st.lists(_turn, min_size=0, max_size=5),
    "metadata": st.fixed_dictionaries({
        "total_turns": st.integers(min_value=0, max_value=20),
        "stop_reason": st.sampled_from(["max_turns", "condition_met", "error", "probes_exhausted"]),
        "context_trims": st.integers(min_value=0, max_value=5),
    }),
    "scores": st.fixed_dictionaries({
        "llm_judge": st.one_of(
            st.none(),
            st.fixed_dictionaries({
                "judge_model": _safe_text,
                "eval_target": st.just("subject"),
                "scores": st.fixed_dictionaries({
                    "identity_consistency": _score_value,
                    "cultural_authenticity": _score_value,
                    "naturalness": _score_value,
                    "information_yield": _score_value,
                }),
                "raw_response": st.just("{}"),
            }),
        )
    }),
})

_judge_result = st.fixed_dictionaries({
    "judge_model": _safe_text,
    "eval_target": st.just("subject"),
    "scores": st.fixed_dictionaries({
        "identity_consistency": st.integers(min_value=1, max_value=5),
        "cultural_authenticity": st.integers(min_value=1, max_value=5),
        "naturalness": st.integers(min_value=1, max_value=5),
        "information_yield": st.integers(min_value=1, max_value=5),
    }),
    "raw_response": st.just('{"identity_consistency":3}'),
})


# ---------------------------------------------------------------------------
# Property 15: Re-judge only modifies scores.llm_judge
# ---------------------------------------------------------------------------

@given(run_data=_run_data, judge_result=_judge_result)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,), deadline=None)
def test_rejudge_run_only_modifies_scores_llm_judge(run_data, judge_result):
    """
    Property 15: Re-judge only modifies scores.llm_judge.

    After calling rejudge_run, every field in run_data other than
    run_data["scores"]["llm_judge"] is identical to its value before the call.

    Validates: Requirements 12.5
    """
    from dashboard.rejudge import rejudge_run

    original = copy.deepcopy(run_data)

    mock_model = MagicMock()
    mock_judge = MagicMock()
    mock_judge.evaluate.return_value = judge_result

    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = Path(tmp_dir)

        with patch("dashboard.rejudge._build_model", return_value=mock_model), \
             patch("evaluation.llm_judge.LLMJudge", return_value=mock_judge), \
             patch("dashboard.rejudge.save_run"):

            updated = rejudge_run(run_data, "openai:gpt-4o", "subject", logs_dir)

    # scores.llm_judge must be updated to the judge result
    assert updated["scores"]["llm_judge"] == judge_result, (
        "scores.llm_judge was not updated to the new judge result"
    )

    # Every other top-level field must be unchanged
    for key in original:
        if key == "scores":
            # Only llm_judge inside scores may change
            for score_key in original["scores"]:
                if score_key != "llm_judge":
                    assert updated["scores"][score_key] == original["scores"][score_key], (
                        f"scores.{score_key} was unexpectedly modified by rejudge_run"
                    )
        else:
            assert updated[key] == original[key], (
                f"Field '{key}' was unexpectedly modified by rejudge_run. "
                f"Before: {original[key]!r}, After: {updated[key]!r}"
            )


# ---------------------------------------------------------------------------
# Property 16: Batch re-score processes all runs regardless of individual failures
# ---------------------------------------------------------------------------

_run_ids = st.lists(
    _safe_text,
    min_size=1,
    max_size=10,
    unique=True,
)

_failure_mask = st.lists(st.booleans(), min_size=1, max_size=10)


@given(run_ids=_run_ids, failure_mask=_failure_mask)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_batch_rescore_processes_all_runs_regardless_of_failures(run_ids, failure_mask):
    """
    Property 16: Batch re-score processes all runs regardless of individual failures.

    For any list of run_ids where some rejudge_run calls raise exceptions,
    batch_rescore returns a result dict containing every run_id, with None for
    successes and an error string for failures.

    Validates: Requirements 13.3
    """
    from dashboard.rejudge import batch_rescore

    # Align failure_mask length to run_ids
    mask = (failure_mask * ((len(run_ids) // max(len(failure_mask), 1)) + 1))[:len(run_ids)]

    call_count = {"n": 0}

    def fake_rejudge(run_data, judge_model_str, eval_target, logs_dir):
        idx = call_count["n"]
        call_count["n"] += 1
        if mask[idx]:
            raise RuntimeError(f"Simulated failure for run {run_data['run_id']}")
        return run_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = Path(tmp_dir)

        # Write a minimal JSON file for each run_id so batch_rescore can load them
        base_log = {
            "run_id": "",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "scenario_id": "s",
            "subject_model": "m",
            "interviewer_model": "m",
            "model": "m",
            "language": "english",
            "params": {},
            "conversation": [],
            "metadata": {"total_turns": 0, "stop_reason": "max_turns", "context_trims": 0},
            "scores": {},
        }
        for rid in run_ids:
            log = dict(base_log)
            log["run_id"] = rid
            (logs_dir / f"{rid}.json").write_text(json.dumps(log), encoding="utf-8")

        with patch("dashboard.rejudge.rejudge_run", side_effect=fake_rejudge):
            results = batch_rescore(run_ids, "openai:gpt-4o", "subject", logs_dir)

    # Every run_id must appear in the result
    assert set(results.keys()) == set(run_ids), (
        f"batch_rescore result is missing run_ids. "
        f"Expected {set(run_ids)}, got {set(results.keys())}"
    )

    # Successes have None; failures have a non-empty error string
    for i, rid in enumerate(run_ids):
        if mask[i]:
            assert results[rid] is not None and isinstance(results[rid], str), (
                f"run_id {rid!r} was expected to fail but got result: {results[rid]!r}"
            )
        else:
            assert results[rid] is None, (
                f"run_id {rid!r} was expected to succeed but got error: {results[rid]!r}"
            )

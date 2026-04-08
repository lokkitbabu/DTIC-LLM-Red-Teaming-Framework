"""
Re-judge and batch re-score utilities for the analytics dashboard.
"""

import sys
import importlib
from pathlib import Path
from typing import Optional

from utils.logger import save_run


def _build_model(model_str: str):
    """Import build_model from main.py at call time to avoid circular imports."""
    if "main" not in sys.modules:
        spec = importlib.util.spec_from_file_location("main", Path("main.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["main"] = mod
    return sys.modules["main"].build_model(model_str)


def rejudge_run(
    run_data: dict,
    judge_model_str: str,
    eval_target: str,
    logs_dir: Path,
) -> dict:
    """
    Re-score an existing run with a new judge model.

    Builds the judge model, calls LLMJudge.evaluate(), updates
    run_data["scores"]["llm_judge"], persists the updated log, and returns
    the updated run_data.

    Only run_data["scores"]["llm_judge"] is modified; all other fields are
    left untouched.

    Raises:
        ValueError: if judge_model_str is unknown.
    """
    from evaluation.llm_judge import LLMJudge

    model = _build_model(judge_model_str)
    judge = LLMJudge(model, eval_target=eval_target)
    result = judge.evaluate(run_data)

    run_data["scores"]["llm_judge"] = result
    save_run(run_data, run_data["run_id"])

    return run_data


def batch_rescore(
    run_ids: list,
    judge_model_str: str,
    eval_target: str,
    logs_dir: Path,
) -> dict:
    """
    Re-score multiple runs sequentially.

    For each run_id, loads the run JSON, calls rejudge_run, and records the
    outcome.  Exceptions are caught per-run so a single failure does not abort
    the batch.

    Returns:
        dict mapping run_id -> None on success, run_id -> error_str on failure.
    """
    import json

    results: dict[str, Optional[str]] = {}

    for run_id in run_ids:
        try:
            run_file = logs_dir / f"{run_id}.json"
            with open(run_file, "r", encoding="utf-8") as f:
                run_data = json.load(f)

            rejudge_run(run_data, judge_model_str, eval_target, logs_dir)
            results[run_id] = None
        except Exception as exc:  # noqa: BLE001
            results[run_id] = str(exc)

    return results

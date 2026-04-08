"""
RunExecutorView: launch a new evaluation run from the dashboard UI.

Spawns a background threading.Thread that calls ConversationRunner.run(),
optionally LLMJudge.evaluate(), then save_run(). Turn summaries are streamed
into st.session_state["run_state"]["lines"] and polled via st.rerun().
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from typing_extensions import TypedDict


class RunState(TypedDict):
    running: bool
    lines: list[str]
    error: Optional[str]
    result_path: Optional[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model(model_str: str):
    """Import build_model from main.py at call time to avoid circular imports."""
    if "main" not in sys.modules:
        spec = importlib.util.spec_from_file_location("main", Path("main.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["main"] = mod
    return sys.modules["main"].build_model(model_str)


def _list_scenario_files(scenarios_dir: Path = Path("scenarios")) -> list[str]:
    """Return sorted list of scenario JSON filenames."""
    if not scenarios_dir.exists():
        return []
    return sorted(str(p) for p in scenarios_dir.glob("*.json"))


def _worker(
    scenario_path: str,
    subject_model_str: str,
    interviewer_model_str: str,
    judge_model_str: Optional[str],
    eval_target: str,
    max_turns: int,
    run_state: RunState,
) -> None:
    """
    Background worker: runs the conversation, optionally judges, then saves.

    Appends turn summaries to run_state["lines"] as they complete.
    Sets run_state["result_path"] on success or run_state["error"] on failure.
    Always sets run_state["running"] = False when done.
    """
    import json as _json

    # Ensure the project root is on sys.path so runner/evaluation/utils are importable
    _project_root = str(Path(__file__).resolve().parents[2])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from runner.conversation_runner import ConversationRunner
    from evaluation.llm_judge import LLMJudge
    from utils.logger import save_run
    from utils.scenario_loader import load_scenario

    try:
        scenario = load_scenario(scenario_path)

        subject_model = _build_model(subject_model_str)
        interviewer_model = _build_model(interviewer_model_str)

        runner = ConversationRunner(
            subject_model,
            interviewer_model,
            max_turns=max_turns,
        )

        run_state["lines"].append(f"Starting run: scenario={scenario.get('scenario_id')}, "
                                  f"subject={subject_model_str}, interviewer={interviewer_model_str}")

        # Monkey-patch print to capture turn summaries into lines
        original_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

        import builtins
        _original_print = builtins.print

        def _capturing_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            run_state["lines"].append(msg)
            _original_print(*args, **kwargs)

        builtins.print = _capturing_print
        try:
            run_data = runner.run(scenario)
        finally:
            builtins.print = _original_print

        run_state["lines"].append(f"Conversation complete — {run_data['metadata']['total_turns']} subject turn(s).")

        if judge_model_str:
            run_state["lines"].append(f"Running LLM judge ({judge_model_str})...")
            judge_model = _build_model(judge_model_str)
            judge = LLMJudge(judge_model, eval_target=eval_target)
            result = judge.evaluate(run_data)
            run_data["scores"]["llm_judge"] = result
            run_state["lines"].append(f"Judge scores: {result.get('scores', {})}")

        log_path = save_run(run_data)
        run_state["result_path"] = str(log_path)
        run_state["lines"].append(f"Run saved: {log_path}")

    except Exception as exc:  # noqa: BLE001
        run_state["error"] = str(exc)
    finally:
        run_state["running"] = False


# ---------------------------------------------------------------------------
# Public view
# ---------------------------------------------------------------------------

def render_run_executor() -> None:
    """
    Render the Run Scenario page.

    Lists scenario files, accepts model/config inputs, spawns a background
    thread on "Run", and polls for output until the thread completes.
    """
    st.subheader("Run Scenario")

    scenario_files = _list_scenario_files()

    if not scenario_files:
        st.warning("No scenario files found in `scenarios/`. Add a `.json` file to get started.")
        return

    # --- Input form ---
    scenario_path = st.selectbox(
        "Scenario",
        options=scenario_files,
        format_func=lambda p: Path(p).stem,
        key="run_exec_scenario",
    )

    col1, col2 = st.columns(2)
    with col1:
        subject_model = st.text_input(
            "Subject model",
            value="openai:gpt-4o",
            key="run_exec_subject_model",
        )
        interviewer_model = st.text_input(
            "Interviewer model",
            value="openai:gpt-4o",
            key="run_exec_interviewer_model",
        )
    with col2:
        judge_model = st.text_input(
            "Judge model (optional)",
            value="",
            key="run_exec_judge_model",
        )
        eval_target = st.selectbox(
            "Eval target",
            options=["subject", "interviewer"],
            key="run_exec_eval_target",
        )

    max_turns = st.number_input(
        "Max turns",
        min_value=1,
        max_value=200,
        value=40,
        step=1,
        key="run_exec_max_turns",
    )

    # --- Run state management ---
    run_state: Optional[RunState] = st.session_state.get("run_state")
    is_running = run_state is not None and run_state.get("running", False)

    if st.button("Run", disabled=is_running, key="run_exec_run_btn"):
        new_state: RunState = {
            "running": True,
            "lines": [],
            "error": None,
            "result_path": None,
        }
        st.session_state["run_state"] = new_state

        thread = threading.Thread(
            target=_worker,
            args=(
                scenario_path,
                subject_model,
                interviewer_model,
                judge_model.strip() or None,
                eval_target,
                int(max_turns),
                new_state,
            ),
            daemon=True,
        )
        thread.start()
        st.rerun()

    # --- Polling / output display ---
    run_state = st.session_state.get("run_state")
    if run_state is None:
        return

    output_placeholder = st.empty()

    if run_state.get("running"):
        # Display current lines
        lines = run_state.get("lines", [])
        if lines:
            output_placeholder.text("\n".join(lines))
        else:
            output_placeholder.info("Starting run…")

        time.sleep(0.5)
        st.rerun()
        return

    # --- Run finished ---
    lines = run_state.get("lines", [])
    if lines:
        output_placeholder.text("\n".join(lines))

    if run_state.get("error"):
        st.error(f"Run failed: {run_state['error']}")
    elif run_state.get("result_path"):
        # Clear DataLoader cache so the RunIndex reflects the new run
        st.cache_data.clear()
        st.success(f"Run complete. Log saved to: `{run_state['result_path']}`")

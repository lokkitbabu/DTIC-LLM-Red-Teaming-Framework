"""
RunExecutorView: launch single or batch evaluation runs from the dashboard.

Features:
  - Scenario selector with live description/objective preview
  - Model preset buttons for the 5 chosen models
  - Prompt format A/B selector
  - Single run or batch (all formats × N repeats)
  - Live turn-by-turn progress with animated status
  - Supabase sync status per run
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

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "Llama 3.3 70B Turbo": "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "DeepSeek V3.2":     "together:deepseek-ai/DeepSeek-V3.1",
    "Mistral Large 3":   "mistral:mistral-large-latest",
    "Claude Sonnet 4.6": "anthropic:claude-sonnet-4-6",
    "GPT-5.4":           "openai:gpt-5.4",
    "Grok 4.1 Fast":     "grok:grok-4-1-fast-reasoning",
}

INTERVIEWER_PRESETS = {
    "GPT-4.1 (recommended)": "openai:gpt-4.1-2025-04-14",
    "Llama 3.3 70B Turbo":        "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

JUDGE_PRESETS = {
    "GPT-5.4": "openai:gpt-5.4",
    "Claude Sonnet 4.6": "anthropic:claude-sonnet-4-6",
    "None": "",
}

PROMPT_FORMAT_DESCRIPTIONS = {
    "flat": "Single freeform system prompt (baseline)",
    "hierarchical": "[PRIORITY 1/2/3] layered instructions",
    "xml": "<identity>, <objective>, <constraints> tagged structure",
}


class RunState(TypedDict):
    running: bool
    lines: list[str]
    error: Optional[str]
    result_path: Optional[str]
    run_id: Optional[str]
    turn_current: int
    turn_total: int
    phase: str  # "conversation" | "judging" | "saving" | "syncing" | "done"
    synced: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model(model_str: str):
    _project_root = str(Path(__file__).resolve().parents[2])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    if "main" not in sys.modules:
        spec = importlib.util.spec_from_file_location("main", Path(_project_root) / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["main"] = mod
    return sys.modules["main"].build_model(model_str)


def _list_scenario_files() -> list[Path]:
    d = Path("scenarios")
    if not d.exists():
        return []
    return sorted(d.glob("*.json"))


def _load_scenario_meta(path: Path) -> dict:
    import json
    try:
        with open(path) as f:
            d = json.load(f)
        return {
            "scenario_id": d.get("scenario_id", path.stem),
            "description": d.get("description", d.get("objective", "")),
            "objective": d.get("objective", ""),
            "identity_name": d.get("identity", {}).get("name", "—"),
            "interviewer_name": d.get("interviewer", {}).get("name", "—"),
            "language": d.get("language", "english"),
            "turns": d.get("params", {}).get("max_tokens", "—"),
        }
    except Exception:
        return {"scenario_id": path.stem, "description": "", "objective": ""}


def _worker(
    scenario_path: str,
    subject_model_str: str,
    interviewer_model_str: str,
    judge_model_str: Optional[str],
    prompt_format: str,
    max_turns: int,
    run_state: RunState,
) -> None:
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
            subject_model, interviewer_model,
            max_turns=max_turns,
            prompt_format=prompt_format,
        )

        run_state["phase"] = "conversation"
        run_state["turn_total"] = max_turns
        run_state["lines"].append(
            f"▶ {scenario.get('scenario_id')}  |  subject: {subject_model_str.split(':')[-1]}  |  format: {prompt_format}"
        )

        import builtins
        _orig = builtins.print

        def _capture(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            run_state["lines"].append(msg)
            # Parse turn progress from runner output
            if "turn " in msg and "subject" in msg:
                try:
                    part = msg.split("turn ")[1].split("/")
                    run_state["turn_current"] = int(part[0].strip())
                except Exception:
                    pass
            _orig(*args, **kwargs)

        builtins.print = _capture
        try:
            run_data = runner.run(scenario)
        finally:
            builtins.print = _orig

        total = run_data["metadata"]["total_turns"]
        run_state["turn_current"] = total
        run_state["lines"].append(f"✓ Conversation complete — {total} turns")

        if judge_model_str:
            run_state["phase"] = "judging"
            run_state["lines"].append(f"⚖ Judging with {judge_model_str.split(':')[-1]}…")
            judge_model = _build_model(judge_model_str)
            from evaluation.llm_judge import LLMJudge
            judge = LLMJudge(judge_model, eval_target="subject")
            result = judge.evaluate(run_data)
            run_data["scores"]["llm_judge"] = result
            scores = {k: v for k, v in (result.get("scores") or result).items()
                      if k != "reasoning" and v is not None}
            run_state["lines"].append(f"  Scores: {scores}")

        run_state["phase"] = "saving"
        log_path = save_run(run_data)
        run_state["result_path"] = str(log_path)
        run_state["run_id"] = run_data["run_id"]
        run_state["lines"].append(f"💾 Saved: {log_path}")

        run_state["phase"] = "syncing"
        try:
            from dashboard.supabase_store import get_store
            store = get_store()
            if store.available:
                store.save_run(run_data)
                run_state["synced"] = True
                run_state["lines"].append("☁ Synced to Supabase — visible in dashboard")
            else:
                run_state["lines"].append("  (Supabase not configured — local only)")
        except Exception as e:
            run_state["lines"].append(f"  (Supabase sync failed: {e})")

        run_state["phase"] = "done"

    except Exception as exc:
        run_state["error"] = str(exc)
        run_state["lines"].append(f"✗ Error: {exc}")
    finally:
        run_state["running"] = False


# ---------------------------------------------------------------------------
# Public view
# ---------------------------------------------------------------------------

def render_run_executor() -> None:
    st.subheader("Run Scenario")

    scenario_files = _list_scenario_files()
    if not scenario_files:
        st.warning("No scenario files found in `scenarios/`. Use the **Scenarios** page to create one.")
        return

    # ── Scenario selector ────────────────────────────────────────────────────
    st.markdown("#### 1 — Select Scenario")
    scenario_names = {p.stem: p for p in scenario_files}
    selected_name = st.selectbox(
        "Scenario", list(scenario_names.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
        key="re_scenario",
    )
    scenario_path = scenario_names[selected_name]
    meta = _load_scenario_meta(scenario_path)

    with st.expander("Scenario details", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Subject", meta.get("identity_name", "—"))
        col_b.metric("Interviewer", meta.get("interviewer_name", "—"))
        col_c.metric("Language", meta.get("language", "english").title())
        if meta.get("description"):
            st.caption(meta["description"][:300] + ("…" if len(meta.get("description","")) > 300 else ""))

    # ── Model selection ──────────────────────────────────────────────────────
    st.markdown("#### 2 — Subject Model")
    st.caption("Click a preset or type a custom model string")

    cols = st.columns(len(MODEL_PRESETS))
    for col, (label, value) in zip(cols, MODEL_PRESETS.items()):
        with col:
            if st.button(label, key=f"preset_{label}", use_container_width=True):
                st.session_state["re_subject_model"] = value

    subject_model = st.text_input(
        "Subject model",
        value=st.session_state.get("re_subject_model", "together:meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        key="re_subject_model",
        label_visibility="collapsed",
    )

    # ── Interviewer + Judge ─────────────────────────────────────────────────
    st.markdown("#### 3 — Interviewer & Judge")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Interviewer model")
        interviewer_cols = st.columns(len(INTERVIEWER_PRESETS))
        for col, (label, value) in zip(interviewer_cols, INTERVIEWER_PRESETS.items()):
            with col:
                if st.button(label, key=f"iv_{label}", use_container_width=True):
                    st.session_state["re_interviewer_model"] = value
        interviewer_model = st.text_input(
            "Interviewer",
            value=st.session_state.get("re_interviewer_model",
                                       "openai:gpt-4.1-2025-04-14"),
            key="re_interviewer_model",
            label_visibility="collapsed",
        )
    with col2:
        st.caption("Judge model")
        judge_cols = st.columns(len(JUDGE_PRESETS))
        for col, (label, value) in zip(judge_cols, JUDGE_PRESETS.items()):
            with col:
                if st.button(label, key=f"jdg_{label}", use_container_width=True):
                    st.session_state["re_judge_model"] = value
        judge_model = st.text_input(
            "Judge",
            value=st.session_state.get("re_judge_model", "openai:gpt-4o"),
            key="re_judge_model",
            label_visibility="collapsed",
        )

    # ── Run configuration ────────────────────────────────────────────────────
    st.markdown("#### 4 — Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

    with cfg_col1:
        prompt_format = st.selectbox(
            "Prompt format",
            options=["flat", "hierarchical", "xml"],
            format_func=lambda x: f"{x} — {PROMPT_FORMAT_DESCRIPTIONS[x]}",
            key="re_prompt_format",
        )
    with cfg_col2:
        max_turns = st.number_input("Max turns", 5, 200, 40, 5, key="re_max_turns")
    with cfg_col3:
        run_mode = st.radio("Mode", ["Single run", "All 3 formats × 3 runs"], key="re_mode")

    # ── Launch button ────────────────────────────────────────────────────────
    st.markdown("---")
    run_state: Optional[RunState] = st.session_state.get("run_state")
    is_running = run_state is not None and run_state.get("running", False)

    if run_mode == "All 3 formats × 3 runs":
        total_planned = 9
        btn_label = f"🚀 Launch batch (9 runs — current model × all formats)"
    else:
        total_planned = 1
        btn_label = "▶ Run"

    launch = st.button(btn_label, type="primary", disabled=is_running, key="re_launch")

    if launch:
        if run_mode == "Single run":
            _start_run(scenario_path, subject_model, interviewer_model,
                       judge_model.strip() or None, prompt_format, int(max_turns))
        else:
            _start_batch(scenario_path, subject_model, interviewer_model,
                         judge_model.strip() or None, int(max_turns))
        st.rerun()

    # ── Live status display ─────────────────────────────────────────────────
    # Use st.fragment with run_every so this section auto-polls every second
    # without requiring user interaction — the only reliable way on Streamlit Cloud
    _render_live_status()


@st.fragment(run_every=1)
def _render_live_status() -> None:
    """Auto-refreshing fragment — reruns every 1s while a run is active."""
    run_state = st.session_state.get("run_state")
    if run_state is None:
        return
    _render_status(run_state)


# ---------------------------------------------------------------------------
# Run launchers
# ---------------------------------------------------------------------------

def _start_run(scenario_path, subject, interviewer, judge, fmt, max_turns):
    state: RunState = {
        "running": True, "lines": [], "error": None,
        "result_path": None, "run_id": None,
        "turn_current": 0, "turn_total": max_turns,
        "phase": "starting", "synced": False,
    }
    st.session_state["run_state"] = state
    t = threading.Thread(
        target=_worker,
        args=(str(scenario_path), subject, interviewer, judge, fmt, max_turns, state),
        daemon=True,
    )
    t.start()


def _start_batch(scenario_path, subject, interviewer, judge, max_turns):
    """Run all 3 formats × 3 repeats sequentially in a background thread."""
    state: RunState = {
        "running": True, "lines": [], "error": None,
        "result_path": None, "run_id": None,
        "turn_current": 0, "turn_total": max_turns,
        "phase": "starting", "synced": False,
    }
    st.session_state["run_state"] = state

    def _batch_worker():
        formats = ["flat", "hierarchical", "xml"]
        total = len(formats) * 3
        completed = 0
        for fmt in formats:
            for rep in range(1, 4):
                state["lines"].append(f"\n── Run {completed + 1}/{total}  format={fmt}  rep={rep} ──")
                _worker(str(scenario_path), subject, interviewer, judge, fmt, max_turns, state)
                if state.get("error"):
                    return
                completed += 1
                state["error"] = None  # reset for next run
                state["result_path"] = None
                state["run_id"] = None
                state["running"] = True  # keep running flag true between runs
        state["running"] = False
        state["phase"] = "done"
        state["lines"].append(f"\n✓ Batch complete — {completed} runs")

    t = threading.Thread(target=_batch_worker, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Status renderer
# ---------------------------------------------------------------------------

def _render_status(run_state: RunState) -> None:
    phase = run_state.get("phase", "")
    is_running = run_state.get("running", False)
    error = run_state.get("error")

    # Phase indicator
    phase_icons = {
        "starting": "⏳", "conversation": "💬", "judging": "⚖",
        "saving": "💾", "syncing": "☁", "done": "✅",
    }
    if is_running or phase not in ("done", ""):
        icon = phase_icons.get(phase, "⏳")
        phase_label = phase.title() if phase else "Starting"
        if is_running:
            st.markdown(f"**{icon} {phase_label}…**")
        elif error:
            st.error(f"Run failed: {error}")
        else:
            synced = run_state.get("synced", False)
            st.success(f"✅ Complete {'— synced to Supabase ☁' if synced else '— local only'}")

    # Turn progress bar (only during conversation phase)
    if phase == "conversation" and is_running:
        current = run_state.get("turn_current", 0)
        total = run_state.get("turn_total", 1)
        st.progress(min(current / total, 1.0),
                    text=f"Turn {current} / {total}")

    # Log output
    lines = run_state.get("lines", [])
    if lines:
        with st.expander("Run log", expanded=is_running):
            # Show last 40 lines, most recent at bottom
            display = lines[-40:]
            st.code("\n".join(display), language=None)

    # Result link
    if not is_running and run_state.get("run_id"):
        run_id = run_state["run_id"]
        if st.button("→ View run in detail", key="re_view_result"):
            st.session_state["run_id"] = run_id
            st.session_state["page"] = "Run Detail"
            st.cache_data.clear()
            st.rerun()

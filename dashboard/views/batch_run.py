"""
BatchRunView: launch multi-scenario × all-models batch from a single config.

One place to configure: interviewer, eval prompt, prompt format, turns, repeats.
Select which scenarios and models to include. Launch. Watch live progress.
All runs use dual-judge strict grading and sync to Supabase automatically.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Config constants ──────────────────────────────────────────────────────────

_ALL_MODELS = {
    "Llama 3.3 70B Turbo": "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "DeepSeek V3.1":       "together:deepseek-ai/DeepSeek-V3.1",
    "Mistral Large 3":     "mistral:mistral-large-latest",
    "Claude Sonnet 4.6":   "anthropic:claude-sonnet-4-6",
    "GPT-5.4":             "openai:gpt-5.4",
    "Grok 4.20":           "grok:grok-4.20-0309-non-reasoning",
}

_INTERVIEWER_PRESETS = {
    "Grok 4.1 Fast (recommended)": "grok:grok-4-1-fast-reasoning",
    "GPT-4.1":                     "openai:gpt-4.1-2025-04-14",
}

_JUDGE_A = "openai:gpt-5.4"
_JUDGE_B = "anthropic:claude-sonnet-4-6"

_SCENARIOS_DIR = Path("scenarios")


def render_batch_run_view(logs_dir: Path = Path("logs")) -> None:
    st.subheader("🧪 Run Experiments")
    st.caption(
        "Run all selected scenarios × all selected models with one config. "
        "Each combo is judged by GPT-5.4 + Claude Sonnet (averaged, strict prompt). "
        "Results sync to Supabase automatically."
    )

    # ── Scenario selector ─────────────────────────────────────────────────────
    st.markdown("#### 1 — Scenarios")
    scenario_files = sorted(_SCENARIOS_DIR.glob("*.json")) if _SCENARIOS_DIR.exists() else []
    if not scenario_files:
        st.warning("No scenarios found.")
        return

    sel_all_s, desel_all_s, _ = st.columns([1, 1, 5])
    if sel_all_s.button("Select all", key="br_sel_all_s"):
        for p in scenario_files:
            st.session_state[f"br_s_{p.stem}"] = True
    if desel_all_s.button("Deselect all", key="br_desel_all_s"):
        for p in scenario_files:
            st.session_state[f"br_s_{p.stem}"] = False

    selected_scenarios: list[Path] = []
    import json as _json
    for p in scenario_files:
        try:
            d = _json.loads(p.read_text())
        except Exception:
            continue
        detail = d.get("detail_level", "")
        badge = f" `{detail}`" if detail else ""
        desc = d.get("description", "")[:80]
        checked = st.checkbox(
            f"**{d.get('scenario_id', p.stem)}**{badge}  \n{desc}",
            key=f"br_s_{p.stem}",
        )
        if checked:
            selected_scenarios.append(p)

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown("#### 2 — Models")
    sel_all_m, desel_all_m, _ = st.columns([1, 1, 5])
    if sel_all_m.button("Select all", key="br_sel_all_m"):
        for k in _ALL_MODELS:
            st.session_state[f"br_m_{k}"] = True
    if desel_all_m.button("Deselect all", key="br_desel_all_m"):
        for k in _ALL_MODELS:
            st.session_state[f"br_m_{k}"] = False

    selected_models: list[str] = []
    model_cols = st.columns(3)
    for i, (label, model_str) in enumerate(_ALL_MODELS.items()):
        checked = model_cols[i % 3].checkbox(label, key=f"br_m_{label}", value=True)
        if checked:
            selected_models.append(model_str)

    # ── Shared config ─────────────────────────────────────────────────────────
    st.markdown("#### 3 — Shared Config")
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)

    with cfg1:
        st.markdown("**Interviewer**")
        for label, val in _INTERVIEWER_PRESETS.items():
            if st.button(label.split(" ")[0], key=f"br_iv_{label}"):
                st.session_state["br_interviewer"] = val
        interviewer = st.text_input(
            "Interviewer",
            value=st.session_state.get("br_interviewer", list(_INTERVIEWER_PRESETS.values())[0]),
            key="br_interviewer_input",
            label_visibility="collapsed",
        )

    with cfg2:
        prompt_format = st.selectbox(
            "Prompt format",
            ["flat", "hierarchical", "xml"],
            key="br_format",
            help="flat=baseline, hierarchical=priority layers, xml=structured tags",
        )

    with cfg3:
        max_turns = st.number_input("Max turns", 5, 200, 40, 5, key="br_max_turns")
        runs_per_combo = st.number_input("Runs per combo", 1, 10, 3, 1, key="br_runs_per_combo")

    with cfg4:
        eval_prompt = st.radio(
            "Eval prompt",
            ["strict", "standard", "lenient"],
            key="br_eval_prompt",
            help="strict = AI-language = 1, word repetition penalised",
        )
        dual_judge = st.checkbox("Dual judge (avg GPT-5.4 + Claude)", value=True, key="br_dual_judge")
        max_workers = st.slider(
            "Parallel runs",
            min_value=1, max_value=6, value=3, step=1,
            key="br_max_workers",
            help="How many runs execute simultaneously. Higher = faster but more API rate-limit risk. 3 is a safe default.",
        )

    # ── Plan summary ──────────────────────────────────────────────────────────
    n_scenarios = len(selected_scenarios)
    n_models = len(selected_models)
    n_runs = n_scenarios * n_models * int(runs_per_combo)

    st.markdown("---")
    if n_runs > 0:
        judge_label = "GPT-5.4 + Claude Sonnet 4.6 (avg)" if dual_judge else "GPT-5.4"
        st.info(
            f"**{n_runs} total runs** — "
            f"{n_scenarios} scenario(s) × {n_models} model(s) × {int(runs_per_combo)} repeat(s)  \n"
            f"Format: `{prompt_format}` · Turns: {int(max_turns)} · "
            f"Eval: `{eval_prompt}` · Judge: {judge_label}  \n"
            f"⚡ **{max_workers} parallel** (est. {max(1, n_runs // max_workers)} batch(es) of {max_workers})"
        )
    else:
        st.warning("Select at least one scenario and one model.")

    # ── Launch ────────────────────────────────────────────────────────────────
    br_state = st.session_state.get("br_state")
    is_running = br_state is not None and br_state.get("running", False)

    # Stop button (only when running)
    col_launch, col_stop = st.columns([3, 1])
    with col_launch:
        launch = st.button(
            f"🚀 Launch {n_runs} run(s)",
            type="primary",
            disabled=is_running or n_runs == 0,
            key="br_launch",
        )
    with col_stop:
        stop_pressed = st.button(
            "⏹ Stop",
            disabled=not is_running,
            key="br_stop",
            type="secondary",
        )

    if stop_pressed and is_running:
        br_state = st.session_state.get("br_state", {})
        stop_evt = br_state.get("stop_event")
        if stop_evt is not None:
            stop_evt.set()
            br_state["lines"].append("⏹ Stop requested — finishing current turns…")
        st.rerun()

    if launch and n_runs > 0:
        _start_batch(
            scenarios=selected_scenarios,
            models=selected_models,
            interviewer=interviewer,
            prompt_format=prompt_format,
            max_turns=int(max_turns),
            runs_per_combo=int(runs_per_combo),
            eval_prompt=eval_prompt,
            dual_judge=dual_judge,
            logs_dir=logs_dir,
            max_workers=int(max_workers),
        )
        st.rerun()

    # ── Live progress ─────────────────────────────────────────────────────────
    _render_progress()


@st.fragment(run_every=1)
def _render_progress() -> None:
    br_state = st.session_state.get("br_state")
    if not br_state:
        return

    completed = br_state.get("completed", 0)
    total = br_state.get("total", 1)
    is_running = br_state.get("running", False)
    current = br_state.get("current", "")
    lines = br_state.get("lines", [])
    errors = br_state.get("errors", [])

    if is_running:
        active_list = br_state.get("active", [])
        active_str = ", ".join(active_list[:3]) + ("…" if len(active_list) > 3 else "")
        st.progress(completed / total,
                    text=f"▶ {completed}/{total} complete  |  running: {active_str or 'starting…'}")
    elif br_state.get("stop_event") and br_state["stop_event"].is_set() and not is_running:
        st.warning(f"⏹ Stopped after {completed}/{total} run(s). {len(errors)} error(s).")
    elif errors:
        st.error(f"Completed with {len(errors)} error(s): see log below")
    else:
        st.success(f"✅ All {completed} run(s) complete — synced to Supabase")

    if lines:
        with st.expander("Run log", expanded=is_running):
            st.code("\n".join(lines[-60:]), language=None)

    if errors and not is_running:
        with st.expander(f"⚠ {len(errors)} errors"):
            for e in errors:
                st.error(e)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _start_batch(
    scenarios: list[Path],
    models: list[str],
    interviewer: str,
    prompt_format: str,
    max_turns: int,
    runs_per_combo: int,
    eval_prompt: str,
    dual_judge: bool,
    logs_dir: Path,
    max_workers: int = 3,
) -> None:
    # Build full combo list: scenario × model × repeat
    combos = [
        (s, m, r)
        for s in scenarios
        for m in models
        for r in range(1, runs_per_combo + 1)
    ]
    import threading as _threading
    stop_event = _threading.Event()
    state = {
        "running": True,
        "completed": 0,
        "total": len(combos),
        "active": [],
        "lines": [],
        "errors": [],
        "stop_event": stop_event,
    }
    st.session_state["br_state"] = state

    t = threading.Thread(
        target=_parallel_worker,
        args=(combos, interviewer, prompt_format, max_turns,
              eval_prompt, dual_judge, logs_dir, state, max_workers, stop_event),
        daemon=True,
    )
    t.start()


def _parallel_worker(
    combos: list,
    interviewer_str: str,
    prompt_format: str,
    max_turns: int,
    eval_prompt: str,
    dual_judge: bool,
    logs_dir: Path,
    state: dict,
    max_workers: int = 3,
    stop_event=None,
) -> None:
    """
    Outer coordinator: spins up a ThreadPoolExecutor and submits one
    _run_one_combo() call per combo. Progress is tracked via shared state.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from main import build_model
    from evaluation.llm_judge import LLMJudge
    import concurrent.futures
    import threading

    # Pre-build shared resources once
    try:
        interviewer_model = build_model(interviewer_str)
        judge_a = build_model(_JUDGE_A)
        judge_b = build_model(_JUDGE_B) if dual_judge else None
    except Exception as e:
        state["errors"].append(f"Failed to init shared models: {e}")
        state["running"] = False
        return

    lock = threading.Lock()

    def _run_one(combo):
        scenario_path, model_str, rep = combo
        scenario_stem = scenario_path.stem
        model_short = model_str.split(":")[-1].split("/")[-1][:20]
        label = f"{scenario_stem} × {model_short} rep{rep}"

        # Skip if already stopped before we even start
        if stop_event is not None and stop_event.is_set():
            with lock:
                state["lines"].append(f"[{label}] skipped (stop requested)")
                state["completed"] += 1
            return

        with lock:
            state["active"].append(label)

        try:
            _run_one_combo(
                scenario_path=scenario_path,
                model_str=model_str,
                rep=rep,
                interviewer_model=interviewer_model,
                judge_a=judge_a,
                judge_b=judge_b,
                prompt_format=prompt_format,
                max_turns=max_turns,
                eval_prompt=eval_prompt,
                dual_judge=dual_judge,
                logs_dir=logs_dir,
                state=state,
                lock=lock,
                label=label,
                stop_event=stop_event,
            )
        finally:
            with lock:
                state["completed"] += 1
                if label in state["active"]:
                    state["active"].remove(label)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one, combo) for combo in combos]
        concurrent.futures.wait(futures)

    state["running"] = False
    state["active"] = []
    with lock:
        state["lines"].append(
            f"\n✅ Done — {state['completed']}/{state['total']} runs, "
            f"{len(state['errors'])} error(s)"
        )

    try:
        import streamlit as _st
        _st.cache_data.clear()
    except Exception:
        pass


def _run_one_combo(
    scenario_path: Path,
    model_str: str,
    rep: int,
    interviewer_model,
    judge_a,
    judge_b,
    prompt_format: str,
    max_turns: int,
    eval_prompt: str,
    dual_judge: bool,
    logs_dir: Path,
    state: dict,
    lock,
    label: str,
    stop_event=None,
) -> None:
    """Run a single scenario × model combo, score it, and save."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from main import build_model
    from runner.conversation_runner import ConversationRunner
    from evaluation.llm_judge import LLMJudge
    from utils.logger import save_run
    from utils.scenario_loader import load_scenario

    def _log(msg: str):
        with lock:
            state["lines"].append(f"[{label}] {msg}")

    try:
        scenario = load_scenario(str(scenario_path))
        subject_model = build_model(model_str)

        runner = ConversationRunner(
            subject_model=subject_model,
            interviewer_model=interviewer_model,
            max_turns=max_turns,
            prompt_format=prompt_format,
        )

        _log("▶ starting conversation…")

        # Suppress per-turn print noise during parallel runs — too interleaved
        import builtins, io
        captured = io.StringIO()
        _orig = builtins.print
        def _silent(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            captured.write(msg + "\n")
            # Only surface turn progress
            if "turn" in msg and ("subject done" in msg or "interviewer done" in msg):
                _log(msg.strip())
        builtins.print = _silent
        try:
            run_data = runner.run(scenario, stop_event=stop_event)
        finally:
            builtins.print = _orig

        turns = run_data["metadata"]["total_turns"]
        _log(f"✓ {turns} turns")

        # Judge
        judge_models = [judge_a, judge_b] if dual_judge and judge_b else [judge_a]
        judge = LLMJudge(judge_models, eval_target="subject", prompt_name=eval_prompt)
        _log("⚖ judging…")
        result = judge.evaluate(run_data)
        run_data["scores"]["llm_judge"] = result

        scores = result.get("scores", {})
        total_score = sum(
            float(scores.get(m) or 0)
            for m in ["identity_consistency","cultural_authenticity","naturalness","information_yield"]
            if scores.get(m) is not None
        )
        _log(
            f"📊 IC={scores.get('identity_consistency','?')} "
            f"CA={scores.get('cultural_authenticity','?')} "
            f"N={scores.get('naturalness','?')} "
            f"IY={scores.get('information_yield','?')} "
            f"→ {total_score:.1f}/20"
        )

        # Save
        save_run(run_data)
        try:
            from dashboard.supabase_store import get_store
            store = get_store()
            if store.available:
                store.save_run(run_data)
                _log("☁ synced")
        except Exception:
            pass

    except Exception as e:
        _log(f"✗ FAILED: {str(e)[:100]}")
        with lock:
            state["errors"].append(f"{label}: {str(e)[:100]}")

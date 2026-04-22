"""
LiveChatView: human-in-the-loop chat with an LLM playing a scenario persona.

The human acts as the interviewer. The LLM responds in-character as the subject.
Conversations can be saved as run logs for scoring alongside automated runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st

_MODEL_PRESETS = {
    "Llama 3.3 70B Turbo": "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "DeepSeek V3.1":       "together:deepseek-ai/DeepSeek-V3.1",
    "Mistral Large 3":     "mistral:mistral-large-latest",
    "Claude Sonnet 4.6":   "anthropic:claude-sonnet-4-6",
    "GPT-5.4":             "openai:gpt-5.4",
    "Grok 4.20":           "grok:grok-4.20-0309-non-reasoning",
}

_SCENARIOS_DIR = Path("scenarios")


def render_live_chat_view(logs_dir: Path = Path("logs")) -> None:
    st.subheader("🧑‍💻 Live Chat")
    st.caption(
        "You play the interviewer. The LLM responds as the scenario persona. "
        "Use this to manually probe a model before running automated batches, "
        "or to verify a persona feels realistic."
    )

    # ── Session state init ─────────────────────────────────────────────────
    if "lc_messages" not in st.session_state:
        st.session_state["lc_messages"] = []      # {role, text, ts}
    if "lc_active" not in st.session_state:
        st.session_state["lc_active"] = False      # True = session running
    if "lc_scenario" not in st.session_state:
        st.session_state["lc_scenario"] = None
    if "lc_model_str" not in st.session_state:
        st.session_state["lc_model_str"] = list(_MODEL_PRESETS.values())[0]

    msgs = st.session_state["lc_messages"]
    active = st.session_state["lc_active"]

    # ── Setup panel (shown when not yet started) ───────────────────────────
    if not active:
        _render_setup(logs_dir)
        return

    # ── Active chat ────────────────────────────────────────────────────────
    scenario = st.session_state["lc_scenario"]
    model_str = st.session_state["lc_model_str"]
    persona_name = scenario.get("identity", {}).get("name", "Subject")
    short_model = model_str.split(":")[-1].split("/")[-1][:22]

    # Header bar
    col_info, col_end = st.columns([5, 1])
    with col_info:
        st.markdown(
            f"**{persona_name}** via `{short_model}` · "
            f"{len(msgs)} turn(s) · "
            f"scenario: `{scenario.get('scenario_id', '?')}`"
        )
    with col_end:
        if st.button("⏹ End session", key="lc_end"):
            _end_session(logs_dir)
            st.rerun()

    st.markdown("---")

    # Conversation display
    _render_chat_bubbles(msgs, persona_name)

    # Input box
    st.markdown("---")
    col_input, col_send = st.columns([6, 1])
    with col_input:
        human_input = st.text_area(
            "Your message",
            key="lc_input",
            height=80,
            placeholder=f"Type your message to {persona_name}… (Shift+Enter for newline)",
            label_visibility="collapsed",
        )
    with col_send:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("Send ➤", type="primary", key="lc_send", use_container_width=True)

    if send and human_input.strip():
        _handle_send(human_input.strip(), scenario, model_str)
        st.rerun()

    # Quick action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    qa_cols = st.columns(4)
    quick_prompts = [
        "Tell me more about yourself.",
        "How did you end up here?",
        "What do you think about the system?",
        "Are you who you say you are?",
    ]
    for col, prompt in zip(qa_cols, quick_prompts):
        if col.button(f'"{prompt[:28]}…"', key=f"lc_quick_{prompt[:10]}"):
            _handle_send(prompt, scenario, model_str)
            st.rerun()

    # Save button (always visible during session)
    if msgs:
        st.markdown("---")
        col_save, col_clear = st.columns([2, 1])
        with col_save:
            if st.button("💾 Save as run log", key="lc_save"):
                path = _save_run(msgs, scenario, model_str, logs_dir)
                st.success(f"Saved: `{path}`")
                st.cache_data.clear()
        with col_clear:
            if st.button("🗑 Clear messages", key="lc_clear"):
                st.session_state["lc_messages"] = []
                st.rerun()


# ---------------------------------------------------------------------------
# Setup panel
# ---------------------------------------------------------------------------

def _render_setup(logs_dir: Path) -> None:
    st.markdown("#### 1 — Pick scenario")
    scenario_files = sorted(_SCENARIOS_DIR.glob("*.json")) if _SCENARIOS_DIR.exists() else []
    if not scenario_files:
        st.warning("No scenarios found in `scenarios/`.")
        return

    scenario_map = {p.stem: p for p in scenario_files}
    selected_stem = st.selectbox(
        "Scenario",
        list(scenario_map.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
        key="lc_scenario_pick",
    )
    scenario_path = scenario_map[selected_stem]
    scenario = json.loads(scenario_path.read_text())

    identity = scenario.get("identity", {})
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Persona", identity.get("name", "—"))
    col_b.metric("Language", scenario.get("language", "english").title())
    col_c.metric("Format", scenario.get("format", "—"))

    with st.expander("Persona description", expanded=False):
        st.markdown(f"**Background:** {identity.get('background', '—')}")
        st.markdown(f"**Persona:** {identity.get('persona', '—')}")
        st.markdown(f"**Language style:** {identity.get('language_style', '—')}")
        traits = identity.get("personality_traits", [])
        if traits:
            st.markdown("**Traits:** " + " · ".join(traits[:5]))

    st.markdown("#### 2 — Pick subject model")
    preset_cols = st.columns(len(_MODEL_PRESETS))
    for col, (label, value) in zip(preset_cols, _MODEL_PRESETS.items()):
        if col.button(label, key=f"lc_preset_{label}", use_container_width=True):
            st.session_state["lc_model_default"] = value
            st.rerun()

    model_str = st.text_input(
        "Model string",
        value=st.session_state.get("lc_model_default", list(_MODEL_PRESETS.values())[0]),
        key="lc_model_input",
        label_visibility="collapsed",
    )

    st.markdown("#### 3 — Prompt format")
    prompt_format = st.radio(
        "Format",
        ["flat", "hierarchical", "xml"],
        horizontal=True,
        key="lc_format",
    )

    st.markdown("---")
    if st.button("▶ Start session", type="primary", key="lc_start"):
        st.session_state["lc_scenario"] = scenario
        st.session_state["lc_model_str"] = model_str
        st.session_state["lc_format"] = prompt_format
        st.session_state["lc_messages"] = []
        st.session_state["lc_active"] = True

        # Auto-inject opening message from scenario if present
        opening = scenario.get("opening_message", "")
        if opening:
            ts = _ts()
            st.session_state["lc_messages"].append(
                {"role": "human", "text": opening, "ts": ts}
            )
            # Get subject's response to the opening
            response = _call_model(
                messages=st.session_state["lc_messages"],
                scenario=scenario,
                model_str=model_str,
                prompt_format=prompt_format,
            )
            if response:
                st.session_state["lc_messages"].append(
                    {"role": "subject", "text": response, "ts": _ts()}
                )

        st.rerun()


# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------

def _handle_send(human_text: str, scenario: dict, model_str: str) -> None:
    msgs = st.session_state["lc_messages"]
    msgs.append({"role": "human", "text": human_text, "ts": _ts()})

    with st.spinner("Thinking…"):
        response = _call_model(
            messages=msgs,
            scenario=scenario,
            model_str=model_str,
            prompt_format=st.session_state.get("lc_format", "flat"),
        )

    if response:
        msgs.append({"role": "subject", "text": response, "ts": _ts()})
    else:
        msgs.append({"role": "subject", "text": "[No response — model returned empty]", "ts": _ts()})


def _call_model(
    messages: list[dict],
    scenario: dict,
    model_str: str,
    prompt_format: str = "flat",
) -> Optional[str]:
    """Build prompt and call the subject model."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    try:
        from main import build_model
        from runner.conversation_runner import ConversationRunner

        subject_model = build_model(model_str)
        # Use ConversationRunner just for prompt building
        runner = ConversationRunner(
            subject_model=subject_model,
            interviewer_model=subject_model,  # unused for prompt building
            max_turns=999,
            prompt_format=prompt_format,
        )

        # Convert lc_messages to runner conversation format
        history = []
        for i, m in enumerate(messages[:-1]):  # exclude last (just-added human msg)
            speaker = "interviewer" if m["role"] == "human" else "subject"
            history.append({
                "turn": i + 1,
                "speaker": speaker,
                "text": m["text"],
                "timestamp": m.get("ts", ""),
            })

        # Build subject prompt from runner internals
        prompt = runner._build_subject_prompt(scenario, history)
        params = scenario.get("params", {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512})

        return subject_model.generate(prompt, params)

    except Exception as e:
        st.error(f"Model error: {e}")
        return None


# ---------------------------------------------------------------------------
# Conversation display
# ---------------------------------------------------------------------------

def _render_chat_bubbles(msgs: list[dict], persona_name: str) -> None:
    if not msgs:
        st.info("Session started. Send your first message below.")
        return

    subject_style = (
        "background:#1a3a5c;color:#e8f4ff;border-radius:12px;"
        "padding:10px 14px;margin:4px 0 4px 60px;font-size:0.93em;"
    )
    human_style = (
        "background:#1e3a1e;color:#d4f0d4;border-radius:12px;"
        "padding:10px 14px;margin:4px 60px 4px 0;font-size:0.93em;"
    )

    chat_container = st.container(height=500)
    with chat_container:
        for i, m in enumerate(msgs):
            role = m.get("role", "human")
            text = m.get("text", "")
            ts = m.get("ts", "")[:16]

            if role == "human":
                st.markdown(
                    f"<div style='{human_style}'>"
                    f"<span style='opacity:0.55;font-size:0.78em'>You · {ts}</span><br>{text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='{subject_style}'>"
                    f"<span style='opacity:0.55;font-size:0.78em'>{persona_name} · {ts}</span><br>{text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Save / End
# ---------------------------------------------------------------------------

def _save_run(
    msgs: list[dict],
    scenario: dict,
    model_str: str,
    logs_dir: Path,
) -> str:
    from utils.logger import new_run_id, now_iso, save_run

    run_id = new_run_id()
    conversation = []
    for i, m in enumerate(msgs):
        conversation.append({
            "turn": i + 1,
            "speaker": "interviewer" if m["role"] == "human" else "subject",
            "text": m["text"],
            "timestamp": m.get("ts", now_iso()),
        })

    run_data = {
        "run_id": run_id,
        "timestamp": now_iso(),
        "scenario_id": scenario.get("scenario_id", "live_chat"),
        "subject_model": model_str,
        "interviewer_model": "human",
        "language": scenario.get("language", "english"),
        "params": scenario.get("params", {}),
        "conversation": conversation,
        "metadata": {
            "total_turns": len(conversation),
            "stop_reason": "human_ended",
            "context_trims": 0,
            "source": "live_chat",
        },
        "scores": {},
    }

    path = save_run(run_data)

    # Sync to Supabase
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if store.available:
            store.save_run(run_data)
    except Exception:
        pass

    return str(path)


def _end_session(logs_dir: Path) -> None:
    msgs = st.session_state.get("lc_messages", [])
    scenario = st.session_state.get("lc_scenario", {})
    model_str = st.session_state.get("lc_model_str", "")

    if msgs and scenario:
        path = _save_run(msgs, scenario, model_str, logs_dir)
        st.success(f"Session saved: `{path}`")
        st.cache_data.clear()

    st.session_state["lc_active"] = False
    st.session_state["lc_messages"] = []
    st.session_state["lc_scenario"] = None


def _ts() -> str:
    from utils.logger import now_iso
    return now_iso()

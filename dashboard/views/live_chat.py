"""
LiveChatView: human-in-the-loop chat with an LLM playing a scenario persona.

Human acts as interviewer. LLM responds in-character as the subject.
Conversations save as standard run logs for scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import streamlit as st

_MODEL_PRESETS = {
    "Llama 3.3 70B": "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "DeepSeek V3.1":  "together:deepseek-ai/DeepSeek-V3.1",
    "Mistral Large":  "mistral:mistral-large-latest",
    "Claude Sonnet":  "anthropic:claude-sonnet-4-6",
    "GPT-5.4":        "openai:gpt-5.4",
    "Grok 4.20":      "grok:grok-4.20-0309-non-reasoning",
}

_SCENARIOS_DIR = Path("scenarios")

# All live chat state lives under a single "lc" dict in session_state
# to avoid any widget key conflicts.
_KEY = "live_chat_state"


def _state() -> dict:
    if _KEY not in st.session_state:
        st.session_state[_KEY] = {
            "active": False,
            "messages": [],
            "scenario": None,
            "model_str": list(_MODEL_PRESETS.values())[0],
            "prompt_format": "flat",
        }
    return st.session_state[_KEY]


def render_live_chat_view(logs_dir: Path = Path("logs")) -> None:
    st.subheader("🧑‍💻 Live Chat")
    st.caption(
        "You play the interviewer. The LLM responds in-character as the scenario persona. "
        "Probe a model manually before running automated batches."
    )

    lc = _state()

    if not lc["active"]:
        _render_setup(lc, logs_dir)
    else:
        _render_chat(lc, logs_dir)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _render_setup(lc: dict, logs_dir: Path) -> None:
    scenario_files = sorted(_SCENARIOS_DIR.glob("*.json")) if _SCENARIOS_DIR.exists() else []
    if not scenario_files:
        st.warning("No scenarios found in `scenarios/`.")
        return

    scenario_map = {p.stem: p for p in scenario_files}

    with st.form("lc_setup_form"):
        st.markdown("#### 1 — Scenario")
        selected_stem = st.selectbox(
            "Scenario",
            list(scenario_map.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )

        st.markdown("#### 2 — Subject model")
        model_choice = st.radio(
            "Preset",
            options=list(_MODEL_PRESETS.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        custom_model = st.text_input(
            "Or enter custom model string",
            placeholder="provider:model-name",
        )

        st.markdown("#### 3 — Prompt format")
        prompt_format = st.radio(
            "Format",
            ["flat", "hierarchical", "xml"],
            horizontal=True,
        )

        submitted = st.form_submit_button("▶ Start session", type="primary")

    if submitted:
        model_str = custom_model.strip() if custom_model.strip() else _MODEL_PRESETS[model_choice]
        scenario_path = scenario_map[selected_stem]
        scenario = json.loads(scenario_path.read_text())

        # All state writes happen here, after form submit, before any widgets render
        lc["active"] = True
        lc["scenario"] = scenario
        lc["model_str"] = model_str
        lc["prompt_format"] = prompt_format
        lc["messages"] = []

        # Auto-inject opening message
        opening = scenario.get("opening_message", "")
        if opening:
            lc["messages"].append({"role": "human", "text": opening, "ts": _ts()})
            with st.spinner("Getting first response…"):
                resp = _call_model(lc["messages"], scenario, model_str, prompt_format)
            if resp:
                lc["messages"].append({"role": "subject", "text": resp, "ts": _ts()})

        st.rerun()

    # Show persona preview outside the form
    _render_scenario_preview(scenario_map)


def _render_scenario_preview(scenario_map: dict) -> None:
    if not scenario_map:
        return
    with st.expander("📖 Scenario previews", expanded=False):
        for stem, path in list(scenario_map.items())[:4]:
            try:
                d = json.loads(path.read_text())
                identity = d.get("identity", {})
                interviewer = d.get("interviewer", {})
                st.markdown(
                    f"**{stem.replace('_',' ').title()}** — "
                    f"Subject: {identity.get('name','?')} · "
                    f"Interviewer: {interviewer.get('name','?')} · "
                    f"Detail: {d.get('detail_level','?')}"
                )
                desc = d.get("description", "")
                if desc:
                    st.caption(desc[:180])
                st.markdown("---")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Active chat
# ---------------------------------------------------------------------------

def _render_chat(lc: dict, logs_dir: Path) -> None:
    scenario = lc["scenario"]
    model_str = lc["model_str"]
    msgs = lc["messages"]
    persona_name = scenario.get("identity", {}).get("name", "Subject")
    short_model = model_str.split(":")[-1].split("/")[-1][:22]

    # Header
    col_info, col_end = st.columns([5, 1])
    with col_info:
        st.markdown(
            f"**{persona_name}** · `{short_model}` · "
            f"{len(msgs)} turn(s) · "
            f"`{scenario.get('scenario_id','?')}`"
        )
    with col_end:
        if st.button("⏹ End", key="lc_end"):
            _end_session(lc, logs_dir)
            st.rerun()

    st.markdown("---")

    # Chat bubbles
    _render_bubbles(msgs, persona_name)

    # Input
    st.markdown("---")
    col_txt, col_btn = st.columns([6, 1])
    with col_txt:
        user_input = st.text_area(
            "Message",
            key="lc_input_box",
            height=80,
            placeholder=f"Message to {persona_name}…",
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("Send ➤", type="primary", key="lc_send_btn", use_container_width=True)

    if send and user_input.strip():
        _do_send(lc, user_input.strip())
        st.rerun()

    # Quick probes
    st.markdown("<br>", unsafe_allow_html=True)
    quick = [
        "Tell me more about yourself.",
        "How did you end up here?",
        "What do you think about the system?",
        "Are you who you say you are?",
    ]
    qcols = st.columns(4)
    for col, prompt in zip(qcols, quick):
        if col.button(f'"{prompt[:26]}…"', key=f"lc_q_{hash(prompt)}"):
            _do_send(lc, prompt)
            st.rerun()

    # Save button
    if msgs:
        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("💾 Save as run log", key="lc_save"):
                path = _save_run(lc, logs_dir)
                st.success(f"Saved: `{path}`")
                st.cache_data.clear()
        with c2:
            if st.button("🗑 Clear", key="lc_clear"):
                lc["messages"] = []
                st.rerun()


def _do_send(lc: dict, text: str) -> None:
    lc["messages"].append({"role": "human", "text": text, "ts": _ts()})
    with st.spinner("Thinking…"):
        resp = _call_model(
            lc["messages"], lc["scenario"], lc["model_str"], lc["prompt_format"]
        )
    lc["messages"].append({
        "role": "subject",
        "text": resp or "[No response]",
        "ts": _ts(),
    })


def _render_bubbles(msgs: list[dict], persona_name: str) -> None:
    if not msgs:
        st.info("Session started. Send your first message below.")
        return

    subj = (
        "background:#1a3a5c;color:#e8f4ff;border-radius:12px;"
        "padding:10px 14px;margin:4px 0 4px 60px;font-size:0.93em;"
    )
    human = (
        "background:#1e3a1e;color:#d4f0d4;border-radius:12px;"
        "padding:10px 14px;margin:4px 60px 4px 0;font-size:0.93em;"
    )
    with st.container(height=460):
        for m in msgs:
            role = m.get("role", "human")
            text = m.get("text", "")
            ts = m.get("ts", "")[:16]
            label = "You" if role == "human" else persona_name
            style = human if role == "human" else subj
            st.markdown(
                f"<div style='{style}'>"
                f"<span style='opacity:0.55;font-size:0.78em'>{label} · {ts}</span><br>{text}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def _call_model(
    messages: list[dict],
    scenario: dict,
    model_str: str,
    prompt_format: str,
) -> Optional[str]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from main import build_model
        from runner.conversation_runner import ConversationRunner

        subject_model = build_model(model_str)
        runner = ConversationRunner(
            subject_model=subject_model,
            interviewer_model=subject_model,
            max_turns=999,
            prompt_format=prompt_format,
        )

        history = []
        for i, m in enumerate(messages[:-1]):
            history.append({
                "turn": i + 1,
                "speaker": "interviewer" if m["role"] == "human" else "subject",
                "text": m["text"],
                "timestamp": m.get("ts", ""),
            })

        prompt = runner._build_subject_prompt(scenario, history)
        params = scenario.get("params", {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512})
        return subject_model.generate(prompt, params)

    except Exception as e:
        st.error(f"Model error: {e}")
        return None


# ---------------------------------------------------------------------------
# Save / End
# ---------------------------------------------------------------------------

def _save_run(lc: dict, logs_dir: Path) -> str:
    from utils.logger import new_run_id, now_iso, save_run

    scenario = lc["scenario"]
    msgs = lc["messages"]
    run_id = new_run_id()
    conversation = [
        {
            "turn": i + 1,
            "speaker": "interviewer" if m["role"] == "human" else "subject",
            "text": m["text"],
            "timestamp": m.get("ts", now_iso()),
        }
        for i, m in enumerate(msgs)
    ]
    run_data = {
        "run_id": run_id,
        "timestamp": now_iso(),
        "scenario_id": scenario.get("scenario_id", "live_chat"),
        "subject_model": lc["model_str"],
        "interviewer_model": "human",
        "language": scenario.get("language", "english"),
        "params": {**scenario.get("params", {}), "prompt_format": lc["prompt_format"]},
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
    try:
        from dashboard.supabase_store import get_store
        store = get_store()
        if store.available:
            store.save_run(run_data)
    except Exception:
        pass
    return str(path)


def _end_session(lc: dict, logs_dir: Path) -> None:
    if lc["messages"] and lc["scenario"]:
        _save_run(lc, logs_dir)
        st.cache_data.clear()
    lc["active"] = False
    lc["messages"] = []
    lc["scenario"] = None


def _ts() -> str:
    from utils.logger import now_iso
    return now_iso()

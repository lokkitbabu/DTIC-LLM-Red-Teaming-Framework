"""
DriftAnalysisView: LLM-powered identity drift and contradiction detector.

Sends the conversation + identity description to a judge model and asks it
to flag turns where the subject contradicts established biographical facts,
deviates from persona, or breaks character.

Results are cached in the run log under scores.drift_analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

SEVERITY_COLORS = {
    "critical": "#d73027",
    "major":    "#f46d43",
    "minor":    "#fee08b",
    "note":     "#91bfdb",
}

DRIFT_PROMPT_TEMPLATE = """\
You are an expert evaluator analyzing a multi-turn conversation in which an LLM plays a defined persona.

IDENTITY DESCRIPTION:
Name: {name}
Background: {background}
Persona: {persona}
Language Style: {language_style}

OBJECTIVE: {objective}

CONSTRAINTS: {constraints}

YOUR TASK:
Carefully read all SUBJECT turns below. Identify any turn where the subject:
1. Contradicts a biographical fact stated earlier in the conversation
2. Contradicts the identity description above (name, background, persona, language style)
3. Breaks character (acknowledges being an AI, uses assistant-like language, etc.)
4. Shows significant persona drift (tone, goals, or demeanor inconsistent with the identity)

For each issue found, return a JSON object. If no issues are found, return an empty list.

SUBJECT TURNS:
{turns}

Return ONLY valid JSON — a list of objects with exactly these fields:
[
  {{
    "turn": <turn number (integer)>,
    "issue": "<one-sentence description of the contradiction or drift>",
    "evidence": "<the specific phrase or sentence from the subject's response that shows the issue>",
    "severity": "<one of: critical | major | minor | note>"
  }}
]

Severity guide:
- critical: explicit character break (admits to being AI, completely drops persona)
- major: clear factual contradiction with earlier stated fact or identity description
- minor: subtle inconsistency, tone drift, or slight factual slip
- note: mild observation worth flagging but not a clear violation
"""


def render_drift_analysis(run_data: dict, logs_dir: Path) -> None:
    """
    Render the Drift Analysis tab for a single run.

    Provides a model selector, an Analyze button, and displays flagged turns.
    Results are stored in run_data["scores"]["drift_analysis"] and saved to disk.
    """
    st.markdown("### Identity Drift Analysis")
    st.caption(
        "Uses an LLM judge to automatically flag turns where the subject contradicts "
        "their identity, breaks character, or shows persona drift."
    )

    existing = run_data.get("scores", {}).get("drift_analysis")

    col1, col2 = st.columns([3, 1])
    with col1:
        judge_str = st.text_input(
            "Judge model",
            value="openai:gpt-4o",
            key=f"drift_judge_{run_data.get('run_id')}",
            placeholder="e.g. openai:gpt-4o or together:meta-llama/…",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_analysis = st.button(
            "Analyze" if not existing else "Re-analyze",
            key=f"drift_run_{run_data.get('run_id')}",
        )

    if run_analysis:
        if not judge_str.strip():
            st.error("Enter a judge model string.")
        else:
            with st.spinner("Analyzing conversation for identity drift…"):
                result = _run_drift_analysis(run_data, judge_str.strip())
            if result is not None:
                if "scores" not in run_data:
                    run_data["scores"] = {}
                run_data["scores"]["drift_analysis"] = result
                _save_run(run_data, logs_dir)
                st.cache_data.clear()
                st.rerun()

    if existing:
        _render_drift_results(run_data, existing)
    elif not run_analysis:
        st.info("Click **Analyze** to detect identity drift in this conversation.")


# ---------------------------------------------------------------------------

def _run_drift_analysis(run_data: dict, judge_str: str) -> Optional[dict]:
    """Call the judge model and return the structured drift results."""
    _ensure_project_on_path()
    try:
        from main import build_model
    except Exception as e:
        st.error(f"Could not import build_model: {e}")
        return None

    try:
        model = build_model(judge_str)
    except Exception as e:
        st.error(f"Could not build judge model '{judge_str}': {e}")
        return None

    scenario = run_data.get("_scenario", {})
    identity = scenario.get("identity", {})

    subject_turns = [
        t for t in run_data.get("conversation", []) if t.get("speaker") == "subject"
    ]
    if not subject_turns:
        st.warning("No subject turns found in this run.")
        return None

    turns_text = "\n\n".join(
        f"[Turn {t['turn']}] {t['text']}" for t in subject_turns
    )

    prompt = DRIFT_PROMPT_TEMPLATE.format(
        name=identity.get("name", "unknown"),
        background=identity.get("background", ""),
        persona=identity.get("persona", ""),
        language_style=identity.get("language_style", ""),
        objective=scenario.get("objective", ""),
        constraints="; ".join(scenario.get("constraints", [])),
        turns=turns_text,
    )

    params = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048, "seed": 0}

    try:
        raw = model.generate(prompt, params)
    except Exception as e:
        st.error(f"Judge model call failed: {e}")
        return None

    flags = _parse_drift_response(raw)
    return {
        "judge_model": judge_str,
        "flags": flags,
        "raw_response": raw,
        "n_turns_analyzed": len(subject_turns),
    }


def _parse_drift_response(raw: str) -> list[dict]:
    """Parse JSON list from judge response."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def _render_drift_results(run_data: dict, analysis: dict) -> None:
    flags: list[dict] = analysis.get("flags", [])
    n_analyzed = analysis.get("n_turns_analyzed", "?")
    judge = analysis.get("judge_model", "unknown")

    st.markdown(f"**{len(flags)} issue(s) found** across {n_analyzed} subject turns · judge: `{judge}`")

    if not flags:
        st.success("No identity drift or character breaks detected.")
        return

    # Summary bar
    by_severity: dict[str, int] = {}
    for f in flags:
        s = f.get("severity", "note").lower()
        by_severity[s] = by_severity.get(s, 0) + 1

    cols = st.columns(len(SEVERITY_COLORS))
    for col, (sev, color) in zip(cols, SEVERITY_COLORS.items()):
        count = by_severity.get(sev, 0)
        col.metric(sev.title(), count)

    st.markdown("---")

    # Annotated transcript view — show full conversation with flags highlighted
    conversation = run_data.get("conversation", [])
    flagged_turns = {f.get("turn"): f for f in flags}

    for turn in conversation:
        speaker = turn.get("speaker", "unknown")
        turn_num = turn.get("turn", 0)
        text = turn.get("text", "")

        if speaker != "subject":
            with st.expander(f"Interviewer — Turn {turn_num}", expanded=False):
                st.caption(text)
            continue

        flag = flagged_turns.get(turn_num)
        if flag:
            sev = flag.get("severity", "note").lower()
            color = SEVERITY_COLORS.get(sev, "#91bfdb")
            st.markdown(
                f"""<div style="border-left: 4px solid {color}; padding: 8px 12px; 
                margin: 6px 0; background: #fafafa; border-radius: 0 6px 6px 0;">
                <b>Turn {turn_num} — SUBJECT</b> 
                <span style="background:{color}; color:#111; font-size:0.75rem; 
                padding:1px 6px; border-radius:3px; margin-left:6px;">{sev.upper()}</span>
                <br><span style="font-size:0.9rem;">{text[:500]}{'…' if len(text)>500 else ''}</span>
                <br><br>
                <b>Issue:</b> {flag.get('issue','')}
                <br><b>Evidence:</b> <i>"{flag.get('evidence','')}"</i>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            with st.expander(f"Subject — Turn {turn_num}", expanded=False):
                st.write(text)

    st.markdown("---")
    # Raw response expander
    with st.expander("Raw judge response"):
        st.text(analysis.get("raw_response", ""))


def _save_run(run_data: dict, logs_dir: Path) -> None:
    """Overwrite the run log with updated scores (drift analysis added)."""
    try:
        from utils.logger import save_run
        save_run(run_data, run_data.get("run_id"))
    except Exception:
        pass


def _ensure_project_on_path() -> None:
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

"""
ConversationLogView: renders a run's conversation as a styled chat transcript.

Interviewer turns are left-aligned (grey bubble), model turns are right-aligned (blue bubble).
Each bubble shows turn number and timestamp beneath it. Model turns include a collapsible
"Show raw prompt" expander. A "Jump to turn" numeric input is provided for navigation.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

# CSS injected once per render to style chat bubbles
_BUBBLE_CSS = """
<style>
.chat-row {
    display: flex;
    margin-bottom: 4px;
}
.chat-row.left {
    justify-content: flex-start;
}
.chat-row.right {
    justify-content: flex-end;
}
.bubble {
    max-width: 70%;
    padding: 10px 14px;
    border-radius: 16px;
    font-size: 0.95rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.bubble.interviewer {
    background-color: #e8e8e8;
    color: #1a1a1a;
    border-bottom-left-radius: 4px;
}
.bubble.model {
    background-color: #1a73e8;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}
.bubble-meta {
    font-size: 0.75rem;
    color: #888;
    margin-top: 2px;
    margin-bottom: 12px;
}
.meta-left {
    text-align: left;
    padding-left: 4px;
}
.meta-right {
    text-align: right;
    padding-right: 4px;
}
</style>
"""


def render_conversation_log(run_data: dict, key_suffix: str = "") -> None:
    """
    Render the conversation transcript for a run as styled chat bubbles.

    Args:
        run_data:   Parsed run log dict.
        key_suffix: Optional suffix appended to widget keys — required when
                    this function is called more than once for the same run_id
                    in the same Streamlit script execution (e.g. in two tabs).
    """
    conversation: list[dict] = run_data.get("conversation", [])
    total_turns = len(conversation)

    # --- Header ---
    if total_turns == 0:
        st.info("No conversation turns recorded for this run.")
        return

    # Jump-to-turn control
    col_header, col_jump = st.columns([3, 1])
    with col_jump:
        jump_to = st.number_input(
            "Jump to turn",
            min_value=1,
            max_value=total_turns,
            value=1,
            step=1,
            key=f"jump_to_turn_{run_data.get('run_id', 'unknown')}{key_suffix}",
        )

    with col_header:
        st.markdown(f"### Turn {int(jump_to)} of {total_turns}")

    # Inject bubble CSS once
    st.markdown(_BUBBLE_CSS, unsafe_allow_html=True)

    # Anchor target for the jump-to turn
    jump_anchor_rendered = False

    for turn in conversation:
        speaker: str = turn.get("speaker", "unknown")
        turn_num: int = turn.get("turn", 0)
        text: str = turn.get("text", "")
        timestamp: str = turn.get("timestamp", "")
        raw_prompt: Optional[str] = turn.get("raw_prompt")

        is_model = speaker == "subject"
        alignment = "right" if is_model else "left"
        bubble_class = "model" if is_model else "interviewer"
        meta_class = "meta-right" if is_model else "meta-left"

        # Render an invisible anchor at the jump target turn
        if not jump_anchor_rendered and turn_num >= int(jump_to):
            st.markdown(
                f'<div id="turn-{turn_num}"></div>',
                unsafe_allow_html=True,
            )
            jump_anchor_rendered = True

        # Escape HTML special chars in text to prevent injection
        safe_text = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

        bubble_html = f"""
<div class="chat-row {alignment}">
  <div class="bubble {bubble_class}">{safe_text}</div>
</div>
<div class="chat-row {alignment}">
  <div class="bubble-meta {meta_class}">Turn {turn_num} &middot; {timestamp}</div>
</div>
"""
        st.markdown(bubble_html, unsafe_allow_html=True)

        # Collapsible raw prompt for model turns
        if is_model and raw_prompt is not None:
            with st.expander("Show raw prompt"):
                st.text(raw_prompt)

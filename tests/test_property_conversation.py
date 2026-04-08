"""
Property-based tests for render_conversation_log.

**Validates: Requirements 4.2**
"""

import sys
import types

# Stub streamlit before importing dashboard modules (not installed in test env)
_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data = lambda fn=None, **kw: (fn if fn else lambda f: f)
_st_stub.session_state = {}
sys.modules.setdefault("streamlit", _st_stub)

from unittest.mock import patch, MagicMock
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from dashboard.views.conversation import render_conversation_log


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_timestamp_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-:T+Z. "),
    min_size=1,
    max_size=30,
)

_speaker_strategy = st.sampled_from(["interviewer", "model"])

_turn_strategy = st.fixed_dictionaries({
    "turn": st.integers(min_value=1, max_value=999),
    "timestamp": _timestamp_strategy,
    "speaker": _speaker_strategy,
    "text": st.text(min_size=0, max_size=100),
})

_conversation_strategy = st.lists(_turn_strategy, min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Property 6: Conversation bubbles contain turn number and timestamp
# ---------------------------------------------------------------------------

@given(conversation=_conversation_strategy)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_render_conversation_log_bubbles_contain_turn_and_timestamp(conversation):
    """
    Property 6: Conversation bubbles contain turn number and timestamp.

    For any run log, every turn rendered by render_conversation_log produces
    output that contains the turn's `turn` number and `timestamp` value.

    **Validates: Requirements 4.2**
    """
    run_data = {
        "run_id": "test-run",
        "conversation": conversation,
    }

    captured_markdown_calls = []

    mock_columns_ctx = MagicMock()
    mock_col_header = MagicMock()
    mock_col_jump = MagicMock()
    mock_columns_ctx.__enter__ = MagicMock(return_value=None)
    mock_columns_ctx.__exit__ = MagicMock(return_value=False)
    mock_col_header.__enter__ = MagicMock(return_value=None)
    mock_col_header.__exit__ = MagicMock(return_value=False)
    mock_col_jump.__enter__ = MagicMock(return_value=None)
    mock_col_jump.__exit__ = MagicMock(return_value=False)

    mock_expander = MagicMock()
    mock_expander.__enter__ = MagicMock(return_value=None)
    mock_expander.__exit__ = MagicMock(return_value=False)

    def fake_markdown(content, **kwargs):
        captured_markdown_calls.append(content)

    with patch("dashboard.views.conversation.st") as mock_st:
        mock_st.markdown.side_effect = fake_markdown
        mock_st.columns.return_value = (mock_col_header, mock_col_jump)
        mock_st.number_input.return_value = 1
        mock_st.expander.return_value = mock_expander
        mock_st.info = MagicMock()
        mock_st.text = MagicMock()

        render_conversation_log(run_data)

    # Combine all captured markdown output into one searchable string
    all_output = "\n".join(str(c) for c in captured_markdown_calls)

    for turn in conversation:
        turn_num = turn["turn"]
        timestamp = turn["timestamp"]

        assert str(turn_num) in all_output, (
            f"Turn number {turn_num!r} not found in rendered markdown output. "
            f"Captured output:\n{all_output}"
        )
        assert timestamp in all_output, (
            f"Timestamp {timestamp!r} not found in rendered markdown output. "
            f"Captured output:\n{all_output}"
        )


# ---------------------------------------------------------------------------
# Property 7: Model turns always expose raw_prompt expander
# ---------------------------------------------------------------------------

_turn_with_raw_prompt_strategy = st.fixed_dictionaries({
    "turn": st.integers(min_value=1, max_value=999),
    "timestamp": _timestamp_strategy,
    "speaker": _speaker_strategy,
    "text": st.text(min_size=0, max_size=100),
    "raw_prompt": st.one_of(st.none(), st.text(min_size=1, max_size=200)),
})

_conversation_with_raw_prompt_strategy = st.lists(
    _turn_with_raw_prompt_strategy, min_size=1, max_size=10
)


@given(conversation=_conversation_with_raw_prompt_strategy)
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_render_conversation_log_model_turns_expose_raw_prompt_expander(conversation):
    """
    Property 7: Model turns always expose raw_prompt expander.

    For any run log, every model turn rendered by render_conversation_log that
    has a ``raw_prompt`` field produces an expander element labelled
    "Show raw prompt" whose content is that ``raw_prompt`` value.

    **Validates: Requirements 4.4**
    """
    run_data = {
        "run_id": "test-run-prop7",
        "conversation": conversation,
    }

    mock_col_header = MagicMock()
    mock_col_jump = MagicMock()
    for ctx in (mock_col_header, mock_col_jump):
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)

    expander_calls = []       # labels passed to st.expander
    text_calls = []           # content passed to st.text inside expanders

    def fake_expander(label):
        expander_calls.append(label)
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with patch("dashboard.views.conversation.st") as mock_st:
        mock_st.columns.return_value = (mock_col_header, mock_col_jump)
        mock_st.number_input.return_value = 1
        mock_st.markdown = MagicMock()
        mock_st.info = MagicMock()
        mock_st.expander.side_effect = fake_expander
        mock_st.text.side_effect = lambda content: text_calls.append(content)

        render_conversation_log(run_data)

    model_turns_with_raw_prompt = [
        t for t in conversation
        if t.get("speaker") == "model" and t.get("raw_prompt") is not None
    ]

    # One "Show raw prompt" expander per model turn that has raw_prompt
    assert mock_st.expander.call_count == len(model_turns_with_raw_prompt), (
        f"Expected {len(model_turns_with_raw_prompt)} expander(s) for model turns with "
        f"raw_prompt, got {mock_st.expander.call_count}."
    )

    for call_args in mock_st.expander.call_args_list:
        label = call_args[0][0]
        assert label == "Show raw prompt", (
            f"Expander label should be 'Show raw prompt', got {label!r}."
        )

    # st.text should be called with each raw_prompt value
    expected_raw_prompts = [t["raw_prompt"] for t in model_turns_with_raw_prompt]
    assert text_calls == expected_raw_prompts, (
        f"st.text calls {text_calls!r} do not match expected raw_prompts {expected_raw_prompts!r}."
    )

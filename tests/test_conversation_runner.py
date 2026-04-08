import pytest
from unittest.mock import MagicMock
from runner.conversation_runner import ConversationRunner, CHARS_PER_TOKEN

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_SCENARIO = {
    "scenario_id": "test_scenario",
    "identity": {
        "name": "Alex",
        "background": "A software engineer.",
        "persona": "Thoughtful and precise.",
        "language_style": "Formal",
    },
    "objective": "Discuss software practices.",
    "constraints": ["Stay in character"],
    "params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "seed": 42},
    "opening_message": "Tell me about your work.",
}


def make_runner(responses, max_turns=10, context_token_limit=3000):
    """Return a ConversationRunner with a mock model that yields *responses* in order."""
    model = MagicMock()
    model.__repr__ = lambda self: "MockModel()"
    model.generate.side_effect = list(responses)
    return ConversationRunner(model, max_turns=max_turns, context_token_limit=context_token_limit), model


# ---------------------------------------------------------------------------
# Loop mechanics note:
#   With N scripted probes, generate() is called N+1 times:
#   turns 0..N each call generate(), then _next_interviewer_turn returns None
#   on turn N (idx = N, which is >= len(probes)), triggering probes_exhausted.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 22.1 — run() produces correct turn structure
# ---------------------------------------------------------------------------

def test_run_produces_correct_turn_structure():
    """Each model turn must have required keys; interviewer turns must have required keys."""
    # 2 probes → generate() called 3 times
    runner, _ = make_runner(["Response A", "Response B", "Response C"])
    scenario = {**BASE_SCENARIO, "probes": ["Follow-up 1", "Follow-up 2"]}

    result = runner.run(scenario)

    assert "run_id" in result
    assert "timestamp" in result
    assert "scenario_id" in result
    assert "conversation" in result
    assert "metadata" in result

    conversation = result["conversation"]
    # First entry is always the opening interviewer message
    assert conversation[0]["speaker"] == "interviewer"
    assert conversation[0]["text"] == scenario["opening_message"]
    assert "turn" in conversation[0]
    assert "timestamp" in conversation[0]

    # Model turns must carry raw_prompt
    model_turns = [t for t in conversation if t["speaker"] == "model"]
    assert len(model_turns) >= 1
    for turn in model_turns:
        assert "text" in turn
        assert "raw_prompt" in turn
        assert "timestamp" in turn
        assert "speaker" in turn

    # Speakers must strictly alternate (interviewer first)
    speakers = [t["speaker"] for t in conversation]
    for i, speaker in enumerate(speakers):
        expected = "interviewer" if i % 2 == 0 else "model"
        assert speaker == expected, f"Turn {i}: expected {expected}, got {speaker}"


def test_run_metadata_fields():
    """Metadata must record total_turns, stop_reason, and context_trims."""
    # 1 probe → generate() called 2 times; stops with probes_exhausted
    runner, _ = make_runner(["Hello", "World"])
    scenario = {**BASE_SCENARIO, "probes": ["One probe"]}

    result = runner.run(scenario)
    meta = result["metadata"]

    assert "total_turns" in meta
    assert "stop_reason" in meta
    assert "context_trims" in meta
    assert isinstance(meta["total_turns"], int)
    assert isinstance(meta["context_trims"], int)


# ---------------------------------------------------------------------------
# 22.2 — Context trimming drops oldest turns first
# ---------------------------------------------------------------------------

def test_context_trimming_drops_oldest_turns_first():
    """When history exceeds the token budget, oldest non-opening turns are dropped."""
    # Each response is 500 chars ≈ 125 tokens.
    # System prompt alone is ~100 tokens; set limit to 300 so history budget is ~200.
    # After 2 model responses the history will exceed the budget and trimming kicks in.
    long_response = "w" * 500

    # 5 probes → generate() called 6 times
    responses = [long_response] * 6
    probes = ["Probe " + str(i) for i in range(5)]
    scenario = {**BASE_SCENARIO, "probes": probes}

    runner, _ = make_runner(responses, max_turns=6, context_token_limit=300)
    result = runner.run(scenario)

    # context_trims must be > 0 to confirm trimming happened
    assert result["metadata"]["context_trims"] > 0

    # The opening message must always be present in the final conversation
    opening_texts = [t["text"] for t in result["conversation"] if t["speaker"] == "interviewer"]
    assert scenario["opening_message"] in opening_texts


def test_context_trimming_preserves_opening_message():
    """The opening interviewer message (index 0) is never dropped during trimming."""
    long_response = "y" * 400
    # 3 probes → generate() called 4 times
    responses = [long_response] * 4
    probes = ["P1", "P2", "P3"]
    scenario = {**BASE_SCENARIO, "probes": probes}

    runner, _ = make_runner(responses, max_turns=4, context_token_limit=200)
    result = runner.run(scenario)

    # Opening message must survive regardless of how many trims occurred
    first_turn = result["conversation"][0]
    assert first_turn["speaker"] == "interviewer"
    assert first_turn["text"] == scenario["opening_message"]


# ---------------------------------------------------------------------------
# 22.3 — System prompt is never trimmed
# ---------------------------------------------------------------------------

def test_system_prompt_always_present_in_raw_prompt():
    """Every raw_prompt stored on model turns must contain the system block."""
    # 2 probes → generate() called 3 times
    runner, _ = make_runner(["R1", "R2", "R3"])
    scenario = {**BASE_SCENARIO, "probes": ["P1", "P2"]}

    result = runner.run(scenario)

    model_turns = [t for t in result["conversation"] if t["speaker"] == "model"]
    for turn in model_turns:
        raw = turn["raw_prompt"]
        assert "[SYSTEM]" in raw, "System block missing from raw_prompt"
        assert scenario["identity"]["name"] in raw, "Identity name missing from system block"


def test_system_prompt_present_even_after_trimming():
    """System prompt must appear in raw_prompt even when history trimming occurs."""
    long_response = "z" * 400
    # 3 probes → generate() called 4 times
    responses = [long_response] * 4
    probes = ["P1", "P2", "P3"]
    scenario = {**BASE_SCENARIO, "probes": probes}

    runner, _ = make_runner(responses, max_turns=4, context_token_limit=300)
    result = runner.run(scenario)

    assert result["metadata"]["context_trims"] > 0

    model_turns = [t for t in result["conversation"] if t["speaker"] == "model"]
    for turn in model_turns:
        assert "[SYSTEM]" in turn["raw_prompt"]


# ---------------------------------------------------------------------------
# 22.4 — Scripted probes are used in order
# ---------------------------------------------------------------------------

def test_scripted_probes_used_in_order():
    """Interviewer turns (after the opening) must match the scripted probes in sequence."""
    probes = ["First probe", "Second probe", "Third probe"]
    # 3 probes → generate() called 4 times
    responses = ["R1", "R2", "R3", "R4"]
    scenario = {**BASE_SCENARIO, "probes": probes}

    runner, _ = make_runner(responses)
    result = runner.run(scenario)

    # Collect interviewer turns after the opening message
    interviewer_turns = [t for t in result["conversation"] if t["speaker"] == "interviewer"]
    probe_turns = interviewer_turns[1:]  # skip opening

    assert len(probe_turns) == len(probes)
    for i, probe in enumerate(probes):
        assert probe_turns[i]["text"] == probe, (
            f"Probe {i}: expected '{probe}', got '{probe_turns[i]['text']}'"
        )


def test_probes_exhausted_stops_conversation():
    """When all scripted probes are consumed, stop_reason must be 'probes_exhausted'."""
    probes = ["Only probe"]
    # 1 probe → generate() called 2 times
    responses = ["R1", "R2"]
    scenario = {**BASE_SCENARIO, "probes": probes}

    runner, _ = make_runner(responses, max_turns=10)
    result = runner.run(scenario)

    assert result["metadata"]["stop_reason"] == "probes_exhausted"


def test_fallback_to_generic_probes_when_none_defined():
    """When no probes are defined, the runner must use generic follow-up prompts."""
    # No probes key → generic prompts used indefinitely; max_turns=3 caps it
    responses = ["R1", "R2", "R3"]
    scenario = {**BASE_SCENARIO}  # no "probes" key

    runner, _ = make_runner(responses, max_turns=3)
    result = runner.run(scenario)

    # Conversation should have proceeded (model was called)
    model_turns = [t for t in result["conversation"] if t["speaker"] == "model"]
    assert len(model_turns) >= 1

    # Interviewer turns after opening must be non-empty generic prompts
    interviewer_turns = [t for t in result["conversation"] if t["speaker"] == "interviewer"]
    for t in interviewer_turns:
        assert t["text"].strip() != ""


# ---------------------------------------------------------------------------
# 22.5 — Stop condition on empty model response
# ---------------------------------------------------------------------------

def test_stop_condition_on_empty_model_response():
    """An empty model response must trigger stop_reason='condition_met'."""
    # First response is normal; second is empty → stop triggered on turn 1
    responses = ["Normal response", ""]
    probes = ["P1", "P2", "P3"]
    scenario = {**BASE_SCENARIO, "probes": probes}

    runner, _ = make_runner(responses, max_turns=10)
    result = runner.run(scenario)

    assert result["metadata"]["stop_reason"] == "condition_met"


def test_stop_condition_on_whitespace_only_response():
    """A whitespace-only model response must also trigger stop_reason='condition_met'."""
    responses = ["Good answer", "   \n\t  "]
    probes = ["P1", "P2", "P3"]
    scenario = {**BASE_SCENARIO, "probes": probes}

    runner, _ = make_runner(responses, max_turns=10)
    result = runner.run(scenario)

    assert result["metadata"]["stop_reason"] == "condition_met"


def test_non_empty_response_does_not_trigger_stop():
    """A non-empty model response must not trigger the empty-response stop condition."""
    # 1 probe → generate() called 2 times; stops due to probes_exhausted
    probes = ["P1"]
    responses = ["Substantive answer", "Another answer"]
    scenario = {**BASE_SCENARIO, "probes": probes}

    runner, _ = make_runner(responses, max_turns=5)
    result = runner.run(scenario)

    # Should stop because probes are exhausted, not because of empty response
    assert result["metadata"]["stop_reason"] != "condition_met"

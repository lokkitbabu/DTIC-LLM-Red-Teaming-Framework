"""
Determinism tests for the Ollama adapter (REQ-006).

Verifies that running the same scenario twice with the same seed and params
produces identical outputs — i.e., the adapter correctly passes the seed
through to the model and the system is reproducible.

**Validates: Requirements REQ-006**
"""

import pytest
from unittest.mock import MagicMock, patch, call

from models.ollama_adapter import OllamaAdapter


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FIXED_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "seed": 42,
}

PROMPT = "You are Ahmed. Tell me about yourself."

FIXED_RESPONSE = "I am Ahmed, a civil engineer from Mosul."


def _make_mock_response(text: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": text}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ===========================================================================
# 32.1  Same scenario twice → identical outputs (mocked Ollama HTTP)
# ===========================================================================


class TestOllamaDeterminism:
    """
    Verify that two calls to OllamaAdapter.generate with identical prompt,
    params (including seed) produce identical outputs.

    The Ollama HTTP endpoint is mocked to return a fixed response for a given
    seed, simulating the real deterministic behaviour of Ollama's seed param.
    """

    def test_same_seed_produces_identical_output(self):
        """
        Two generate() calls with the same prompt and seed must return the
        same string.
        """
        with patch("models.ollama_adapter.requests.post",
                   return_value=_make_mock_response(FIXED_RESPONSE)):
            adapter = OllamaAdapter(model_name="mistral")
            result_1 = adapter.generate(PROMPT, FIXED_PARAMS)
            result_2 = adapter.generate(PROMPT, FIXED_PARAMS)

        assert result_1 == result_2, (
            f"Expected identical outputs for same seed, got:\n"
            f"  run 1: {result_1!r}\n"
            f"  run 2: {result_2!r}"
        )

    def test_seed_is_included_in_payload_both_runs(self):
        """
        The seed value must appear in the options payload on every call so
        that the model can honour it.
        """
        captured_payloads = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured_payloads.append(json)
            return _make_mock_response(FIXED_RESPONSE)

        with patch("models.ollama_adapter.requests.post", side_effect=fake_post):
            adapter = OllamaAdapter(model_name="mistral")
            adapter.generate(PROMPT, FIXED_PARAMS)
            adapter.generate(PROMPT, FIXED_PARAMS)

        assert len(captured_payloads) == 2
        for i, payload in enumerate(captured_payloads):
            assert payload["options"]["seed"] == FIXED_PARAMS["seed"], (
                f"Run {i + 1}: seed missing or wrong in payload options. "
                f"Got: {payload['options']}"
            )

    def test_same_params_sent_on_both_runs(self):
        """
        All generation params (temperature, top_p, num_predict, seed) must be
        identical across both runs.
        """
        captured_options = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured_options.append(json["options"])
            return _make_mock_response(FIXED_RESPONSE)

        with patch("models.ollama_adapter.requests.post", side_effect=fake_post):
            adapter = OllamaAdapter(model_name="mistral")
            adapter.generate(PROMPT, FIXED_PARAMS)
            adapter.generate(PROMPT, FIXED_PARAMS)

        assert captured_options[0] == captured_options[1], (
            f"Options differed between runs:\n"
            f"  run 1: {captured_options[0]}\n"
            f"  run 2: {captured_options[1]}"
        )

    def test_different_seeds_would_differ(self):
        """
        Sanity check: if the mock returns different responses for different
        seeds (as a real model would), the outputs differ.
        This confirms the test infrastructure correctly distinguishes seeds.
        """
        responses_by_seed = {
            42: "Response with seed 42",
            99: "Response with seed 99",
        }

        def seed_aware_post(url, json=None, timeout=None, **kwargs):
            seed = json["options"]["seed"]
            text = responses_by_seed.get(seed, "unknown seed response")
            return _make_mock_response(text)

        with patch("models.ollama_adapter.requests.post", side_effect=seed_aware_post):
            adapter = OllamaAdapter(model_name="mistral")
            result_seed_42 = adapter.generate(PROMPT, {**FIXED_PARAMS, "seed": 42})
            result_seed_99 = adapter.generate(PROMPT, {**FIXED_PARAMS, "seed": 99})

        assert result_seed_42 != result_seed_99, (
            "Expected different outputs for different seeds"
        )

    def test_full_scenario_run_twice_identical(self):
        """
        Simulate running a full scenario twice end-to-end with the Ollama
        adapter (mocked). Both runs must produce the same sequence of
        model responses when given the same seed.
        """
        from runner.conversation_runner import ConversationRunner

        scenario = {
            "scenario_id": "determinism_test_001",
            "model": "mistral",
            "identity": {
                "name": "Ahmed",
                "background": "Civil engineer from Mosul",
                "persona": "Thoughtful and reserved",
                "language_style": "Formal",
            },
            "objective": "Test determinism",
            "constraints": ["Stay in character"],
            "params": FIXED_PARAMS,
            "opening_message": "Tell me about yourself.",
            "probes": ["What are your thoughts on the current situation?"],
        }

        # Fixed sequence of responses the mock will return
        mock_sequence = [
            "I am Ahmed, a civil engineer from Mosul.",
            "The situation is difficult but we persevere.",
        ]

        def make_sequential_mock():
            call_count = {"n": 0}

            def fake_post(url, json=None, timeout=None, **kwargs):
                idx = call_count["n"] % len(mock_sequence)
                call_count["n"] += 1
                return _make_mock_response(mock_sequence[idx])

            return fake_post

        # Run 1
        with patch("models.ollama_adapter.requests.post",
                   side_effect=make_sequential_mock()):
            adapter_1 = OllamaAdapter(model_name="mistral")
            runner_1 = ConversationRunner(adapter_1, max_turns=5)
            run_data_1 = runner_1.run(scenario)

        # Run 2 — identical setup
        with patch("models.ollama_adapter.requests.post",
                   side_effect=make_sequential_mock()):
            adapter_2 = OllamaAdapter(model_name="mistral")
            runner_2 = ConversationRunner(adapter_2, max_turns=5)
            run_data_2 = runner_2.run(scenario)

        # Extract model responses from both runs
        responses_1 = [
            t["text"] for t in run_data_1["conversation"] if t["speaker"] == "model"
        ]
        responses_2 = [
            t["text"] for t in run_data_2["conversation"] if t["speaker"] == "model"
        ]

        assert responses_1 == responses_2, (
            f"Model responses differed between runs:\n"
            f"  run 1: {responses_1}\n"
            f"  run 2: {responses_2}"
        )
        assert len(responses_1) > 0, "No model responses were captured"

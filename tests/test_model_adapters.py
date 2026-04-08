"""
Unit tests for model adapters (REQ-005).

Covers:
  - 23.1  EnvironmentError raised when API key is missing
  - 23.2  Ollama adapter constructs the correct HTTP payload
  - 23.3  Output parsing for each adapter (mocked HTTP / SDK layer)
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "temperature": 0.5,
    "top_p": 0.8,
    "max_tokens": 256,
    "seed": 7,
}

PROMPT = "Hello, world!"


# ===========================================================================
# 23.1  EnvironmentError when API key is missing
# ===========================================================================


class TestMissingApiKeys:
    """Each adapter that requires an API key must raise EnvironmentError on
    instantiation when the relevant env-var is absent."""

    def test_openai_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from models.openai_adapter import OpenAIAdapter
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            OpenAIAdapter()

    def test_anthropic_missing_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from models.anthropic_adapter import AnthropicAdapter
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            AnthropicAdapter()

    def test_grok_missing_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from models.grok_adapter import GrokAdapter
        with pytest.raises(EnvironmentError, match="XAI_API_KEY"):
            GrokAdapter()

    def test_hf_missing_key(self, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        from models.hf_adapter import HuggingFaceAdapter
        with pytest.raises(EnvironmentError, match="HF_API_TOKEN"):
            HuggingFaceAdapter(model_name="gpt2")

    def test_ollama_requires_no_key(self):
        """Ollama is local — instantiation must succeed without any env-var."""
        from models.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter()
        assert adapter is not None


# ===========================================================================
# 23.2  Ollama adapter constructs the correct payload
# ===========================================================================


class TestOllamaPayload:
    """Verify the exact JSON body sent to the Ollama HTTP endpoint."""

    def test_payload_structure(self):
        from models.ollama_adapter import OllamaAdapter

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch("models.ollama_adapter.requests.post", return_value=mock_response) as mock_post:
            adapter = OllamaAdapter(model_name="mistral")
            adapter.generate(PROMPT, DEFAULT_PARAMS)

            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

        assert payload["model"] == "mistral"
        assert payload["prompt"] == PROMPT
        assert payload["stream"] is False
        opts = payload["options"]
        assert opts["temperature"] == DEFAULT_PARAMS["temperature"]
        assert opts["top_p"] == DEFAULT_PARAMS["top_p"]
        assert opts["num_predict"] == DEFAULT_PARAMS["max_tokens"]
        assert opts["seed"] == DEFAULT_PARAMS["seed"]

    def test_payload_url(self):
        from models.ollama_adapter import OllamaAdapter

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch("models.ollama_adapter.requests.post", return_value=mock_response) as mock_post:
            adapter = OllamaAdapter(model_name="llama3", base_url="http://localhost:11434")
            adapter.generate(PROMPT, DEFAULT_PARAMS)

            args, _ = mock_post.call_args
            assert args[0] == "http://localhost:11434/api/generate"

    def test_payload_default_params(self):
        """When params are empty, defaults must be applied."""
        from models.ollama_adapter import OllamaAdapter

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "default"}
        mock_response.raise_for_status = MagicMock()

        with patch("models.ollama_adapter.requests.post", return_value=mock_response) as mock_post:
            adapter = OllamaAdapter()
            adapter.generate(PROMPT, {})

            _, kwargs = mock_post.call_args
            opts = kwargs["json"]["options"]

        assert opts["temperature"] == 0.7
        assert opts["top_p"] == 0.9
        assert opts["num_predict"] == 512
        assert opts["seed"] == 42


# ===========================================================================
# 23.3  Output parsing — mocked HTTP / SDK responses
# ===========================================================================


class TestOllamaOutputParsing:
    def test_returns_response_field(self):
        from models.ollama_adapter import OllamaAdapter

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Bonjour!"}
        mock_response.raise_for_status = MagicMock()

        with patch("models.ollama_adapter.requests.post", return_value=mock_response):
            adapter = OllamaAdapter()
            result = adapter.generate(PROMPT, DEFAULT_PARAMS)

        assert result == "Bonjour!"


class TestHuggingFaceOutputParsing:
    def _make_adapter(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "hf-test-token")
        from models.hf_adapter import HuggingFaceAdapter
        return HuggingFaceAdapter(model_name="gpt2")

    def test_parses_list_response(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_response = MagicMock()
        mock_response.json.return_value = [{"generated_text": "Hello from HF"}]
        mock_response.raise_for_status = MagicMock()

        with patch("models.hf_adapter.requests.post", return_value=mock_response):
            result = adapter.generate(PROMPT, DEFAULT_PARAMS)

        assert result == "Hello from HF"

    def test_parses_empty_generated_text(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_response = MagicMock()
        mock_response.json.return_value = [{"generated_text": ""}]
        mock_response.raise_for_status = MagicMock()

        with patch("models.hf_adapter.requests.post", return_value=mock_response):
            result = adapter.generate(PROMPT, DEFAULT_PARAMS)

        assert result == ""

    def test_hf_payload_structure(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_response = MagicMock()
        mock_response.json.return_value = [{"generated_text": "ok"}]
        mock_response.raise_for_status = MagicMock()

        with patch("models.hf_adapter.requests.post", return_value=mock_response) as mock_post:
            adapter.generate(PROMPT, DEFAULT_PARAMS)
            _, kwargs = mock_post.call_args

        assert kwargs["json"]["inputs"] == PROMPT
        p = kwargs["json"]["parameters"]
        assert p["temperature"] == DEFAULT_PARAMS["temperature"]
        assert p["top_p"] == DEFAULT_PARAMS["top_p"]
        assert p["max_new_tokens"] == DEFAULT_PARAMS["max_tokens"]
        assert p["return_full_text"] is False


class TestOpenAIOutputParsing:
    def _make_adapter(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("models.openai_adapter.OpenAI"):
            from models.openai_adapter import OpenAIAdapter
            return OpenAIAdapter()

    def test_returns_message_content(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_response = MagicMock()
        mock_response.output_text = "OpenAI says hi"

        adapter.client.responses.create.return_value = mock_response
        result = adapter.generate(PROMPT, DEFAULT_PARAMS)

        assert result == "OpenAI says hi"

    def test_passes_params_to_sdk(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_response = MagicMock()
        mock_response.output_text = "response"

        adapter.client.responses.create.return_value = mock_response
        adapter.generate(PROMPT, DEFAULT_PARAMS)

        call_kwargs = adapter.client.responses.create.call_args[1]
        assert call_kwargs["temperature"] == DEFAULT_PARAMS["temperature"]
        assert call_kwargs["top_p"] == DEFAULT_PARAMS["top_p"]
        assert call_kwargs["max_output_tokens"] == DEFAULT_PARAMS["max_tokens"]
        assert call_kwargs["seed"] == DEFAULT_PARAMS["seed"]
        assert call_kwargs["input"] == PROMPT


class TestAnthropicOutputParsing:
    def _make_adapter(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        with patch("models.anthropic_adapter.anthropic.Anthropic"):
            from models.anthropic_adapter import AnthropicAdapter
            return AnthropicAdapter()

    def test_returns_first_content_text(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_block = MagicMock()
        mock_block.text = "Anthropic says hi"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        adapter.client.messages.create.return_value = mock_response
        result = adapter.generate(PROMPT, DEFAULT_PARAMS)

        assert result == "Anthropic says hi"

    def test_passes_params_to_sdk(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_block = MagicMock()
        mock_block.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        adapter.client.messages.create.return_value = mock_response
        adapter.generate(PROMPT, DEFAULT_PARAMS)

        call_kwargs = adapter.client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == DEFAULT_PARAMS["temperature"]
        assert call_kwargs["max_tokens"] == DEFAULT_PARAMS["max_tokens"]
        assert call_kwargs["messages"] == [{"role": "user", "content": PROMPT}]


class TestGrokOutputParsing:
    def _make_adapter(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        with patch("models.grok_adapter.OpenAI"):
            from models.grok_adapter import GrokAdapter
            return GrokAdapter()

    def test_returns_message_content(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_message = MagicMock()
        mock_message.content = "Grok says hi"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        adapter.client.chat.completions.create.return_value = mock_completion
        result = adapter.generate(PROMPT, DEFAULT_PARAMS)

        assert result == "Grok says hi"

    def test_passes_params_to_sdk(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)

        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        adapter.client.chat.completions.create.return_value = mock_completion
        adapter.generate(PROMPT, DEFAULT_PARAMS)

        call_kwargs = adapter.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == DEFAULT_PARAMS["temperature"]
        assert call_kwargs["top_p"] == DEFAULT_PARAMS["top_p"]
        assert call_kwargs["max_tokens"] == DEFAULT_PARAMS["max_tokens"]
        assert call_kwargs["messages"] == [{"role": "user", "content": PROMPT}]

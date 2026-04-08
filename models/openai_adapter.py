import os
from openai import OpenAI
from models.base import ModelAdapter

# Reasoning models that don't support temperature/top_p sampling params
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI Responses API (gpt-4o, gpt-5.4, etc.)."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, timeout=60.0)

    def generate(self, prompt: str, params: dict) -> str:
        kwargs = dict(
            model=self.model_name,
            input=prompt,
            max_output_tokens=max(16, params.get("max_tokens", 512)),
        )

        # Reasoning models (gpt-5.x, o1, o3, o4) reject temperature and top_p
        if not _is_reasoning_model(self.model_name):
            kwargs["temperature"] = params.get("temperature", 0.7)
            kwargs["top_p"] = params.get("top_p", 0.9)

        response = self.client.responses.create(**kwargs)
        return response.output_text

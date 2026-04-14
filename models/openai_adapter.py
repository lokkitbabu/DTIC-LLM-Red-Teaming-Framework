import os
import re
import time
from openai import OpenAI, RateLimitError
from models.base import ModelAdapter

# Reasoning models that don't support temperature/top_p sampling params
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_MAX_RETRIES = 5
_BASE_WAIT = 5.0   # seconds


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _retry_wait(error_message: str, attempt: int) -> float:
    """Extract retry-after from error message or use exponential backoff."""
    match = re.search(r"try again in ([0-9.]+)s", error_message)
    if match:
        return float(match.group(1)) + 0.5   # add small buffer
    return _BASE_WAIT * (2 ** attempt)       # 5s, 10s, 20s, 40s, 80s


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI API with automatic rate-limit retry."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, timeout=120.0)

    def generate(self, prompt: str, params: dict) -> str:
        kwargs = dict(
            model=self.model_name,
            input=prompt,
            max_output_tokens=max(16, params.get("max_tokens", 512)),
        )
        if not _is_reasoning_model(self.model_name):
            kwargs["temperature"] = params.get("temperature", 0.7)
            kwargs["top_p"] = params.get("top_p", 0.9)

        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.responses.create(**kwargs)
                return response.output_text

            except RateLimitError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _retry_wait(str(e), attempt)
                print(
                    f"    [openai/{self.model_name}] rate limit — "
                    f"waiting {wait:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})",
                    flush=True,
                )
                time.sleep(wait)

        return ""  # unreachable but satisfies type checker

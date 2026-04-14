import os
import re
import time
from openai import OpenAI, RateLimitError, APIStatusError
from models.base import ModelAdapter

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_MAX_RETRIES = 8
_MIN_WAIT = 5.0
_MAX_WAIT = 90.0


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _is_rate_limit(exc: Exception) -> bool:
    """Match both RateLimitError and any 429 APIStatusError."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return True
    return False


def _retry_wait(error_message: str, attempt: int) -> float:
    """Parse suggested wait from error message, then clamp to bounds."""
    match = re.search(r"try again in ([0-9.]+)s", error_message)
    suggested = float(match.group(1)) + 1.0 if match else _MIN_WAIT * (2 ** attempt)
    return min(max(suggested, _MIN_WAIT), _MAX_WAIT)


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

        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.responses.create(**kwargs)
                return response.output_text

            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                last_exc = e
                wait = _retry_wait(str(e), attempt)
                print(
                    f"    [openai/{self.model_name}] rate limit — "
                    f"waiting {wait:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})",
                    flush=True,
                )
                time.sleep(wait)

        raise last_exc

import os
import re
import time
from openai import OpenAI, RateLimitError
from models.base import ModelAdapter

_MAX_RETRIES = 5
_BASE_WAIT = 5.0


def _retry_wait(error_message: str, attempt: int) -> float:
    match = re.search(r"try again in ([0-9.]+)s", error_message)
    if match:
        return float(match.group(1)) + 0.5
    return _BASE_WAIT * (2 ** attempt)


class GrokAdapter(ModelAdapter):
    """Adapter for xAI Grok API (OpenAI-compatible) with rate-limit retry."""

    def __init__(self, model_name: str = "grok-beta"):
        self.model_name = model_name
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise EnvironmentError("XAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=120.0)

    def generate(self, prompt: str, params: dict) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 0.9),
                    max_tokens=params.get("max_tokens", 512),
                )
                return response.choices[0].message.content or ""

            except RateLimitError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _retry_wait(str(e), attempt)
                print(
                    f"    [grok/{self.model_name}] rate limit — "
                    f"waiting {wait:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})",
                    flush=True,
                )
                time.sleep(wait)

        return ""

"""
Adapter for Mistral AI's OpenAI-compatible chat completions API.
Includes automatic rate-limit retry with exponential backoff.
"""

import os
import re
import time
from openai import OpenAI, RateLimitError, APIStatusError
from models.base import ModelAdapter

_MAX_RETRIES = 8
_MIN_WAIT = 10.0   # Mistral rate limits recover slower than OpenAI
_MAX_WAIT = 120.0


def _retry_wait(error_message: str, attempt: int) -> float:
    match = re.search(r"try again in ([0-9.]+)s", error_message)
    suggested = float(match.group(1)) + 2.0 if match else _MIN_WAIT * (2 ** attempt)
    return min(max(suggested, _MIN_WAIT), _MAX_WAIT)


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return True
    return False


class MistralAdapter(ModelAdapter):
    """Adapter for Mistral AI serverless inference with rate-limit retry."""

    BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=120.0)

    def generate(self, prompt: str, params: dict) -> str:
        messages = self._parse_prompt(prompt)
        last_exc = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 0.9),
                    max_tokens=max(16, params.get("max_tokens", 512)),
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                last_exc = e
                wait = _retry_wait(str(e), attempt)
                print(
                    f"    [mistral/{self.model_name}] rate limit — "
                    f"waiting {wait:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})",
                    flush=True,
                )
                time.sleep(wait)

        raise last_exc

    def _parse_prompt(self, prompt: str) -> list[dict]:
        system_content = ""
        user_content = prompt
        if "[SYSTEM]" in prompt:
            after_tag = prompt.split("[SYSTEM]", 1)[1]
            for divider in ("[CONVERSATION]", "[YOUR RESPONSE]", "[YOUR NEXT QUESTION]"):
                if divider in after_tag:
                    idx = after_tag.index(divider)
                    system_content = after_tag[:idx].strip()
                    user_content = after_tag[idx:].strip()
                    break
            else:
                system_content = after_tag.strip()
                user_content = ""
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_content:
            messages.append({"role": "user", "content": user_content})
        return messages

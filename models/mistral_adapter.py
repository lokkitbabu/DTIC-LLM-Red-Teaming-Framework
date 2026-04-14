"""
Adapter for Mistral AI's OpenAI-compatible chat completions API.
Supports mistral-large-latest, mistral-small-latest, mistral-medium-latest, etc.
"""

import os
from openai import OpenAI
from models.base import ModelAdapter


class MistralAdapter(ModelAdapter):
    """Adapter for Mistral AI serverless inference (chat.completions)."""

    BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=120.0)

    def generate(self, prompt: str, params: dict) -> str:
        messages = self._parse_prompt(prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            max_tokens=max(16, params.get("max_tokens", 512)),
        )
        return response.choices[0].message.content or ""

    def _parse_prompt(self, prompt: str) -> list[dict]:
        """Split [SYSTEM] / [CONVERSATION] prompt into messages list."""
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

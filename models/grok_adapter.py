import os
from openai import OpenAI
from models.base import ModelAdapter


class GrokAdapter(ModelAdapter):
    """Adapter for xAI Grok API (OpenAI-compatible endpoint)."""

    def __init__(self, model_name: str = "grok-beta"):
        self.model_name = model_name
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise EnvironmentError("XAI_API_KEY environment variable not set")
        # Grok uses an OpenAI-compatible endpoint
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    def generate(self, prompt: str, params: dict) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            max_tokens=params.get("max_tokens", 512),
        )
        return response.choices[0].message.content

import os
import anthropic
from models.base import ModelAdapter


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic API (claude-3-5-sonnet, claude-3-opus, etc.)."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        self.model_name = model_name
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, params: dict) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

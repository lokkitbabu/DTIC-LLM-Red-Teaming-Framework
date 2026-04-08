import requests
from models.base import ModelAdapter


class OllamaAdapter(ModelAdapter):
    """Adapter for locally running Ollama models."""

    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str, params: dict) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 0.9),
                "num_predict": params.get("max_tokens", 512),
                "seed": params.get("seed", 42),
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

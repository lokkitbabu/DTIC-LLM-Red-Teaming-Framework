import os
import requests
from models.base import ModelAdapter


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for HuggingFace Inference API."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_token = os.environ.get("HF_API_TOKEN")
        if not self.api_token:
            raise EnvironmentError("HF_API_TOKEN environment variable not set")
        self.base_url = f"https://api-inference.huggingface.co/models/{model_name}"

    def generate(self, prompt: str, params: dict) -> str:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 0.9),
                "max_new_tokens": params.get("max_tokens", 512),
                "return_full_text": False,
            }
        }

        response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        # HF returns a list of generated texts
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "")
        return str(result)

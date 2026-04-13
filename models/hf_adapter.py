import os
from models.base import ModelAdapter


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for HuggingFace Inference API."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._token = os.environ.get("HF_API_TOKEN", "")

    def generate(self, prompt: str, params: dict) -> str:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed. "
                "Add it to requirements.txt or install with: pip install huggingface_hub"
            )
        client = InferenceClient(token=self._token)
        response = client.text_generation(
            prompt,
            model=self.model_name,
            max_new_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
        )
        return response

    def __repr__(self):
        return f"HuggingFaceAdapter(model={self.model_name})"

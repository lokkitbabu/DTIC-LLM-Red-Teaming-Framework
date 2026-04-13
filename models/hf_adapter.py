import os
from huggingface_hub import InferenceClient
from models.base import ModelAdapter


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for Hugging Face Inference API via huggingface_hub InferenceClient.

    Supports:
    - Inference Providers (serverless): pass a model ID, e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    - Inference Endpoints (dedicated): pass the endpoint URL as model_name
    - Provider routing: set HF_PROVIDER env var (e.g. "together", "sambanova", "groq")
    """

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "HF_API_TOKEN (or HF_TOKEN) environment variable not set"
            )
        provider = os.environ.get("HF_PROVIDER")  # optional, e.g. "together"
        self.client = InferenceClient(
            model=model_name,
            token=token,
            provider=provider,  # None → auto-select best available provider
        )

    def generate(self, prompt: str, params: dict) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            seed=params.get("seed"),
        )
        return response.choices[0].message.content

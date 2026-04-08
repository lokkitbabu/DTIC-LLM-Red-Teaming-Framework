from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Abstract base class for all model adapters."""

    @abstractmethod
    def generate(self, prompt: str, params: dict) -> str:
        """
        Generate a response given a prompt string and generation params.

        Args:
            prompt: The full formatted prompt string
            params: Dict with keys like temperature, top_p, max_tokens, seed

        Returns:
            Plain string response from the model
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model={getattr(self, 'model_name', 'unknown')})"

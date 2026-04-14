"""
Adapter for Together AI's OpenAI-compatible chat completions API.
Supports all Together-hosted models via a single unified interface.
"""

import os
from openai import OpenAI
from models.base import ModelAdapter


class TogetherAdapter(ModelAdapter):
    """Adapter for Together AI serverless inference (chat.completions)."""

    BASE_URL = "https://api.together.xyz/v1"

    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise EnvironmentError("TOGETHER_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, timeout=120.0)

    def generate(self, prompt: str, params: dict) -> str:
        messages = self._parse_prompt(prompt)

        # Extra kwargs forwarded to the API (e.g. thinking_budget for Gemma 4)
        extra: dict = {}
        if "thinking_budget" in params:
            extra["thinking_budget"] = params["thinking_budget"]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            max_tokens=max(16, params.get("max_tokens", 512)),
            **extra,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        # --- Observability: log raw response for debugging ---
        raw_len = len(content)
        finish = getattr(choice, "finish_reason", "unknown")
        reasoning = getattr(choice.message, "reasoning_content", None)
        print(
            f"    [together/{self.model_name.split('/')[-1]}] "
            f"finish={finish} raw_len={raw_len} "
            f"has_reasoning={reasoning is not None}",
            flush=True,
        )
        if raw_len < 10:
            # Log first 200 chars of raw content and reasoning so we can diagnose
            print(f"    [RAW content repr]: {repr(content[:200])}", flush=True)
            if reasoning:
                print(f"    [RAW reasoning repr]: {repr(reasoning[:200])}", flush=True)

        # Gemma 4 and other reasoning models wrap output in <think>...</think>.
        # Strip thinking sections and return only the final response.
        content = self._strip_thinking(content)

        # Fallback: if content is empty, try reasoning_content field
        if not content.strip() and reasoning:
            print(f"    [fallback to reasoning_content]", flush=True)
            content = self._strip_thinking(reasoning).strip()

        # Last resort: if still empty, log a clear warning
        if not content.strip():
            print(
                f"    [WARNING] {self.model_name} returned empty response "
                f"(finish={finish}, raw_len={raw_len}). "
                f"Check Together AI playground for this model.",
                flush=True,
            )

        return content

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> or <|think|>...</|think|> blocks."""
        import re
        # Remove standard <think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove pipe-style thinking tokens used by some Gemma variants
        text = re.sub(r"<\|think\|>.*?<\|/think\|>", "", text, flags=re.DOTALL)
        return text.strip()

    def _parse_prompt(self, prompt: str) -> list[dict]:
        """
        Split the flat [SYSTEM] / [CONVERSATION] prompt string into
        the messages list format expected by chat.completions.

        The ConversationRunner produces prompts in the form:
            [SYSTEM]
            <identity / constraints>

            [CONVERSATION]
            <history>

            [YOUR RESPONSE]  or  [YOUR NEXT QUESTION]

        We extract the SYSTEM block as the system message and pass
        the rest as the user message.
        """
        system_content = ""
        user_content = prompt

        if "[SYSTEM]" in prompt:
            # Split on the first double-newline after [SYSTEM] header
            after_tag = prompt.split("[SYSTEM]", 1)[1]

            # Find where the next major block starts
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

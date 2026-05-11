"""Cloud LLM provider — Groq API (Llama 4 Scout)."""

import asyncio

from app.llm.base import LLMProvider
from app.llm.config import LLMSettings


class CloudLLMProvider(LLMProvider):
    """Groq API (Llama 4 Scout) — cloud fast-reasoning LLM provider.

    Uses the Groq SDK for ultra-fast inference.
    Requires: pip install justsay-backend[cloud]
    """

    def __init__(self, settings: LLMSettings):
        self._settings = settings
        self._client = None

    @property
    def model_name(self) -> str:
        return f"groq/{self._settings.groq_model}"

    def _get_client(self):
        if self._client is None:
            if not self._settings.groq_api_key:
                raise RuntimeError(
                    "Groq API key is missing. Go to Settings → Keys and add your key."
                )
            from groq import Groq

            self._client = Groq(api_key=self._settings.groq_api_key)
        return self._client

    async def process(self, text: str, system_prompt: str, temperature: float = 0.1) -> str:
        client = self._get_client()

        result = await asyncio.to_thread(
            self._call_groq, client, self._settings.groq_model, text, system_prompt, temperature
        )

        return result.strip() if result else ""

    @staticmethod
    def _call_groq(client, model: str, text: str, system_prompt: str, temperature: float) -> str:
        """Isolated SDK call — mockable in tests without installing groq."""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content

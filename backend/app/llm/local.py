"""Local LLM provider — Ollama (Gemma 3 4B)."""

import asyncio
import logging

from app.llm.base import LLMProvider
from app.llm.config import LLMSettings

log = logging.getLogger(__name__)


class LocalLLMProvider(LLMProvider):
    """Ollama (Gemma 3 4B) — local privacy-first LLM provider.

    Requires Ollama running locally with the configured model pulled.
    Requires: pip install justsay-backend[local]
    """

    def __init__(self, settings: LLMSettings):
        self._settings = settings
        self._client = None

    @property
    def model_name(self) -> str:
        return f"ollama/{self._settings.ollama_model}"

    def _get_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self._settings.ollama_host)
        return self._client

    async def process(self, text: str, system_prompt: str, temperature: float = 0.1) -> str:
        client = self._get_client()

        result = await asyncio.to_thread(
            self._call_ollama, client, self._settings.ollama_model, text, system_prompt, temperature
        )

        return result.strip() if result else ""

    def cleanup(self) -> None:
        """Unload model from Ollama VRAM/RAM and close client."""
        if self._client is not None:
            try:
                log.info("Unloading Ollama model %s", self._settings.ollama_model)
                # keep_alive=0 tells Ollama to immediately unload the model from memory
                self._client.generate(
                    model=self._settings.ollama_model,
                    prompt="",
                    keep_alive=0,
                )
                log.info("Ollama model unloaded")
            except Exception as e:
                log.warning("Failed to unload Ollama model: %s", e)
            self._client = None

    @staticmethod
    def _call_ollama(client, model: str, text: str, system_prompt: str, temperature: float) -> str:
        """Isolated SDK call — mockable in tests without installing ollama."""
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]

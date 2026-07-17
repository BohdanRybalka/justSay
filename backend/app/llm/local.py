"""Local LLM provider — Ollama (Gemma 3 4B)."""

import asyncio
import logging

from app.llm.base import LLMProvider
from app.llm.config import LLMSettings
from app.llm.tasks import DEFAULT_TASK, get_task_profile

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

    async def process(self, text: str, system_prompt: str, task: str = DEFAULT_TASK) -> str:
        client = self._get_client()
        profile = get_task_profile(task)

        result = await asyncio.to_thread(
            self._call_ollama,
            client,
            self._settings.ollama_model,
            text,
            system_prompt,
            temperature=profile.temperature,
            top_p=profile.top_p,
            max_tokens=profile.max_tokens,
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
    def _call_ollama(
        client, model: str, text: str, system_prompt: str,
        temperature: float, top_p: float, max_tokens: int,
    ) -> str:
        """Isolated SDK call — mockable in tests without installing ollama.

        ``think=False`` is unconditional: Qwen3 is a hybrid-reasoning model
        whose reasoning text otherwise arrives in a separate ``thinking``
        field ahead of ``content``, competing with it for the ``num_predict``
        budget once a cap is in place (see ADR 010).
        """
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
            think=False,
        )
        message = response["message"]
        content = message["content"]
        if not content and message.get("thinking"):
            log.warning(
                "Ollama returned only 'thinking' with empty content for model=%s — "
                "verify think=False is honored by this server/model",
                model,
            )
        return content

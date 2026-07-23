"""Local embedding provider — Ollama ``nomic-embed-text``.

``is_model_available`` is a thin wrapper over
``app.llm.local_setup._list_models``, a private (underscore-prefixed)
helper. This is a documented internal-API boundary crossing
(``app.embeddings.local`` → ``app.llm.local_setup``), accepted per the
plan's Risks section rather than promoted to a public helper — a residual
maintenance risk if ``local_setup.py``'s internals shift later.
"""

import asyncio
import logging

import httpx

from app.llm.config import LLMSettings
from app.llm.local_setup import _CONNECT_TIMEOUT, _READ_TIMEOUT, _list_models

log = logging.getLogger(__name__)


async def is_model_available(llm: LLMSettings, model: str) -> bool:
    """True if Ollama at ``llm.ollama_host`` reports ``model`` as pulled.

    Any connectivity failure (Ollama not running, unreachable host) is
    treated as "not available" — the caller's job is to disable the
    feature gracefully, not to distinguish failure modes here.
    """
    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=_READ_TIMEOUT
    )
    try:
        async with httpx.AsyncClient(base_url=llm.ollama_host, timeout=timeout) as client:
            _, downloaded, _ = await _list_models(client, model)
            return downloaded
    except Exception:
        return False


class LocalEmbeddingProvider:
    """Ollama nomic-embed-text — local privacy-first embedding provider.

    Requires Ollama running locally with the configured model pulled.
    Requires: pip install justsay-backend[local-llm]
    """

    def __init__(self, ollama_host: str, model: str):
        self._host = ollama_host
        self._model = model
        self._client = None

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    def _get_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self._host)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        return await asyncio.to_thread(self._call_embed, client, self._model, text)

    def cleanup(self) -> None:
        """Unload nomic-embed-text from Ollama's memory. The project's stated
        8 GB unified-memory Local-mode target platform (CLAUDE.md) makes
        leaving an embedding model resident indefinitely a real, not
        hypothetical, cost."""
        if self._client is not None:
            try:
                log.info("Unloading Ollama embedding model %s", self._model)
                # keep_alive=0 tells Ollama to immediately unload the model from memory
                self._client.embeddings(model=self._model, prompt="", keep_alive=0)
                log.info("Ollama embedding model unloaded")
            except Exception as e:
                log.warning("Failed to unload Ollama embedding model: %s", e)
            self._client = None

    @staticmethod
    def _call_embed(client, model: str, text: str) -> list[float]:
        """Isolated SDK call — mockable in tests without installing ollama."""
        response = client.embeddings(model=model, prompt=text)
        return list(response["embedding"])

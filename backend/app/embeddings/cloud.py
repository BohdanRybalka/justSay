"""Cloud embedding provider — Gemini embeddings via google-genai.

Reuses ``settings.stt.gemini_api_key``, the key already present for cloud STT;
there is no separate embeddings key (ADR 001).

The SDK call shape: ``client.models.embed_content(model=..., contents=text)``
returns an ``EmbedContentResponse`` with ``.embeddings: list[ContentEmbedding]``,
each carrying ``.values: list[float]``.
"""

import asyncio

from app.core.constants import GEMINI_EMBEDDING_TIMEOUT_SECONDS
from app.core.errors import ConfigurationError


class CloudEmbeddingProvider:
    """Gemini cloud embedding provider.

    The model id comes from ``EmbeddingSettings.cloud_model`` (ADR 001).
    Requires: pip install justsay-backend[cloud]
    """

    def __init__(self, gemini_api_key: str, model: str):
        self._api_key = gemini_api_key
        self._model = model
        self._client = None

    @property
    def model_name(self) -> str:
        return f"gemini/{self._model}"

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise ConfigurationError(
                    "Gemini API key is missing. Go to Settings → Keys and add your key."
                )
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    timeout=int(GEMINI_EMBEDDING_TIMEOUT_SECONDS * 1000)
                ),
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        """One embedding, on a worker that a request timeout can end.

        `asyncio.to_thread` cannot be cancelled, so the client's own
        `GEMINI_EMBEDDING_TIMEOUT_SECONDS` budget is what ends an unanswered embed.
        """
        client = self._get_client()
        return await asyncio.to_thread(self._call_embed, client, self._model, text)

    def cleanup(self) -> None:
        """No-op — the google-genai client holds no persistent local
        resource worth releasing."""

    @staticmethod
    def _call_embed(client, model: str, text: str) -> list[float]:
        """Isolated SDK call — mockable in tests without installing google-genai."""
        response = client.models.embed_content(model=model, contents=text)
        return list(response.embeddings[0].values)

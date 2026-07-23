"""Cloud embedding provider — Gemini ``text-embedding-004`` via google-genai.

Reuses the Gemini API key already used for cloud STT
(``settings.stt.gemini_api_key``) — no separate embeddings key. See
``docs/adr/001-sqlite-vec-embedding-provider-selection.md``.

Call shape verified against the pinned SDK (``google-genai==1.70.0``,
satisfies the ``>=1.0.0`` pin in ``pyproject.toml``) at implementation
time: ``client.models.embed_content(model=..., contents=text)`` returns an
``EmbedContentResponse`` with ``.embeddings: list[ContentEmbedding]``, each
carrying ``.values: list[float]``.
"""

import asyncio


class CloudEmbeddingProvider:
    """Gemini text-embedding-004 — cloud embedding provider.

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
                raise RuntimeError(
                    "Gemini API key is missing. Go to Settings → Keys and add your key."
                )
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
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

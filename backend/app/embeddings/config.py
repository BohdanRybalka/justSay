"""Embedding provider configuration — model names and the Ollama host.

No ``mode`` field: eligibility is derived purely from ``STTSettings.mode``
(see ``resolve_embedding_provider`` in ``app.embeddings``), never a toggle of
its own. No API key field: Cloud embeddings reuse
``settings.stt.gemini_api_key`` (already present for cloud STT) — a second
Gemini key field would be pure duplication. ``ollama_host`` lives here because
the local embedding provider is the only code that reads it. See
``docs/adr/071-semantic-search-keys-on-the-dictation-mode.md`` and
``docs/adr/001-sqlite-vec-embedding-provider-selection.md``.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    cloud_model: str = "gemini-embedding-001"
    local_model: str = "nomic-embed-text"

    ollama_host: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_prefix="JUSTSAY_EMBEDDINGS_", env_file=".env", extra="ignore"
    )

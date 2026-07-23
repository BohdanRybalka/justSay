"""Embedding provider configuration — model names for Cloud/Local embeddings.

No ``mode`` field: eligibility is derived purely from ``STTSettings.mode``
and ``LLMSettings.mode`` (see ``resolve_embedding_provider`` in
``app.embeddings``), never a third user-set toggle. No API key field:
Cloud embeddings reuse ``settings.stt.gemini_api_key`` (already present for
cloud STT) — a second Gemini key field would be pure duplication. See
``docs/adr/001-sqlite-vec-embedding-provider-selection.md``.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    cloud_model: str = "gemini-embedding-001"
    local_model: str = "nomic-embed-text"

    model_config = SettingsConfigDict(
        env_prefix="JUSTSAY_EMBEDDINGS_", env_file=".env", extra="ignore"
    )

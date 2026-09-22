"""Embedding provider configuration — model names and the Ollama host.

No ``mode`` field: eligibility is derived from ``STTSettings.mode`` by
``resolve_embedding_provider``, never a toggle of its own. No API key field:
cloud embeddings reuse ``settings.stt.gemini_api_key``, already present for
cloud STT. ``ollama_host`` lives here because the local embedding provider is
the only code that reads it. See ADR 071 and ADR 001.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    cloud_model: str = "gemini-embedding-001"
    local_model: str = "nomic-embed-text"

    ollama_host: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_prefix="JUSTSAY_EMBEDDINGS_", env_file=".env", extra="ignore"
    )


embedding_settings = EmbeddingSettings()

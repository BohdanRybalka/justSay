"""Application-level config — composition root.

Owns the application-level fields and exposes each feature package's own live
settings instance as a read-only property, so ``settings.stt`` is the very
object ``app.stt.config`` defines rather than a second construction of it. A
property is not a pydantic field, so nothing can rebuild it and the identity
holds without an environment-reading option having to stay just so (ADR 091).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.audio.config import AudioSettings, audio_settings
from app.embeddings.config import EmbeddingSettings, embedding_settings
from app.stt.config import STTSettings, stt_settings


class AppSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 9377
    debug: bool = False

    api_token: str = ""

    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost"]
    )

    model_config = SettingsConfigDict(
        env_prefix="JUSTSAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def stt(self) -> STTSettings:
        """The live instance ``app.stt`` reads and writes."""
        return stt_settings

    @property
    def audio(self) -> AudioSettings:
        """The live instance ``app.audio`` reads and writes."""
        return audio_settings

    @property
    def embeddings(self) -> EmbeddingSettings:
        """The live instance ``app.embeddings`` reads and writes."""
        return embedding_settings


settings = AppSettings()

"""Application-level config — composition root.

Assembles the child-module configs (stt, audio, embeddings) into a single
AppSettings object. Each child Settings reads its own env scope via its own
``env_prefix`` (``JUSTSAY_STT_GEMINI_API_KEY`` → ``settings.stt.gemini_api_key``),
with ``env_nested_delimiter="__"`` configured here as a fallback for the
double-underscore form. Callers keep spelling this module ``app.core.config``,
which re-exports what is defined here (ADR 076).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.audio.config import AudioSettings
from app.embeddings.config import EmbeddingSettings
from app.stt.config import STTSettings


class AppSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 9377
    debug: bool = False

    api_token: str = ""

    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost"]
    )

    stt: STTSettings = Field(default_factory=STTSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    model_config = SettingsConfigDict(
        env_prefix="JUSTSAY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = AppSettings()

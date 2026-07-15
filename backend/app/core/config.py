"""Application-level config — composition root.

Imports child-module configs (stt, llm, audio) and assembles them into a single
AppSettings object. Each child Settings reads its own env scope via its own
``env_prefix`` (e.g. ``JUSTSAY_STT_GEMINI_API_KEY`` → ``settings.stt.gemini_api_key``).
``env_nested_delimiter="__"`` is configured here as a fallback for the double-
underscore form (``JUSTSAY_STT__GEMINI_API_KEY``).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.audio.config import AudioSettings
from app.embeddings.config import EmbeddingSettings
from app.llm.config import LLMSettings
from app.stt.config import STTSettings


class AppSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 9377
    debug: bool = False

    # default_factory ensures each ``AppSettings()`` re-reads env (matters for
    # tests that ``monkeypatch.setenv`` after module import).
    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
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

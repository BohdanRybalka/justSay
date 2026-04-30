"""Application-level config — composition root.

This module intentionally imports from child modules (stt, llm, audio) to assemble
their configs into a single AppSettings object. This is the composition root pattern,
not a circular dependency — child modules never import from core.config.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.stt.config import STTSettings
from app.llm.config import LLMSettings
from app.audio.config import AudioSettings


class AppSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 9377
    debug: bool = False

    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)

    model_config = SettingsConfigDict(
        env_prefix="JUSTSAY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = AppSettings()

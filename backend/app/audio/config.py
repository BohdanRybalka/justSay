from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    sample_rate: int = 16000
    channels: int = 1
    max_duration_seconds: float = 300.0  # 5 minutes
    temp_dir: Path = Field(default_factory=lambda: Path.home() / ".justsay" / "tmp")

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_AUDIO_", env_file=".env", extra="ignore")

    @field_validator("sample_rate")
    @classmethod
    def sample_rate_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("sample_rate must be positive")
        return v

    @field_validator("channels")
    @classmethod
    def channels_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("channels must be positive")
        return v

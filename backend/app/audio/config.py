from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.app_paths import resolve_temp_dir


class AudioSettings(BaseSettings):
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    temp_dir: Path = Field(default_factory=resolve_temp_dir)

    silence_peak_dbfs: float = Field(default=-45.0)
    silence_frame_dbfs: float = Field(default=-50.0)
    silence_min_speech_frames: int = Field(default=5, ge=0)
    silence_min_speech_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    silence_min_analysis_ms: float = Field(default=100.0, ge=0.0)

    silence_vad_enabled: bool = Field(default=True)
    silence_vad_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    silence_vad_min_speech_frames: int = Field(default=5, ge=0)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_AUDIO_", env_file=".env", extra="ignore")

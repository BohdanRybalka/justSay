from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.audio.endpoint_selection import EndpointRole
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

    meeting_block_frames: int = Field(default=1024, gt=0)
    meeting_max_raw_bytes: int = Field(default=700_000_000, gt=0)
    meeting_gap_tolerance_blocks: float = Field(default=1.5, gt=1.0)
    meeting_rate_tolerance: float = Field(default=0.05, gt=0.0, lt=1.0)

    meeting_system_endpoint_role: EndpointRole = Field(default="communications")
    meeting_macos_tap_path: Path | None = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_AUDIO_", env_file=".env", extra="ignore")

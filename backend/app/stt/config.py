from app.core.types import ProviderMode
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class STTSettings(BaseSettings):
    mode: ProviderMode = ProviderMode.CLOUD

    # Cloud: Gemini (long audio, ai_prompt)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Cloud: Groq Whisper (short audio + normal)
    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"

    # Smart routing: audio duration (s) at or below which we use Groq Whisper.
    # Above the threshold, or when style == "ai_prompt", we route to Gemini.
    cloud_routing_threshold: float = 30.0

    # Local: faster-whisper
    whisper_model_size: str = "large-v3-turbo"
    whisper_device: str = "auto"  # auto | cpu | cuda

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_STT_", env_file=".env", extra="ignore")

    @field_validator("cloud_routing_threshold")
    @classmethod
    def _threshold_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("cloud_routing_threshold must be > 0")
        return v

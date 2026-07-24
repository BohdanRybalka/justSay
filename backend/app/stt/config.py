from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.types import ProviderMode

SttEngine = Literal["auto", "groq", "gemini"]


class STTSettings(BaseSettings):
    mode: ProviderMode = ProviderMode.CLOUD

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"

    engine: SttEngine = "auto"

    cloud_routing_threshold: float = Field(default=30.0, gt=0)

    whisper_model_size: str = Field(default="large-v3-turbo", pattern=r"\A[A-Za-z0-9._-]+\z")
    whisper_device: str = "auto"

    initial_prompt: str = Field(default="", max_length=500)

    no_speech_prob_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_STT_", env_file=".env", extra="ignore")

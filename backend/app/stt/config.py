from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from app.core.package_settings import PackageSettings
from app.core.types import ProviderMode

STTEngine = Literal["auto", "groq", "gemini"]


class STTSettings(PackageSettings):
    """Runtime STT configuration -- what the pipeline reads on every request.
    `app.preferences.user_settings.sync_to_runtime` copies the user-editable half
    here on every save, and `cloud_routing_threshold` decides the
    Groq-against-Gemini split alone, never Local mode's boundary (ADR 073).
    """

    mode: ProviderMode = ProviderMode.CLOUD

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"

    engine: STTEngine = "auto"

    cloud_routing_threshold: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Seconds of audio that decide Cloud mode's automatic engine choice: a "
            "recording at or below this length goes to Groq, a longer one to Gemini, "
            "and one in a format Groq cannot read goes to Gemini whatever its length. "
            "Read only while the Cloud engine is left on automatic -- pinning it to "
            "Groq or to Gemini ignores this field, and so does Local mode, whose own "
            "short-clip boundary is a separate fixed number."
        ),
    )

    whisper_model_size: str = Field(default="large-v3-turbo", pattern=r"\A[A-Za-z0-9._-]+\z")
    whisper_device: str = "auto"

    initial_prompt: str = Field(
        default="",
        max_length=500,
        description=(
            "Custom vocabulary biasing transcription, stored whole and never truncated "
            "on disk. Gemini receives all of it inside a fenced glossary block; the "
            "Whisper-family engines receive it unchanged when it fits their send-time "
            "budget, and otherwise whole terms, then whole words, then a character cut "
            "only inside a run the user wrote no boundary into."
        ),
    )

    no_speech_prob_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_STT_", env_file=".env", extra="ignore")


stt_settings = STTSettings()

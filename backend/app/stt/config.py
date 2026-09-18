from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.types import ProviderMode

SttEngine = Literal["auto", "groq", "gemini"]


class STTSettings(BaseSettings):
    """Runtime STT configuration -- what the pipeline reads on every request.

    `app.preferences.user_settings.sync_to_runtime` copies the user-editable
    half of it here on every save, so each shared field is declared twice and
    assigned a third time; `tests/test_settings_defaults_agree.py` pins the
    copies against each other.

    `cloud_routing_threshold` decides the Groq-against-Gemini split in
    `app.stt.routing.get_routed_provider` and nothing else. Local mode's
    short-clip boundary is `app.stt.local.SHORT_CLIP_SECONDS`, which this field
    deliberately does not decide (ADR 073): raising the threshold to send more
    audio to Groq must not also drop local transcription to beam 1 without
    cross-segment context. The two hold the same number and are two different
    facts.
    """

    mode: ProviderMode = ProviderMode.CLOUD

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"

    engine: SttEngine = "auto"

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

    initial_prompt: str = Field(default="", max_length=500)

    no_speech_prob_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_STT_", env_file=".env", extra="ignore")

from typing import Literal

from app.core.types import ProviderMode
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Cloud STT engine override. ``auto`` keeps the duration+style routing.
# ``groq`` and ``gemini`` pin a specific provider — useful when the user wants
# predictable behaviour and is willing to live with the trade-offs (Groq can't
# structure ``ai_prompt`` so a one-shot Gemini fallback kicks in for that style).
SttEngine = Literal["auto", "groq", "gemini"]


class STTSettings(BaseSettings):
    mode: ProviderMode = ProviderMode.CLOUD

    # Cloud: Gemini (long audio, ai_prompt)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Cloud: Groq Whisper (short audio + normal)
    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"

    # Cloud engine override — see SttEngine docstring above.
    engine: SttEngine = "auto"

    # Smart routing: audio duration (s) at or below which we use Groq Whisper.
    # Above the threshold, or when style == "ai_prompt", we route to Gemini.
    cloud_routing_threshold: float = Field(default=30.0, gt=0)

    # Local: faster-whisper
    whisper_model_size: str = "large-v3-turbo"
    whisper_device: str = "auto"  # auto | cpu | cuda

    # User-tunable glossary / vocabulary hint. Threaded into every STT
    # provider with provider-specific semantics:
    #   - faster-whisper: passed as ``initial_prompt`` (decoder conditioning,
    #     hard-truncated to ~224 Whisper tokens — keep short!)
    #   - Groq Whisper: passed as the ``prompt`` parameter (text prefix)
    #   - Gemini: appended to the system prompt inside <glossary>...</glossary>
    #     markers so the model treats it as data, not instructions
    # 500 chars is a conservative ceiling that stays inside Whisper's token
    # budget even for Cyrillic input (~180 tokens at 3 chars/token).
    initial_prompt: str = Field(default="", max_length=500)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_STT_", env_file=".env", extra="ignore")

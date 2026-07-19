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

    # Post-model no-speech gate (spec 033 / docs/adr/019-ten-vad-neural-silence-gate.md).
    # Whisper's own built-in suppression is `no_speech_prob > 0.6 AND
    # avg_logprob < -1.0`, so a CONFIDENTLY-decoded hallucination is never
    # suppressed. 0.6 is upstream Whisper's own default for exactly this
    # quantity -- inherited, not calibrated against our single recording --
    # and thresholding it ALONE (dropping the avg_logprob clause) is the
    # deliberate strengthening. The value compared against it is the MINIMUM
    # across returned segments, so one confident-speech segment keeps the
    # whole transcription.
    no_speech_prob_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_STT_", env_file=".env", extra="ignore")

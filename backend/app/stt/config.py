from pydantic import Field
from pydantic_settings import SettingsConfigDict

from app.core.package_settings import PackageSettings
from app.core.types import ProviderMode


class STTSettings(PackageSettings):
    """Runtime STT configuration -- what the pipeline reads on every request.
    `app.preferences.user_settings.sync_to_runtime` copies the user-editable half
    here on every save. Cloud mode transcribes on Groq; the Gemini key serves
    cloud embeddings only.
    """

    mode: ProviderMode = ProviderMode.CLOUD

    gemini_api_key: str = ""

    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"
    groq_cleanup_model: str = "openai/gpt-oss-20b"

    whisper_model_size: str = Field(default="large-v3-turbo", pattern=r"\A[A-Za-z0-9._-]+\z")
    whisper_device: str = "auto"

    initial_prompt: str = Field(
        default="",
        max_length=500,
        description=(
            "Custom vocabulary biasing transcription, stored whole and never truncated "
            "on disk. The dictation cleanup receives all of it as a glossary; the "
            "Whisper-family engines receive it unchanged when it fits their send-time "
            "budget, and otherwise whole terms, then whole words, then a character cut "
            "only inside a run the user wrote no boundary into."
        ),
    )

    no_speech_prob_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_STT_", env_file=".env", extra="ignore")


stt_settings = STTSettings()

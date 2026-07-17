from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.app_paths import resolve_app_data_root


class AudioSettings(BaseSettings):
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    # Derived from resolve_app_data_root() -- see docs/adr/012-dev-mode-data-directory-isolation.md.
    # An explicit JUSTSAY_AUDIO_TEMP_DIR env var still overrides this default_factory.
    temp_dir: Path = Field(default_factory=lambda: resolve_app_data_root() / "tmp")

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_AUDIO_", env_file=".env", extra="ignore")

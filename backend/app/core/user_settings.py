"""User settings — runtime-mutable preferences stored in ~/.justsay/settings.json.

This is Layer 2 config (user preferences). Layer 1 (secrets/.env) is read-only.
"""

import json
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


SETTINGS_DIR = Path.home() / ".justsay"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


class UserSettings(BaseModel):
    """User-editable settings. Auto-saved to disk on mutation."""

    language: str = "uk"
    shortcut: str = "Ctrl+Alt+KeyV"
    output_dir: str = Field(default_factory=lambda: str(SETTINGS_DIR / "output"))

    # Provider modes
    stt_mode: Literal["cloud", "local"] = "cloud"
    llm_mode: Literal["cloud", "local"] = "cloud"

    # Local STT (faster-whisper)
    whisper_model_size: str = "large-v3-turbo"
    whisper_device: str = "auto"  # auto | cpu | cuda

    # Local LLM (Ollama)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"

    # Audio
    max_recording_seconds: int = 300

    # Transcription
    transcription_style: Literal["normal", "ai_prompt"] = "normal"

    # Smart routing threshold — audio <= this goes to Groq Whisper, above to Gemini.
    cloud_routing_threshold: float = 30.0


_lock = threading.RLock()
_settings: UserSettings | None = None


def get_user_settings() -> UserSettings:
    """Load settings from disk (cached after first load)."""
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = _load()
    return _settings


def update_user_settings(updates: dict) -> UserSettings:
    """Merge partial updates into settings and save to disk."""
    with _lock:
        current = get_user_settings()
        merged = current.model_copy(update=updates)
        _save(merged)
        global _settings
        _settings = merged
    return merged


def _load() -> UserSettings:
    """Load from disk or return defaults."""
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return UserSettings.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            # Corrupt file — reset to defaults
            pass
    return UserSettings()


def sync_to_runtime(us: UserSettings) -> None:
    """Push user settings into the runtime AppSettings objects.

    This bridges Layer 2 (user prefs) → Layer 1 (runtime config)
    for fields that users can change at runtime.
    """
    from app.core.config import settings
    from app.core.types import ProviderMode

    stt_mode = ProviderMode(us.stt_mode)
    llm_mode = ProviderMode(us.llm_mode)

    changed_stt = (
        settings.stt.mode != stt_mode
        or settings.stt.whisper_model_size != us.whisper_model_size
        or settings.stt.whisper_device != us.whisper_device
    )
    changed_llm = (
        settings.llm.mode != llm_mode
        or settings.llm.ollama_model != us.ollama_model
        or settings.llm.ollama_host != us.ollama_host
    )

    # Sync all mutable fields
    settings.stt.mode = stt_mode
    settings.stt.whisper_model_size = us.whisper_model_size
    settings.stt.whisper_device = us.whisper_device
    settings.stt.cloud_routing_threshold = us.cloud_routing_threshold

    settings.llm.mode = llm_mode
    settings.llm.ollama_model = us.ollama_model
    settings.llm.ollama_host = us.ollama_host

    # Clear provider caches if config changed — forces re-init with new settings
    if changed_stt:
        from app.stt import clear_cache as clear_stt_cache
        clear_stt_cache()
    if changed_llm:
        from app.llm import clear_cache as clear_llm_cache
        clear_llm_cache()


def _save(s: UserSettings) -> None:
    """Write settings to disk."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        s.model_dump_json(indent=2),
        encoding="utf-8",
    )

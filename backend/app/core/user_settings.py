"""User settings — runtime-mutable preferences stored in ~/.justsay/settings.json.

This is Layer 2 config (user preferences). Layer 1 (secrets/.env) is read-only.
"""

import json
import logging
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core import history


SETTINGS_DIR = Path.home() / ".justsay"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

log = logging.getLogger(__name__)


class UserSettings(BaseModel):
    """User-editable settings. Auto-saved to disk on mutation."""

    language: str = "uk"
    shortcut: str = "Ctrl+Alt+KeyV"
    output_dir: str = Field(default_factory=lambda: str(SETTINGS_DIR))

    # Provider modes
    stt_mode: Literal["cloud", "local"] = "cloud"
    llm_mode: Literal["cloud", "local"] = "cloud"

    # Cloud STT engine override — Auto keeps the duration+style routing,
    # Groq / Gemini pin a single provider.
    stt_engine: Literal["auto", "groq", "gemini"] = "auto"

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


@dataclass
class UpdateOutcome:
    settings: UserSettings
    warning: str | None = None


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


def update_user_settings(updates: dict) -> UpdateOutcome:
    """Merge partial updates into settings, validate, and save to disk.

    For ``output_dir`` the flow is: validate → relocate history file →
    only on success persist settings.json. On relocate failure the
    in-memory + on-disk settings are left unchanged.
    """
    with _lock:
        current = get_user_settings()
        warning: str | None = None

        if "output_dir" in updates:
            new_dir = _validate_output_dir(updates["output_dir"])

            if new_dir != Path(current.output_dir):
                result, reason = history.relocate(new_dir)
                if result == history.RelocateResult.FAILED:
                    raise RuntimeError(reason or "History relocate failed")
                if result == history.RelocateResult.NEW_ALREADY_HAS_FILE:
                    warning = (
                        "Existing history file at the new location was preserved; "
                        "previous history was not migrated."
                    )

            updates = {**updates, "output_dir": str(new_dir)}

        merged = current.model_copy(update=updates)
        _save(merged)
        global _settings
        _settings = merged
        return UpdateOutcome(settings=merged, warning=warning)


def _validate_output_dir(value: object) -> Path:
    """Validate a candidate output_dir. Raises ValueError on rejection."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("output_dir must be a non-empty string")

    candidate = Path(value).expanduser()

    if not candidate.is_absolute():
        raise ValueError("output_dir must be an absolute path")

    candidate = candidate.resolve(strict=False)

    for forbidden in _FORBIDDEN_PARENTS:
        try:
            inside = candidate.is_relative_to(forbidden)
        except (ValueError, OSError):
            # Different drives on Windows, or unresolvable path — skip this
            # particular forbidden parent and check the rest.
            continue
        if inside:
            raise ValueError(f"output_dir is inside a system directory: {forbidden}")

    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError("output_dir exists but is not a directory")
    elif not candidate.parent.exists():
        raise ValueError("output_dir parent directory does not exist")
    else:
        try:
            candidate.mkdir(parents=False, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Could not create output_dir: {e}") from e

    probe = candidate / f".justsay-write-probe-{uuid.uuid4().hex[:8]}"
    try:
        probe.write_bytes(b"x")
    except OSError as e:
        raise ValueError(f"output_dir is not writable: {e}") from e
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError as e:
            log.warning("Failed to clean up write-probe %s: %s", probe, e)

    return candidate


def _forbidden_parents() -> list[Path]:
    if sys.platform == "win32":
        return [
            Path("C:/Windows"),
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
            Path("C:/ProgramData/Microsoft"),
        ]
    return [
        Path("/etc"),
        Path("/usr"),
        Path("/sys"),
        Path("/proc"),
        Path("/bin"),
        Path("/sbin"),
        Path("/boot"),
        Path("/dev"),
    ]


_FORBIDDEN_PARENTS = _forbidden_parents()


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
    """Push user settings into the runtime AppSettings objects."""
    from app.core.config import settings
    from app.core.types import ProviderMode

    stt_mode = ProviderMode(us.stt_mode)
    llm_mode = ProviderMode(us.llm_mode)

    changed_stt = (
        settings.stt.mode != stt_mode
        or settings.stt.whisper_model_size != us.whisper_model_size
        or settings.stt.whisper_device != us.whisper_device
        or settings.stt.engine != us.stt_engine
    )
    changed_llm = (
        settings.llm.mode != llm_mode
        or settings.llm.ollama_model != us.ollama_model
        or settings.llm.ollama_host != us.ollama_host
    )

    settings.stt.mode = stt_mode
    settings.stt.whisper_model_size = us.whisper_model_size
    settings.stt.whisper_device = us.whisper_device
    settings.stt.cloud_routing_threshold = us.cloud_routing_threshold
    settings.stt.engine = us.stt_engine

    settings.llm.mode = llm_mode
    settings.llm.ollama_model = us.ollama_model
    settings.llm.ollama_host = us.ollama_host

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

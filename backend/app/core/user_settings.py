"""User settings — runtime-mutable preferences stored in ~/.justsay/settings.json.

This is Layer 2 config (user preferences). Layer 1 (secrets/.env) is read-only.
"""

import json
import logging
import re
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core import history
from app.core.app_paths import resolve_app_data_root, resolve_temp_dir


def _settings_dir() -> Path:
    return resolve_app_data_root()


def _settings_path() -> Path:
    return _settings_dir() / "settings.json"


log = logging.getLogger(__name__)


_WHISPER_MODEL_SIZE_CHARS = r"[A-Za-z0-9._-]+"
_WHISPER_MODEL_SIZE_PATTERN = rf"\A{_WHISPER_MODEL_SIZE_CHARS}\z"
_WHISPER_MODEL_SIZE_RE = re.compile(_WHISPER_MODEL_SIZE_CHARS)


class UserSettings(BaseModel):
    """User-editable settings. Auto-saved to disk on mutation."""

    language: str = "uk"
    shortcut: str = "Ctrl+Alt+KeyV"
    output_dir: str = Field(default_factory=lambda: str(_settings_dir()))

    stt_mode: Literal["cloud", "local"] = "cloud"
    llm_mode: Literal["cloud", "local"] = "cloud"

    stt_engine: Literal["auto", "groq", "gemini"] = "auto"

    whisper_model_size: str = Field(
        default="large-v3-turbo", pattern=_WHISPER_MODEL_SIZE_PATTERN
    )
    whisper_device: str = "auto"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"

    cloud_routing_threshold: float = 30.0

    initial_prompt: str = Field(default="", max_length=500)

    gemini_api_key: str = ""
    groq_api_key: str = ""


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

        if "whisper_model_size" in updates:
            _validate_whisper_model_size(updates["whisper_model_size"])

        merged = current.model_copy(update=updates)
        _save(merged)
        global _settings
        _settings = merged
        return UpdateOutcome(settings=merged, warning=warning)


def _validate_whisper_model_size(value: object) -> None:
    """Reject a whisper_model_size that could escape its model cache path.

    Raises ValueError (which ``put_settings`` maps to 400) on any value that
    is not a plain model-size token: it must consist only of ``[A-Za-z0-9._-]``
    (full-string match -- a trailing newline is rejected) and must not contain
    ``..``.
    """
    if (
        not isinstance(value, str)
        or not _WHISPER_MODEL_SIZE_RE.fullmatch(value)
        or ".." in value
    ):
        raise ValueError(
            "whisper_model_size must contain only letters, digits, '.', '_', "
            "or '-', and must not contain '..'"
        )


def _is_inside_scratch(candidate: Path) -> bool:
    """Whether ``candidate`` would put history.db inside the scratch tree.

    The rule is one-way on purpose (ADR 033). In the default layout the
    scratch directory sits *inside* ``output_dir`` -- app-data root holds
    ``history.db`` next to ``tmp/`` -- so a symmetric "these must not nest"
    check would reject every healthy install. What must never happen is the
    reverse: history living under the directory the cleanup endpoint empties.

    Both sides are resolved here rather than at the call site. ``output_dir``
    arrives from settings.json exactly as it was written, so a redirected
    Windows profile or a stray ``..`` segment would otherwise compare unequal
    to a path it in fact denotes -- and the startup repair would return
    "healthy" for a database sitting in the scratch tree.
    """
    try:
        scratch = resolve_temp_dir().resolve(strict=False)
        resolved = candidate.expanduser().resolve(strict=False)
        return resolved == scratch or resolved.is_relative_to(scratch)
    except (ValueError, OSError):
        return False


def _reject_scratch_directory(candidate: Path) -> None:
    """Raise if ``candidate`` is the scratch directory or lives inside it."""
    if _is_inside_scratch(candidate):
        raise ValueError(
            f"output_dir cannot be inside the temporary audio directory "
            f"({resolve_temp_dir()}); files there are deleted by Clear Temp Files"
        )


def repair_scratch_output_dir() -> Path:
    """Startup repair for history that already lives inside the scratch tree.

    Returns the directory history should be opened from. A healthy
    configuration is returned untouched and nothing is written.

    On a broken one the rows are merged into the app-data root, the stored
    ``output_dir`` is rewritten so the repair runs once, and the old file is
    kept aside by ``consolidate_into``. If the merge fails the *old*
    directory is returned deliberately: serving the user's real rows from a
    risky location beats serving an empty database from a safe one, and the
    ownership-scoped cleanup (ADR 033) already stops that location from
    being emptied.

    Must run before ``history.bootstrap`` -- bootstrap opens the connection,
    so a later repair would already have served the wrong file.
    """
    current = Path(get_user_settings().output_dir)
    if not _is_inside_scratch(current):
        return current

    safe = resolve_app_data_root()
    result, reason = history.consolidate_into(current, safe)
    if result == history.ConsolidateResult.FAILED:
        log.error(
            "History sits inside the scratch directory (%s) and could not be moved out: %s. "
            "Continuing from the old location; Clear Temp Files will not touch it.",
            current, reason,
        )
        return current

    with _lock:
        global _settings
        merged = get_user_settings().model_copy(update={"output_dir": str(safe)})
        _save(merged)
        _settings = merged
    log.warning("Moved history out of the scratch directory: %s → %s", current, safe)
    return safe


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
            continue
        if inside:
            raise ValueError(f"output_dir is inside a system directory: {forbidden}")

    _reject_scratch_directory(candidate)

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
    settings_path = _settings_path()
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            return UserSettings.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass
    return UserSettings()


def sync_to_runtime(us: UserSettings) -> bool:
    """Push user settings into the runtime AppSettings objects.

    Returns whether an STT-relevant field changed (the same `changed_stt`
    check that already gates this function's own cache invalidation below),
    so callers that need to react specifically to an STT-relevant change
    (put_settings()'s prewarm gate) don't have to re-derive it themselves.
    """
    from app.core.config import settings
    from app.core.types import ProviderMode

    stt_mode = ProviderMode(us.stt_mode)
    llm_mode = ProviderMode(us.llm_mode)

    changed_stt = (
        settings.stt.mode != stt_mode
        or settings.stt.whisper_model_size != us.whisper_model_size
        or settings.stt.whisper_device != us.whisper_device
        or settings.stt.engine != us.stt_engine
        or settings.stt.initial_prompt != us.initial_prompt
        or (us.gemini_api_key and settings.stt.gemini_api_key != us.gemini_api_key)
        or (us.groq_api_key and settings.stt.groq_api_key != us.groq_api_key)
    )
    changed_llm = (
        settings.llm.mode != llm_mode
        or settings.llm.ollama_model != us.ollama_model
        or settings.llm.ollama_host != us.ollama_host
        or (us.groq_api_key and settings.llm.groq_api_key != us.groq_api_key)
    )

    settings.stt.mode = stt_mode
    settings.stt.whisper_model_size = us.whisper_model_size
    settings.stt.whisper_device = us.whisper_device
    settings.stt.cloud_routing_threshold = us.cloud_routing_threshold
    settings.stt.engine = us.stt_engine
    settings.stt.initial_prompt = us.initial_prompt
    if us.gemini_api_key:
        settings.stt.gemini_api_key = us.gemini_api_key
    if us.groq_api_key:
        settings.stt.groq_api_key = us.groq_api_key
        settings.llm.groq_api_key = us.groq_api_key

    settings.llm.mode = llm_mode
    settings.llm.ollama_model = us.ollama_model
    settings.llm.ollama_host = us.ollama_host

    if changed_stt:
        from app.stt import clear_cache as clear_stt_cache
        clear_stt_cache()
        from app.embeddings import clear_cache as clear_embeddings_cache
        clear_embeddings_cache()
    if changed_llm:
        from app.embeddings import clear_cache as clear_embeddings_cache
        clear_embeddings_cache()

    return bool(changed_stt)


def _save(s: UserSettings) -> None:
    """Write settings to disk."""
    _settings_dir().mkdir(parents=True, exist_ok=True)
    _settings_path().write_text(
        s.model_dump_json(indent=2),
        encoding="utf-8",
    )

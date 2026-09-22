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

from pydantic import BaseModel, Field, ValidationError

from app import embeddings
from app.core.app_paths import resolve_app_data_root, resolve_temp_dir
from app.core.errors import ConfigurationError
from app.core.types import ProviderMode
from app.stt import routing as stt_routing
from app.stt.config import stt_settings
from app.transcripts import relocation


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

    stt_engine: Literal["auto", "groq", "gemini"] = "auto"

    whisper_model_size: str = Field(
        default="large-v3-turbo", pattern=_WHISPER_MODEL_SIZE_PATTERN
    )
    whisper_device: str = "auto"

    ollama_host: str = "http://localhost:11434"

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

    initial_prompt: str = Field(default="", max_length=500)

    gemini_api_key: str = ""
    groq_api_key: str = ""

    meeting_consent_acknowledged: bool = False


@dataclass
class UpdateResult:
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


def update_user_settings(updates: dict) -> UpdateResult:
    """Merge partial updates into settings, validate, and save to disk.

    For ``output_dir``: validate, relocate the history file, then persist ``settings.json``. On a
    relocate failure the in-memory and on-disk settings are left unchanged.
    """
    with _lock:
        current = get_user_settings()
        warning: str | None = None

        if "output_dir" in updates:
            new_dir = _validate_output_dir(updates["output_dir"])

            if new_dir != Path(current.output_dir):
                result, reason = relocation.relocate(new_dir)
                if result == relocation.RelocateOutcome.FAILED:
                    raise RuntimeError(reason or "History relocate failed")
                if result == relocation.RelocateOutcome.NEW_ALREADY_HAS_FILE:
                    warning = (
                        "Existing history file at the new location was preserved; "
                        "previous history was not migrated."
                    )

            updates = {**updates, "output_dir": str(new_dir)}

        if "whisper_model_size" in updates:
            _validate_whisper_model_size(updates["whisper_model_size"])

        merged = UserSettings.model_validate({**current.model_dump(), **updates})
        _save(merged)
        global _settings
        _settings = merged
        return UpdateResult(settings=merged, warning=warning)


def _validate_whisper_model_size(value: object) -> None:
    """Reject a ``whisper_model_size`` that could escape its model cache path.

    Raises ``ConfigurationError`` (400) unless the value matches ``[A-Za-z0-9._-]`` end to end — a
    trailing newline is rejected — and contains no ``..``.
    """
    if (
        not isinstance(value, str)
        or not _WHISPER_MODEL_SIZE_RE.fullmatch(value)
        or ".." in value
    ):
        raise ConfigurationError(
            "whisper_model_size must contain only letters, digits, '.', '_', "
            "or '-', and must not contain '..'"
        )


def _is_inside_scratch(candidate: Path) -> bool:
    """Whether ``candidate`` would put ``history.db`` inside the scratch tree (ADR 033).

    One-way on purpose: the scratch directory sits inside ``output_dir`` in the default layout.
    Both sides are resolved here, so a redirected or ``..``-bearing path still compares equal.
    """
    try:
        scratch = resolve_temp_dir().resolve(strict=False)
        resolved = candidate.expanduser().resolve(strict=False)
        return resolved == scratch or resolved.is_relative_to(scratch)
    except (ValueError, OSError):
        return False


def _reject_scratch_directory(candidate: Path) -> None:
    """Refuse ``candidate`` if it is the scratch directory or lives inside it."""
    if _is_inside_scratch(candidate):
        raise ConfigurationError(
            f"output_dir cannot be inside the temporary audio directory "
            f"({resolve_temp_dir()}); files there are deleted by Clear Temp Files"
        )


def repair_scratch_output_dir() -> Path:
    """Startup repair for history that already lives inside the scratch tree.

    Returns the directory history should be opened from on every path: unchanged when healthy, the
    old one when a merge failed, the app-data root when it succeeded. Run before ``bootstrap``.
    """
    current = Path(get_user_settings().output_dir)
    if not _is_inside_scratch(current):
        return current

    safe = resolve_app_data_root()
    result, reason = relocation.consolidate_into(current, safe)
    if result == relocation.ConsolidateOutcome.FAILED:
        log.error(
            "History sits inside the scratch directory (%s) and could not be moved out: %s. "
            "Continuing from the old location; Clear Temp Files will not touch it.",
            current, reason,
        )
        return current

    with _lock:
        global _settings
        merged = get_user_settings().model_copy(update={"output_dir": str(safe)})
        try:
            _save(merged)
        except Exception:
            log.warning(
                "Moved history out of the scratch directory (%s → %s) but could not "
                "store the new output_dir: this launch reads and writes the moved rows, "
                "and the repair runs again on the next launch, where the merge is a "
                "no-op.",
                current, safe, exc_info=True,
            )
            return safe
        _settings = merged
    log.warning("Moved history out of the scratch directory: %s → %s", current, safe)
    return safe


def _validate_output_dir(value: object) -> Path:
    """Validate a candidate output_dir. Raises ``ConfigurationError`` on rejection."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("output_dir must be a non-empty string")

    candidate = Path(value).expanduser()

    if not candidate.is_absolute():
        raise ConfigurationError("output_dir must be an absolute path")

    candidate = candidate.resolve(strict=False)

    for forbidden in _FORBIDDEN_PARENTS:
        try:
            inside = candidate.is_relative_to(forbidden)
        except (ValueError, OSError):
            continue
        if inside:
            raise ConfigurationError(f"output_dir is inside a system directory: {forbidden}")

    _reject_scratch_directory(candidate)

    if candidate.exists():
        if not candidate.is_dir():
            raise ConfigurationError("output_dir exists but is not a directory")
    elif not candidate.parent.exists():
        raise ConfigurationError("output_dir parent directory does not exist")
    else:
        try:
            candidate.mkdir(parents=False, exist_ok=True)
        except OSError as e:
            raise ConfigurationError(f"Could not create output_dir: {e}") from e

    probe = candidate / f".justsay-write-probe-{uuid.uuid4().hex[:8]}"
    try:
        probe.write_bytes(b"x")
    except OSError as e:
        raise ConfigurationError(f"output_dir is not writable: {e}") from e
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError as e:
            log.warning("Failed to clean up write-probe %s: %s", probe, e)

    return candidate


def _forbidden_parents() -> list[Path]:
    """System roots an ``output_dir`` may not sit inside, in resolved form (ADR 065).

    Resolved because ``_validate_output_dir`` resolves the candidate before comparing, and the two
    sides must live in the same space — on macOS ``/etc`` is a symlink to ``/private/etc``.
    """
    if sys.platform == "win32":
        roots = [
            Path("C:/Windows"),
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
            Path("C:/ProgramData/Microsoft"),
        ]
    elif sys.platform == "darwin":
        roots = [
            Path("/System"),
            Path("/Library"),
            Path("/Applications"),
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/dev"),
            Path("/private/etc"),
        ]
    else:
        roots = [
            Path("/etc"),
            Path("/usr"),
            Path("/sys"),
            Path("/proc"),
            Path("/bin"),
            Path("/sbin"),
            Path("/boot"),
            Path("/dev"),
        ]
    return [root.resolve(strict=False) for root in roots]


_FORBIDDEN_PARENTS = _forbidden_parents()


def _load() -> UserSettings:
    """Load from disk or return defaults."""
    settings_path = _settings_path()
    if not settings_path.exists():
        return UserSettings()
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("Could not read %s; starting from defaults", settings_path, exc_info=True)
        return UserSettings()
    if not isinstance(data, dict):
        log.warning("%s does not hold an object; starting from defaults", settings_path)
        return UserSettings()
    try:
        return UserSettings.model_validate(data)
    except ValidationError as failure:
        return _load_without_rejected_fields(data, failure)


def _load_without_rejected_fields(data: dict, failure: ValidationError) -> UserSettings:
    """Keep every stored field that validates when one of them does not.

    Each field named in ``failure`` falls back to its own default; every other stored value
    survives, so one out-of-range number cannot discard the whole file.
    """
    rejected = {str(error["loc"][0]) for error in failure.errors() if error["loc"]}
    log.warning("Ignoring invalid stored settings, falling back to defaults for: %s",
                ", ".join(sorted(rejected)))
    kept = {name: value for name, value in data.items() if name not in rejected}
    try:
        return UserSettings.model_validate(kept)
    except ValidationError:
        log.warning("Stored settings could not be salvaged; starting from defaults", exc_info=True)
        return UserSettings()


def sync_to_runtime(us: UserSettings) -> bool:
    """Push user settings onto the runtime settings each package owns.

    Returns whether an STT-relevant field changed — the same check that gates this function's own
    cache invalidation — so a caller gating a prewarm does not have to re-derive it.
    """
    from app.embeddings.config import embedding_settings

    stt_mode = ProviderMode(us.stt_mode)

    changed_stt = (
        stt_settings.mode != stt_mode
        or stt_settings.whisper_model_size != us.whisper_model_size
        or stt_settings.whisper_device != us.whisper_device
        or stt_settings.engine != us.stt_engine
        or stt_settings.initial_prompt != us.initial_prompt
        or (us.gemini_api_key and stt_settings.gemini_api_key != us.gemini_api_key)
        or (us.groq_api_key and stt_settings.groq_api_key != us.groq_api_key)
    )
    changed_embeddings = embedding_settings.ollama_host != us.ollama_host

    stt_settings.mode = stt_mode
    stt_settings.whisper_model_size = us.whisper_model_size
    stt_settings.whisper_device = us.whisper_device
    stt_settings.cloud_routing_threshold = us.cloud_routing_threshold
    stt_settings.engine = us.stt_engine
    stt_settings.initial_prompt = us.initial_prompt
    if us.gemini_api_key:
        stt_settings.gemini_api_key = us.gemini_api_key
    if us.groq_api_key:
        stt_settings.groq_api_key = us.groq_api_key

    embedding_settings.ollama_host = us.ollama_host

    if changed_stt:
        stt_routing.clear_cache()
        embeddings.clear_cache()
    if changed_embeddings:
        embeddings.clear_cache()

    return bool(changed_stt)


def _save(s: UserSettings) -> None:
    """Write settings to disk."""
    _settings_dir().mkdir(parents=True, exist_ok=True)
    _settings_path().write_text(
        s.model_dump_json(indent=2),
        encoding="utf-8",
    )

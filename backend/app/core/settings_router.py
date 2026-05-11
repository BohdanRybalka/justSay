"""Settings endpoints — CRUD for user preferences."""

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import history
from app.core.utils import compute_dir_size
from app.core.user_settings import (
    UserSettings,
    get_user_settings,
    update_user_settings,
    sync_to_runtime,
    SETTINGS_DIR,
)


router = APIRouter(prefix="/settings", tags=["Settings"])

_KEY_FIELDS = {"gemini_api_key", "groq_api_key"}
_MASKED_PLACEHOLDER = "***"


def _mask_keys(s: UserSettings) -> UserSettings:
    """Return a copy of s with key fields replaced by the masked placeholder or empty string."""
    return s.model_copy(update={
        f: (_MASKED_PLACEHOLDER if getattr(s, f) else "")
        for f in _KEY_FIELDS
    })


class StorageInfo(BaseModel):
    temp_dir: str
    temp_size_bytes: int
    output_dir: str
    history_path: str
    history_entries: int


class CleanupResult(BaseModel):
    freed_bytes: int


class SettingsUpdateResponse(BaseModel):
    settings: UserSettings
    warning: str | None = None


@router.get("", response_model=UserSettings)
async def get_settings():
    return _mask_keys(get_user_settings())


@router.put("", response_model=SettingsUpdateResponse)
async def put_settings(updates: dict):
    allowed_fields = set(UserSettings.model_fields.keys())
    # Strip the masked placeholder — sending "***" back must not overwrite a stored key.
    filtered = {
        k: v for k, v in updates.items()
        if k in allowed_fields
        and not (k in _KEY_FIELDS and v == _MASKED_PLACEHOLDER)
    }
    try:
        outcome = update_user_settings(filtered)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    sync_to_runtime(outcome.settings)
    return SettingsUpdateResponse(settings=_mask_keys(outcome.settings), warning=outcome.warning)


class CloudKeyStatus(BaseModel):
    gemini_key_set: bool
    groq_key_set: bool


@router.get("/cloud-status", response_model=CloudKeyStatus)
async def cloud_key_status():
    """Whether each Cloud API key is currently active in the runtime config.

    Checks the runtime AppSettings (not UserSettings) so that keys provided
    via .env are correctly reflected even if the user has never opened Settings → Keys.
    """
    from app.core.config import settings as runtime_settings
    return CloudKeyStatus(
        gemini_key_set=bool(runtime_settings.stt.gemini_api_key),
        groq_key_set=bool(runtime_settings.stt.groq_api_key or runtime_settings.llm.groq_api_key),
    )


def _mask_home(p: Path) -> str:
    """Replace user's home prefix with ~ so API responses don't leak it.

    Tries ``relative_to`` first (handles canonical paths). Falls back to a
    case-insensitive prefix match so Windows paths like ``c:\\users\\admin``
    vs ``C:\\Users\\Admin`` still get masked.

    Paths OUTSIDE the home tree (network drives, /Volumes, custom data
    folders) are returned as-is — this masking is a privacy-hygiene measure
    for the common case, not a security boundary. If `Path.home()` itself
    fails (e.g. missing `HOME`/`USERPROFILE` in a sandbox), we degrade to
    returning the raw path rather than 500-ing the endpoint.
    """
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return str(p)
    try:
        rel = p.relative_to(home)
        return "~/" + str(rel).replace("\\", "/")
    except ValueError:
        pass
    s = str(p)
    h = str(home)
    if s.lower().startswith(h.lower()):
        return "~" + s[len(h):].replace("\\", "/")
    return s


@router.get("/storage", response_model=StorageInfo)
async def get_storage_info():
    s = get_user_settings()
    tmp_dir = SETTINGS_DIR / "tmp"
    hpath = history.history_path()
    # All three paths are passed through `_mask_home`. `temp_dir` and the
    # default `history_path` always live under `~/.justsay`, so masking is
    # effectively unconditional there. `output_dir` may be relocated outside
    # the home tree (network drives, external SSDs, custom data folders);
    # in that case `_mask_home` returns the raw path unchanged — privacy
    # hygiene is best-effort, not a hard boundary. `GET /settings` (the
    # canonical settings endpoint that the PUT round-trip reads back from)
    # is intentionally UNCHANGED so the frontend's revert logic in
    # `src/settings/tabs/storage.ts` keeps working with real paths.
    return StorageInfo(
        temp_dir=_mask_home(tmp_dir),
        temp_size_bytes=compute_dir_size(tmp_dir),
        output_dir=_mask_home(Path(s.output_dir)),
        history_path=_mask_home(hpath),
        history_entries=history.get_count(),
    )


@router.post("/cleanup", response_model=CleanupResult)
async def cleanup_temp():
    tmp_dir = SETTINGS_DIR / "tmp"
    freed = compute_dir_size(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    return CleanupResult(freed_bytes=freed)

"""Settings endpoints — CRUD for user preferences."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings as runtime_settings
from app.core.user_settings import (
    UserSettings,
    get_user_settings,
    sync_to_runtime,
    update_user_settings,
)

log = logging.getLogger(__name__)

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
    temp_size_bytes: int


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
    changed_stt = sync_to_runtime(outcome.settings)
    if changed_stt:
        from app.stt.local_setup import maybe_prewarm_local

        maybe_prewarm_local(runtime_settings.stt)
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
    return CloudKeyStatus(
        gemini_key_set=bool(runtime_settings.stt.gemini_api_key),
        groq_key_set=bool(runtime_settings.stt.groq_api_key or runtime_settings.llm.groq_api_key),
    )


_SCRATCH_PREFIXES = ("rec_", "pipeline_")


def _scratch_files(tmp_dir: Path) -> list[Path]:
    """Files in the scratch directory that this app wrote (ADR 033).

    ``rec_*`` comes from the microphone recorder, ``pipeline_*`` from the
    upload path -- the only two producers. Deletion is scoped by ownership
    rather than by location, so anything else found there survives because
    it was never in scope, not because it was added to an exception list.
    A user's ``history.db`` is the case that made this necessary.
    """
    if not tmp_dir.is_dir():
        return []
    return [
        entry
        for entry in tmp_dir.iterdir()
        if entry.is_file() and entry.name.startswith(_SCRATCH_PREFIXES)
    ]


def _scratch_size(tmp_dir: Path) -> int:
    total = 0
    for entry in _scratch_files(tmp_dir):
        try:
            total += entry.stat().st_size
        except OSError:
            continue
    return total


@router.get("/storage", response_model=StorageInfo)
async def get_storage_info():
    return StorageInfo(temp_size_bytes=_scratch_size(runtime_settings.audio.temp_dir))


@router.post("/cleanup", response_model=CleanupResult)
async def cleanup_temp():
    tmp_dir = runtime_settings.audio.temp_dir
    freed = 0
    for entry in _scratch_files(tmp_dir):
        try:
            size = entry.stat().st_size
            entry.unlink()
        except OSError:
            log.warning("Could not remove scratch file %s", entry, exc_info=True)
            continue
        freed += size
    return CleanupResult(freed_bytes=freed)

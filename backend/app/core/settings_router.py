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
    return get_user_settings()


@router.put("", response_model=SettingsUpdateResponse)
async def put_settings(updates: dict):
    allowed_fields = set(UserSettings.model_fields.keys())
    filtered = {k: v for k, v in updates.items() if k in allowed_fields}
    try:
        outcome = update_user_settings(filtered)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    sync_to_runtime(outcome.settings)
    return SettingsUpdateResponse(settings=outcome.settings, warning=outcome.warning)


def _mask_home(p: Path) -> str:
    """Replace user's home prefix with ~ so API responses don't leak it.

    Tries ``relative_to`` first (handles canonical paths). Falls back to a
    case-insensitive prefix match so Windows paths like ``c:\\users\\admin``
    vs ``C:\\Users\\Admin`` still get masked.
    """
    home = Path.home()
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
    return StorageInfo(
        temp_dir=str(tmp_dir),
        temp_size_bytes=compute_dir_size(tmp_dir),
        output_dir=s.output_dir,
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

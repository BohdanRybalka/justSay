"""Settings endpoints — CRUD for user preferences."""

import shutil

from fastapi import APIRouter
from pydantic import BaseModel

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


class CleanupResult(BaseModel):
    freed_bytes: int


@router.get("", response_model=UserSettings)
async def get_settings():
    return get_user_settings()


@router.put("", response_model=UserSettings)
async def put_settings(updates: dict):
    allowed_fields = set(UserSettings.model_fields.keys())
    filtered = {k: v for k, v in updates.items() if k in allowed_fields}
    updated = update_user_settings(filtered)
    sync_to_runtime(updated)
    return updated


@router.get("/storage", response_model=StorageInfo)
async def get_storage_info():
    s = get_user_settings()
    tmp_dir = SETTINGS_DIR / "tmp"
    tmp_size = 0
    if tmp_dir.exists():
        tmp_size = sum(f.stat().st_size for f in tmp_dir.rglob("*") if f.is_file())
    return StorageInfo(temp_dir=str(tmp_dir), temp_size_bytes=tmp_size, output_dir=s.output_dir)


@router.post("/cleanup", response_model=CleanupResult)
async def cleanup_temp():
    tmp_dir = SETTINGS_DIR / "tmp"
    freed = 0
    if tmp_dir.exists():
        freed = sum(f.stat().st_size for f in tmp_dir.rglob("*") if f.is_file())
        shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    return CleanupResult(freed_bytes=freed)

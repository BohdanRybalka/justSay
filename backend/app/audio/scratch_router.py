"""Scratch-directory endpoints — how much the app left there, and reaping it.

Mounted at the same ``/settings`` prefix the preferences router uses, so the
two paths are one surface on the wire. It lives in ``app.audio`` because that
is the package owning ``temp_dir`` and every producer writing into it; reading
the directory from ``app.preferences`` instead would put a package cycle back
(ADR 091).
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.audio.config import audio_settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["Settings"])

_SCRATCH_PREFIXES = ("rec_", "pipeline_", "meeting_")


class StorageInfo(BaseModel):
    temp_size_bytes: int


class CleanupResult(BaseModel):
    freed_bytes: int


def _scratch_files(tmp_dir: Path) -> list[Path]:
    """Files in the scratch directory that this app wrote (ADR 033).

    ``rec_*`` from the microphone recorder, ``pipeline_*`` from the upload path, ``meeting_*`` from
    the meeting recorder. Deletion is scoped by ownership, so anything else found there survives.
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
    return StorageInfo(temp_size_bytes=_scratch_size(audio_settings.temp_dir))


@router.post("/cleanup", response_model=CleanupResult)
async def cleanup_temp():
    tmp_dir = audio_settings.temp_dir
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

"""History API — list, delete, clear transcript history."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.history import (
    HistoryEntry,
    get_entries,
    get_count,
    delete_entry,
    clear_all,
)

router = APIRouter(prefix="/history", tags=["History"])


class HistoryListResponse(BaseModel):
    entries: list[HistoryEntry]
    total: int


class ClearResult(BaseModel):
    deleted: int


@router.get("", response_model=HistoryListResponse)
async def list_history(limit: int = 50, offset: int = 0):
    return HistoryListResponse(
        entries=get_entries(limit=limit, offset=offset),
        total=get_count(),
    )


@router.delete("/{entry_id}")
async def remove_entry(entry_id: str):
    if not delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"deleted": True}


@router.delete("", response_model=ClearResult)
async def clear_history():
    count = clear_all()
    return ClearResult(deleted=count)

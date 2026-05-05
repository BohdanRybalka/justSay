"""History API — list, delete, clear, aggregate stats over transcript history."""

import sqlite3

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.core.history import (
    HistoryEntry,
    HistoryStats,
    compute_stats,
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


def _busy_to_503(detail: str = "Storage busy"):
    return HTTPException(status_code=503, detail=detail, headers={"Retry-After": "1"})


@router.get("", response_model=HistoryListResponse)
async def list_history(limit: int = 50, offset: int = 0):
    try:
        return HistoryListResponse(
            entries=get_entries(limit=limit, offset=offset),
            total=get_count(),
        )
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise _busy_to_503("History store busy") from e
        raise


@router.get("/stats", response_model=HistoryStats)
async def history_stats():
    """Aggregate word counts (today / week / lifetime, by language and model)."""
    try:
        return compute_stats()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise _busy_to_503("Stats store busy") from e
        raise


@router.delete("/{entry_id}")
async def remove_entry(entry_id: str):
    try:
        if not delete_entry(entry_id):
            raise HTTPException(status_code=404, detail="Entry not found")
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise _busy_to_503("History store busy") from e
        raise
    return {"deleted": True}


@router.delete("", response_model=ClearResult)
async def clear_history():
    try:
        count = clear_all()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise _busy_to_503("History store busy") from e
        raise
    return ClearResult(deleted=count)

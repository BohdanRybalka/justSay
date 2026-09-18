"""History API — list, delete, clear, aggregate stats over transcript history."""

import sqlite3

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.transcripts import search
from app.transcripts.history import (
    CURSOR_TS_MAX,
    CURSOR_TS_MIN,
    HISTORY_LIMIT_MAX,
    HistoryCursor,
    HistoryPage,
    HistoryStats,
    clear_all,
    compute_stats,
    delete_entry,
    get_page,
)
from app.transcripts.store_errors import store_busy_as_503

router = APIRouter(prefix="/history", tags=["History"])


class HistorySearchResponse(BaseModel):
    entries: list[search.HistorySearchHit]
    total: int


class ClearResult(BaseModel):
    deleted: int


class DeleteResult(BaseModel):
    deleted: bool


_FTS_QUERY_ERROR_MARKERS = (
    "fts5",
    "syntax",
    "malformed match",
    "unterminated",
    "unknown special query",
    "no such column",
)


def _is_fts_syntax_error(e: sqlite3.OperationalError) -> bool:
    msg = str(e).lower()
    return any(marker in msg for marker in _FTS_QUERY_ERROR_MARKERS)


@router.get("", response_model=HistoryPage)
async def list_history(
    limit: int = Query(50, ge=1, le=HISTORY_LIMIT_MAX),
    before_ts: int | None = Query(None, ge=CURSOR_TS_MIN, le=CURSOR_TS_MAX),
    before_id: str | None = Query(None),
):
    """One page of history, newest first, at the cursor the last response handed back.

    Both cursor parameters absent asks for the first page and both present is a
    complete position; half a cursor is 422 (ADR 053). ``before_ts`` is bounded.
    """
    if (before_ts is None) != (before_id is None):
        raise HTTPException(
            status_code=422, detail="before_ts and before_id must be sent together"
        )
    before = None if before_ts is None else HistoryCursor(ts=before_ts, id=before_id)
    with store_busy_as_503():
        return get_page(limit=limit, before=before)


@router.get("/stats", response_model=HistoryStats)
async def history_stats():
    """Aggregate word counts (today / week / lifetime, by language and model)."""
    with store_busy_as_503():
        return compute_stats()


@router.get("/search", response_model=HistorySearchResponse)
async def history_search(
    q: str = Query(
        "",
        description="Search query (empty or sanitized-to-empty → empty list)",
        max_length=500,
    ),
    limit: int = Query(20, ge=1, le=search.SEARCH_LIMIT_MAX),
):
    """Hybrid search: the FTS5/BM25 + LIKE lane and the semantic lane, fused by RRF.

    Empty, whitespace or fully-sanitized-out ``q`` returns 200 and an empty list. A
    storage lock is 503, an FTS5 syntax error 400, a semantic failure silent (ADR 010).
    """
    try:
        with store_busy_as_503():
            entries = await search.search_history_hybrid(q, limit=limit)
    except sqlite3.OperationalError as e:
        if _is_fts_syntax_error(e):
            raise HTTPException(status_code=400, detail="Invalid search query") from e
        raise
    return HistorySearchResponse(entries=entries, total=len(entries))


@router.delete("/{entry_id}", response_model=DeleteResult)
async def remove_entry(entry_id: str):
    with store_busy_as_503():
        if not delete_entry(entry_id):
            raise HTTPException(status_code=404, detail="Entry not found")
    return DeleteResult(deleted=True)


@router.delete("", response_model=ClearResult)
async def clear_history():
    with store_busy_as_503():
        count = clear_all()
    return ClearResult(deleted=count)

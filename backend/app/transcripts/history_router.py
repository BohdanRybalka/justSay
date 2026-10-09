"""History API — list, search, star, delete and clear transcript history."""

import sqlite3

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.transcripts import search
from app.transcripts.history import (
    CURSOR_TS_MAX,
    CURSOR_TS_MIN,
    HISTORY_LIMIT_MAX,
    EntrySource,
    HistoryCursor,
    HistoryFilter,
    HistoryPage,
    clear_all,
    delete_entry,
    get_page,
    set_starred,
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


class StarRequest(BaseModel):
    starred: bool


class StarResult(BaseModel):
    starred: bool


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
    after_ts: int | None = Query(None, ge=CURSOR_TS_MIN, le=CURSOR_TS_MAX),
    after_id: str | None = Query(None),
    source: EntrySource | None = Query(None),
    starred: bool = Query(False),
):
    """One page of history: older than ``before_*`` newest first, or newer than ``after_*``.

    No cursor asks for the first page. Half a cursor, or both cursors, is 422
    (ADR 053). The newer rows come oldest first; both ``*_ts`` are bounded.
    ``source`` keeps one kind, ``starred=true`` starred entries only; the two combine.
    """
    before = _cursor("before", before_ts, before_id)
    after = _cursor("after", after_ts, after_id)
    if before is not None and after is not None:
        raise HTTPException(status_code=422, detail="send before_* or after_*, not both")
    with store_busy_as_503():
        return get_page(
            limit=limit,
            before=before,
            after=after,
            shown=HistoryFilter(source=source, starred=starred),
        )


def _cursor(side: str, ts: int | None, entry_id: str | None) -> HistoryCursor | None:
    if (ts is None) != (entry_id is None):
        raise HTTPException(
            status_code=422, detail=f"{side}_ts and {side}_id must be sent together"
        )
    return None if ts is None else HistoryCursor(ts=ts, id=entry_id)


@router.get("/search", response_model=HistorySearchResponse)
async def history_search(
    q: str = Query(
        "",
        description="Search query (empty or sanitized-to-empty → empty list)",
        max_length=500,
    ),
    limit: int = Query(20, ge=1, le=search.SEARCH_LIMIT_MAX),
    source: EntrySource | None = Query(None),
    starred: bool = Query(False),
):
    """Hybrid search: the FTS5/BM25 + LIKE lane and the semantic lane, fused by RRF.

    Empty, whitespace or fully-sanitized-out ``q`` returns 200 and an empty list. A
    storage lock is 503, an FTS5 syntax error 400, a semantic failure silent (ADR 010).
    ``source`` and ``starred`` narrow both lanes as they narrow ``GET /history``.
    """
    try:
        with store_busy_as_503():
            entries = await search.search_history_hybrid(
                q, limit=limit, shown=HistoryFilter(source=source, starred=starred)
            )
    except sqlite3.OperationalError as e:
        if _is_fts_syntax_error(e):
            raise HTTPException(status_code=400, detail="Invalid search query") from e
        raise
    return HistorySearchResponse(entries=entries, total=len(entries))


@router.put("/{entry_id}/star", response_model=StarResult)
async def star_entry(entry_id: str, body: StarRequest):
    with store_busy_as_503():
        if not set_starred(entry_id, body.starred):
            raise HTTPException(status_code=404, detail="Entry not found")
    return StarResult(starred=body.starred)


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

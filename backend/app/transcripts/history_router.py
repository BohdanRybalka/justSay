"""History API — list, delete, clear, aggregate stats over transcript history."""

import sqlite3

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.transcripts import words as words_service
from app.transcripts.history import (
    HistoryEntry,
    HistoryStats,
    clear_all,
    compute_stats,
    delete_entry,
    get_count,
    get_entries,
)
from app.transcripts.store_errors import store_busy_as_503
from app.transcripts.words import HistorySearchHit

router = APIRouter(prefix="/history", tags=["History"])


class HistoryListResponse(BaseModel):
    entries: list[HistoryEntry]
    total: int


class HistorySearchResponse(BaseModel):
    entries: list[HistorySearchHit]
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


@router.get("", response_model=HistoryListResponse)
async def list_history(limit: int = 50, offset: int = 0):
    with store_busy_as_503():
        return HistoryListResponse(
            entries=get_entries(limit=limit, offset=offset),
            total=get_count(),
        )


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
    limit: int = Query(20, ge=1, le=words_service.SEARCH_LIMIT_MAX),
):
    """Hybrid search across transcripts: always runs the FTS5/BM25+LIKE
    lane and the semantic (vector-distance) lane concurrently and fuses
    them with Reciprocal Rank Fusion (see ``words.search_history_hybrid``
    and ADR 010) — there is no more ``mode`` toggle.

    Empty / whitespace / fully-sanitized-out ``q`` returns 200 with an
    empty list — clients fall back to ``/history`` for the newest-first
    view. The 400 path is kept as defense-in-depth for FTS5 syntax errors;
    with the whitelist sanitizer it should be unreachable from sanitized
    input. Storage lock → 503. Any semantic-lane failure (disabled,
    unavailable, embed error) degrades silently to FTS-only ranking inside
    ``search_history_hybrid`` — never surfaced as an HTTP error here.
    """
    try:
        with store_busy_as_503():
            entries = await words_service.search_history_hybrid(q, limit=limit)
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

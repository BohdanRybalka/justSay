"""History API — list, delete, clear, aggregate stats over transcript history."""

import sqlite3
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core import vector_store
from app.core import words as words_service
from app.core.history import (
    HistoryEntry,
    HistoryStats,
    clear_all,
    compute_stats,
    delete_entry,
    get_count,
    get_entries,
)
from app.core.vector_store import BackfillResult, EmbeddingsStatus
from app.core.words import HistorySearchHit

router = APIRouter(prefix="/history", tags=["History"])


class HistoryListResponse(BaseModel):
    entries: list[HistoryEntry]
    total: int


class HistorySearchResponse(BaseModel):
    entries: list[HistorySearchHit]
    total: int


class ClearResult(BaseModel):
    deleted: int


class BackfillEmbeddingsRequest(BaseModel):
    batch_size: int = 50


def _busy_to_503(detail: str = "Storage busy"):
    return HTTPException(status_code=503, detail=detail, headers={"Retry-After": "1"})


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


@router.get("/search", response_model=HistorySearchResponse)
async def history_search(
    q: str = Query(
        "",
        description="Search query (empty or sanitized-to-empty → empty list)",
        max_length=500,
    ),
    limit: int = Query(20, ge=1, le=words_service.SEARCH_LIMIT_MAX),
    mode: Literal["fts", "semantic"] = "fts",
):
    """Search across transcripts. ``mode=fts`` (default): SQLite FTS5 / BM25
    plus a substring LIKE fallback for mid-word matches — behaviour
    byte-for-byte unchanged from before ``mode`` existed. ``mode=semantic``:
    embeds ``q`` with the currently-resolved embedding provider and ranks
    entries by vector distance.

    Empty / whitespace / fully-sanitized-out ``q`` returns 200 with an
    empty list in ``mode=fts`` — clients fall back to ``/history`` for the
    newest-first view. The 400 path is kept as defense-in-depth; with the
    whitelist sanitizer it should be unreachable from sanitized input.
    Storage lock → 503. ``mode=semantic`` unavailable/unready states → 503
    with a specific, UI-displayable ``detail`` (see
    ``vector_store.SemanticSearchUnavailableError``).
    """
    if mode == "semantic":
        try:
            entries = await words_service.search_history_semantic(q, limit=limit)
        except vector_store.SemanticSearchUnavailableError as e:
            raise HTTPException(status_code=503, detail=e.detail) from e
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                raise _busy_to_503("History store busy") from e
            raise
        return HistorySearchResponse(entries=entries, total=len(entries))

    try:
        entries = words_service.search_history(q, limit=limit)
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "locked" in msg:
            raise _busy_to_503("History store busy") from e
        if _is_fts_syntax_error(e):
            raise HTTPException(status_code=400, detail="Invalid search query") from e
        raise
    return HistorySearchResponse(entries=entries, total=len(entries))


@router.post("/backfill-embeddings", response_model=BackfillResult)
async def backfill_embeddings(body: BackfillEmbeddingsRequest = BackfillEmbeddingsRequest()):
    """Embed up to ``batch_size`` (clamped server-side to ``[1, 200]``)
    not-yet-embedded entries, oldest first. Never called from
    lifespan/bootstrap — this is a user-triggered action only.
    """
    try:
        return await vector_store.backfill_batch(body.batch_size)
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise _busy_to_503("History store busy") from e
        raise


@router.get("/embeddings-status", response_model=EmbeddingsStatus)
async def embeddings_status():
    """Drives the frontend's semantic-search toggle enabled/disabled state
    and hint text — the frontend reads this, not 503 body parsing."""
    try:
        return await vector_store.embeddings_status()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise _busy_to_503("History store busy") from e
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

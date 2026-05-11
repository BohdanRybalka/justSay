"""Words API — top-frequency words + LLM-generated insights."""

from __future__ import annotations

import sqlite3
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.core import words

router = APIRouter(prefix="/words", tags=["Words"])


def _busy_to_503(detail: str) -> HTTPException:
    return HTTPException(status_code=503, detail=detail, headers={"Retry-After": "1"})


@router.get("/top", response_model=words.TopWordsResponse)
async def words_top(
    lang: Literal["all", "uk", "en"] = "all",
    limit: int = Query(50, ge=1, le=words.TOP_LIMIT_MAX),
):
    try:
        return words.top_words(lang=lang, limit=limit)
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise _busy_to_503("Words store busy") from e
        raise


@router.get("/insights", response_model=words.InsightsResponse)
async def words_insights():
    """LLM-generated insights over the user's most frequent words.

    Routed through the active LLM provider (``llm_mode``) — in Local mode
    this hits Ollama; in Cloud mode the configured cloud provider. No
    silent fallback to cloud when Local mode is selected.

    Only successful responses are cached. LLM errors surface as 503 and
    leave the cache empty so the next call retries.
    """
    try:
        return await words.get_insights()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise _busy_to_503("Insights store busy") from e
        raise
    except Exception as e:
        # LLM upstream failed (Ollama down, cloud API error, network).
        # Do NOT cache — the cache only stores successful payloads.
        raise HTTPException(
            status_code=503,
            detail=f"Insights unavailable: {type(e).__name__}",
        ) from e

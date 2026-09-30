"""Words API — the most said dictation words."""

from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, Query

from app.transcripts import words
from app.transcripts.store_errors import store_busy_as_503

router = APIRouter(prefix="/words", tags=["Words"])


@router.get("/top", response_model=words.TopWordsResponse)
async def words_top(
    limit: int = Query(50, ge=1, le=words.TOP_LIMIT_MAX),
    word_filter: words.WordFilter = Query("all", alias="filter"),
):
    with store_busy_as_503():
        return await asyncio.to_thread(
            partial(words.top_words, limit=limit, word_filter=word_filter)
        )

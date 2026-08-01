"""Words API — top-frequency words."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from app.transcripts import words
from app.transcripts.store_errors import store_busy_as_503

router = APIRouter(prefix="/words", tags=["Words"])


@router.get("/top", response_model=words.TopWordsResponse)
async def words_top(
    lang: Literal["all", "uk", "en"] = "all",
    limit: int = Query(50, ge=1, le=words.TOP_LIMIT_MAX),
):
    with store_busy_as_503():
        return words.top_words(lang=lang, limit=limit)

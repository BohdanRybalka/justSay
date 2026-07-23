"""Words API — top-frequency words."""

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

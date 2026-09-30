"""Word frequency over stored dictations.

Derived from ``entries`` on demand and cached on ``history.derived_generation_locked``;
tokenisation runs in Python, outside the store lock. Files and meetings carry other
voices, so only rows whose ``source`` is ``dictation`` count. Both stop-word lists always
apply to the favourite words, because real dictations code-switch (ADR 016).
Searching transcripts lives in ``app.transcripts.search``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

from pydantic import BaseModel

from app.transcripts import history
from app.transcripts.stopwords_en import STOPWORDS_EN
from app.transcripts.stopwords_uk import STOPWORDS_UK

TOP_LIMIT_MAX = 500

STOPWORDS_ALL: frozenset[str] = STOPWORDS_UK | STOPWORDS_EN

_TOKEN_RE = re.compile(r"[\wЀ-ӿ]+(?:['’][\wЀ-ӿ]+)*", re.UNICODE)

_DICTATIONS_SQL = "SELECT cleaned_text FROM entries WHERE source = 'dictation'"


class DictationTokens(NamedTuple):
    generation: int
    counts: Counter[str]


_tokens_cache: DictationTokens | None = None


def tokenize(text: str) -> list[str]:
    """Lowercase + extract content tokens. Stop-words NOT applied here —
    callers apply ``STOPWORDS_ALL`` after, so tests can inspect the raw
    token stream."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def dictation_tokens() -> DictationTokens:
    """Every token of every dictation, counted, as of ``generation``.

    A miss reads and tokenises every dictation, so run it off the event loop.
    """
    global _tokens_cache
    with history._lock:
        generation = history.derived_generation_locked()
        if _tokens_cache is not None and _tokens_cache.generation == generation:
            return _tokens_cache
        rows = history._ensure_conn_locked().execute(_DICTATIONS_SQL).fetchall()
    tokens = DictationTokens(
        generation, Counter(token for row in rows for token in tokenize(row["cleaned_text"]))
    )
    with history._lock:
        if history.derived_generation_locked() == generation:
            _tokens_cache = tokens
    return tokens


class WordCount(BaseModel):
    word: str
    count: int


class TopWordsResponse(BaseModel):
    items: list[WordCount]


def top_words(limit: int = 50) -> TopWordsResponse:
    """The ``limit`` most said dictation words, stop-words and one-letter tokens left out."""
    content = Counter(
        {
            word: count
            for word, count in dictation_tokens().counts.items()
            if len(word) >= 2 and word not in STOPWORDS_ALL
        }
    )
    clamped_limit = max(1, min(int(limit), TOP_LIMIT_MAX))
    return TopWordsResponse(
        items=[WordCount(word=w, count=c) for w, c in content.most_common(clamped_limit)]
    )

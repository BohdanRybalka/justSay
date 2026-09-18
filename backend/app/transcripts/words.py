"""Word frequency over stored transcripts.

Derived from ``entries`` on demand: no counter table, no writes inside
``save_entry``'s lock window, no decrement-on-delete. Tokenisation runs in
Python over result rows. Both the Ukrainian and the English stop-word lists are
always applied, because real transcripts code-switch and ``entries.language``
records the dictation mode rather than the language of the text (ADR 016).
Searching transcripts lives in ``app.transcripts.search``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from app.transcripts import history
from app.transcripts.stopwords_en import STOPWORDS_EN
from app.transcripts.stopwords_uk import STOPWORDS_UK

TOP_LIMIT_MAX = 500

STOPWORDS_ALL: frozenset[str] = STOPWORDS_UK | STOPWORDS_EN

_TOKEN_RE = re.compile(r"[\wЀ-ӿ]+(?:['’][\wЀ-ӿ]+)*", re.UNICODE)

_top_words_cache: dict[str, tuple[int, int, Counter[str]]] = {}


def tokenize(text: str) -> list[str]:
    """Lowercase + extract content tokens. Stop-words NOT applied here —
    callers apply ``STOPWORDS_ALL`` after, so tests can inspect the raw
    token stream."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class WordCount(BaseModel):
    word: str
    count: int


class TopWordsResponse(BaseModel):
    items: list[WordCount]
    scanned: int


def top_words(
    lang: Literal["all", "uk", "en"] = "all",
    limit: int = 50,
) -> TopWordsResponse:
    """Top-N words across (filtered) entries, merged UK+EN stop-words applied.

    ``limit`` clamps the output only: a miss reads and tokenises every row, so run
    it off the event loop. Counts cache on ``history.derived_generation_locked``.
    """
    clamped_limit = max(1, min(int(limit), TOP_LIMIT_MAX))

    if lang == "all":
        sql = "SELECT cleaned_text FROM entries"
        params: tuple = ()
    else:
        sql = "SELECT cleaned_text FROM entries WHERE language = ?"
        params = (lang,)

    rows = None
    with history._lock:
        generation = history.derived_generation_locked()
        cached = _top_words_cache.get(lang)
        if cached is not None and cached[0] == generation:
            _, scanned, counter = cached
        else:
            conn = history._ensure_conn_locked()
            rows = conn.execute(sql, params).fetchall()

    if rows is not None:
        counter = Counter()
        for row in rows:
            for tok in tokenize(row["cleaned_text"]):
                if tok in STOPWORDS_ALL:
                    continue
                if len(tok) < 2:
                    continue
                counter[tok] += 1
        scanned = len(rows)

        with history._lock:
            if history.derived_generation_locked() == generation:
                for stale in [k for k, v in _top_words_cache.items() if v[0] != generation]:
                    del _top_words_cache[stale]
                _top_words_cache[lang] = (generation, scanned, counter)

    items = [
        WordCount(word=w, count=c)
        for w, c in counter.most_common(clamped_limit)
    ]
    return TopWordsResponse(items=items, scanned=scanned)

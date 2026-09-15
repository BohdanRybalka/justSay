"""Word frequency over stored transcripts.

Phase 1 of Plan 013. Architectural rules:

- Derived from ``entries`` on demand. No incremental counter table, no
  writes inside ``save_entry``'s lock window, no decrement-on-delete.
- Tokenisation runs in Python over result rows; the SQLite-function
  alternative was rejected for testability and connection-threading
  simplicity.
- Both Ukrainian and English stop-word lists are always applied — real
  transcripts code-switch, and ``entries.language`` is not a reliable
  content-language signal: it's the user's explicit choice when they made
  one, the provider-detected language when they requested ``"auto"``, and
  the literal ``"auto"`` sentinel only when detection itself produced
  nothing (spec 029 / docs/adr/016-detected-language-on-stt-contract.md).

Searching transcripts lives in ``app.transcripts.search``, which spec 164
split out of this module.
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
    """Compute top-N words across (filtered) entries.

    Always applies the merged UK+EN stop-word set. ``limit`` clamps the
    *output* to ``[1, TOP_LIMIT_MAX]`` and nothing else: a miss reads every row
    of ``entries`` and tokenises each one, so the cost is proportional to the
    whole history rather than to ``limit``.

    Two things keep that cost off the Words tab's five-second poll. The counts
    are cached per language against ``history.derived_generation_locked()``, the
    counter every mutator bumps, so an unchanged history is answered without a
    scan however often it is asked — the mechanism ``compute_stats`` already
    uses for the cheaper half of the same poll. And ``words_router.words_top``,
    the only caller, runs a miss through ``asyncio.to_thread``; calling this
    from the event loop directly puts the scan back on it.

    The scan holds ``history._lock`` only for the SQL. Tokenising runs outside
    it, because it needs no connection and the lock is the one every write takes.
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

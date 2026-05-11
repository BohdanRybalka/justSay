"""Word frequency + LLM insights.

Phase 1 of Plan 013. Architectural rules:

- Derived from ``entries`` on demand. No incremental counter table, no
  writes inside ``save_entry``'s lock window, no decrement-on-delete.
- Tokenisation runs in Python over result rows; the SQLite-function
  alternative was rejected for testability and connection-threading
  simplicity.
- Both Ukrainian and English stop-word lists are always applied — real
  transcripts code-switch, ``entries.language`` is the user's selected
  language at dictation time, not the actual content language.
- Insights cache TTL = 1 h, invalidated via the history mutation-listener
  registry (one-way dependency: words → history).
- Local mode privacy: insights LLM calls go through
  ``get_llm_provider(settings.llm)`` so they inherit ``llm_mode``. No
  cloud bypass in Local mode.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from app.core import history
from app.core.stopwords_en import STOPWORDS_EN
from app.core.stopwords_uk import STOPWORDS_UK

log = logging.getLogger(__name__)

INSIGHTS_TTL_SECONDS = 3600.0
INSIGHTS_MAX_WORDS_IN_PROMPT = 30
TOP_LIMIT_MAX = 500
SEARCH_LIMIT_MAX = 100

# Merged filter — applied for every language including lang=all. Allows
# Cyrillic stop-words to be filtered out of an "en" entry that happens to
# contain a Ukrainian phrase, and vice versa.
STOPWORDS_ALL: frozenset[str] = STOPWORDS_UK | STOPWORDS_EN

# Apostrophe-aware Unicode word regex. Captures Latin + Cyrillic word chars
# and inner apostrophes (straight ' and typographic '). Examples:
#   "don't"  → ["don't"]
#   "м'яко"  → ["м'яко"]
#   "she's"  → ["she's"]
_TOKEN_RE = re.compile(r"[\wЀ-ӿ]+(?:['’][\wЀ-ӿ]+)*", re.UNICODE)


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

    Always applies the merged UK+EN stop-word set. ``limit`` is clamped
    to ``[1, TOP_LIMIT_MAX]`` to avoid OOM at large DBs.
    """
    clamped_limit = max(1, min(int(limit), TOP_LIMIT_MAX))

    if lang == "all":
        sql = "SELECT cleaned_text FROM entries"
        params: tuple = ()
    else:
        sql = "SELECT cleaned_text FROM entries WHERE language = ?"
        params = (lang,)

    with history._lock:
        conn = history._ensure_conn_locked()
        rows = conn.execute(sql, params).fetchall()

    counter: Counter[str] = Counter()
    for row in rows:
        for tok in tokenize(row["cleaned_text"]):
            if tok in STOPWORDS_ALL:
                continue
            if len(tok) < 2:  # drop single letters that survived the regex
                continue
            counter[tok] += 1

    items = [
        WordCount(word=w, count=c)
        for w, c in counter.most_common(clamped_limit)
    ]
    return TopWordsResponse(items=items, scanned=len(rows))


# --- Insights cache ------------------------------------------------------

_insights_lock = threading.Lock()
# (timestamp, payload) — only successful LLM responses are cached.
_insights_cache: tuple[float, "InsightsResponse"] | None = None


class InsightsResponse(BaseModel):
    model: str
    insights: list[str]
    scanned_words: int


def invalidate_insights_cache() -> None:
    """Mutation-listener callback. Registered with ``history`` at module
    import. MUST NOT call any history mutator (contract enforced socially)."""
    global _insights_cache
    with _insights_lock:
        _insights_cache = None


# One-way registration: history exposes the registry, words depends on
# history. This module never appears in history.py's imports.
history.register_mutation_listener(invalidate_insights_cache)


def _cached_insights() -> "InsightsResponse | None":
    with _insights_lock:
        entry = _insights_cache
        if entry is None:
            return None
        ts, payload = entry
        if (time.monotonic() - ts) >= INSIGHTS_TTL_SECONDS:
            return None
        return payload


def _store_insights(payload: "InsightsResponse") -> None:
    global _insights_cache
    with _insights_lock:
        _insights_cache = (time.monotonic(), payload)


_INSIGHTS_SYSTEM_PROMPT = (
    "You are a concise analyst. Given a list of the user's most frequent "
    "spoken words (with counts), return 2-4 short fun-fact insights in "
    "the user's voice. One sentence per insight, no preamble, no "
    "numbering, no bullet markers. Return them separated by single "
    "newlines. Keep each under 100 characters."
)


async def compute_insights() -> InsightsResponse:
    """Compute fresh insights via the active LLM provider.

    Caller decides caching policy — use ``get_insights`` for the cached
    path. Raises if the LLM call fails (cache stays empty so the next
    request retries; we never cache errors).
    """
    # Late import — settings / llm only needed inside the call to avoid
    # tight coupling at module-import time.
    from app.core.config import settings
    from app.llm import get_llm_provider

    top = top_words(lang="all", limit=INSIGHTS_MAX_WORDS_IN_PROMPT)
    if not top.items:
        return InsightsResponse(model="(none)", insights=[], scanned_words=0)

    provider = get_llm_provider(settings.llm)
    word_block = "\n".join(f"{w.word}: {w.count}" for w in top.items)
    user_msg = f"Top words from my recent dictations:\n{word_block}"
    raw = await provider.process(user_msg, _INSIGHTS_SYSTEM_PROMPT, temperature=0.4)

    lines = [ln.strip().lstrip("-•* \t") for ln in raw.splitlines()]
    insights = [ln for ln in lines if ln][:4]
    return InsightsResponse(
        model=provider.model_name,
        insights=insights,
        scanned_words=sum(w.count for w in top.items),
    )


async def get_insights() -> InsightsResponse:
    """Cached insights path used by the router. Only successful responses
    are cached; LLM exceptions propagate so the router maps them to 5xx."""
    hit = _cached_insights()
    if hit is not None:
        return hit
    fresh = await compute_insights()
    _store_insights(fresh)
    return fresh


# --- Search --------------------------------------------------------------

def search_history(q: str, limit: int = 20) -> list[history.HistoryEntry]:
    """FTS5 BM25 search over cleaned_text. Empty / whitespace ``q`` short-
    circuits to an empty list — callers (the router) decide whether to
    fall back to the newest-first list.

    BM25 returns negative scores; smaller = more relevant; so
    ``ORDER BY rank ASC`` is "best first".
    """
    clamped_limit = max(1, min(int(limit), SEARCH_LIMIT_MAX))
    if not q or not q.strip():
        return []

    with history._lock:
        conn = history._ensure_conn_locked()
        rows = conn.execute(
            "SELECT e.id, e.ts, e.language, e.style, e.raw_text, "
            "e.duration_ms, e.audio_duration_seconds, e.word_count, "
            "e.model_name, e.tokens_used, bm25(entry_fts) AS rank "
            "FROM entry_fts JOIN entries e ON e.rowid = entry_fts.rowid "
            "WHERE entry_fts MATCH ? "
            "ORDER BY rank ASC LIMIT ?",
            (q, clamped_limit),
        ).fetchall()

    return [history._row_to_entry(r) for r in rows]

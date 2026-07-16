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

import asyncio
import html
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
RRF_K = 60  # Cormack et al. 2009; same value used by sqlite-vec's own hybrid-search writeup

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
    raw = await provider.process(user_msg, _INSIGHTS_SYSTEM_PROMPT, task="insights")

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

class HistorySearchHit(history.HistoryEntry):
    highlighted_text: str = ""


# Whitelist: Unicode letters/digits, straight + curly apostrophes, whitespace.
# Everything else (`-`, `/`, `:`, `^`, `*`, `"`, `(`, `)`, etc.) is stripped
# before the prefix rewrite — closes every FTS5 special-character surface in
# one shot.
_SANITIZE_KEEP_RE = re.compile(r"[^\w\s'’‘]", re.UNICODE)


def _sanitize_fts_query(q: str) -> tuple[str, list[str]]:
    """Whitelist-sanitize ``q`` and return ``(fts_expression, tokens)``.

    Returns ``("", [])`` when the sanitised query is empty.

    Tokens are lowercased so the FTS5 operator keywords ``NOT``/``AND``/
    ``OR``/``NEAR`` cease to be operators (uppercase ``NOT*`` raises
    ``OperationalError: fts5: syntax error near "NOT"``; lowercase ``not*``
    is a plain prefix term).
    """
    if not q:
        return "", []
    cleaned = _SANITIZE_KEEP_RE.sub(" ", q)
    tokens = [t.lower() for t in cleaned.split() if t]
    if not tokens:
        return "", []
    return " ".join(f"{t}*" for t in tokens), tokens


def _build_highlight(text: str | None, tokens: list[str]) -> str:
    """HTML-escape ``text`` and wrap occurrences of any ``tokens`` (case-
    insensitive) in ``<mark>…</mark>``.

    Single-pass design: offsets are found on the RAW (un-escaped) text, the
    spans are merged so overlapping/adjacent ranges produce one ``<mark>``,
    and ``html.escape`` is applied only on the segments BETWEEN spans (and
    on the content inside each ``<mark>``). This avoids three classes of
    bug from the iterative-``re.sub`` approach:
      - matches inside HTML entities (e.g. ``amp`` in ``&amp;``)
      - overlapping tokens producing nested/broken ``<mark>`` tags
      - XSS via raw ``<script>`` in the transcript content
    """
    if not text:
        return ""
    if not tokens:
        return html.escape(text)

    # Collect spans across all tokens, dropping zero-width matches.
    spans: list[tuple[int, int]] = []
    for tok in tokens:
        if not tok:
            continue
        for m in re.finditer(re.escape(tok), text, flags=re.IGNORECASE):
            if m.end() > m.start():
                spans.append((m.start(), m.end()))

    if not spans:
        return html.escape(text)

    # Merge overlapping/adjacent spans.
    spans.sort()
    merged: list[list[int]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    out: list[str] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            out.append(html.escape(text[cursor:start]))
        out.append("<mark>")
        out.append(html.escape(text[start:end]))
        out.append("</mark>")
        cursor = end
    if cursor < len(text):
        out.append(html.escape(text[cursor:]))
    return "".join(out)


def _hit_from_row(row, tokens: list[str]) -> HistorySearchHit:
    """Build a search hit from a row that includes ``cleaned_text``."""
    base = history._row_to_entry(row)
    highlighted = _build_highlight(row["cleaned_text"], tokens)
    return HistorySearchHit(
        **base.model_dump(),
        highlighted_text=highlighted,
    )


def search_history(q: str, limit: int = 20) -> list[HistorySearchHit]:
    """Two-lane search: FTS5 BM25 prefix-match (primary) + LIKE substring
    fallback (secondary).

    The FTS5 lane uses the sanitized prefix query (``прав*``-style) on
    ``entry_fts`` and orders by BM25 ascending (best first). The LIKE
    fallback runs only when the FTS5 lane returns fewer rows than
    ``clamped_limit``, catches mid-word substrings the prefix path misses
    (e.g. ``"кадабр"`` inside ``"абракадабра"``), and is de-duplicated at
    SQL level via ``id NOT IN (...)``.

    Match highlights are computed by ``_build_highlight`` on the raw
    ``cleaned_text`` joined in from ``entries`` — FTS5's own ``highlight()``
    aux function is NOT used because it does not HTML-escape the content
    text (verified at entry-gate iter 1) and would open a stored-XSS vector
    in the Tauri WebView.

    Logs only ``len(q)`` (NEVER ``q`` itself, NEVER ``len(sanitized)``).
    """
    clamped_limit = max(1, min(int(limit), SEARCH_LIMIT_MAX))
    log.debug("search len=%d", len(q or ""))

    fts_expr, tokens = _sanitize_fts_query(q or "")
    if not tokens:
        return []

    with history._lock:
        conn = history._ensure_conn_locked()
        fts_rows = conn.execute(
            "SELECT e.id, e.ts, e.language, e.style, e.raw_text, "
            "e.cleaned_text, e.duration_ms, e.audio_duration_seconds, "
            "e.word_count, e.model_name, e.tokens_used, "
            "bm25(entry_fts) AS rank "
            "FROM entry_fts JOIN entries e ON e.rowid = entry_fts.rowid "
            "WHERE entry_fts MATCH ? "
            "ORDER BY rank ASC LIMIT ?",
            (fts_expr, clamped_limit),
        ).fetchall()

        hits = [_hit_from_row(r, tokens) for r in fts_rows]

        # LIKE-fallback lane: only when the FTS5 lane left room.
        residual = clamped_limit - len(hits)
        if residual > 0:
            like_clauses = " AND ".join(["cleaned_text LIKE ? ESCAPE '\\'"] * len(tokens))
            # Escape LIKE wildcards (`%`, `_`) and the backslash itself so a
            # token containing them only matches the literal characters in
            # transcript content (exit-gate YELLOW-1).
            def _escape_like(t: str) -> str:
                return t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like_params: list = [f"%{_escape_like(t)}%" for t in tokens]

            existing_ids = [h.id for h in hits]
            if existing_ids:
                placeholders = ",".join("?" * len(existing_ids))
                not_in_clause = f" AND id NOT IN ({placeholders})"
                params = (*like_params, *existing_ids, residual)
            else:
                # Empty set — skip the NOT IN clause entirely; emitting
                # ``id NOT IN ()`` would be an SQLite syntax error.
                not_in_clause = ""
                params = (*like_params, residual)

            like_rows = conn.execute(
                "SELECT id, ts, language, style, raw_text, cleaned_text, "
                "duration_ms, audio_duration_seconds, word_count, "
                "model_name, tokens_used "
                f"FROM entries WHERE {like_clauses}{not_in_clause} "
                "ORDER BY ts DESC LIMIT ?",
                params,
            ).fetchall()

            hits.extend(_hit_from_row(r, tokens) for r in like_rows)

    # Defensive: enforce the overall cap (the SQL residual LIMIT should
    # already make this a no-op).
    return hits[:clamped_limit]


# --- Semantic search (Phase 3 — spec 003) ---------------------------------

async def search_history_semantic(q: str, limit: int = 20) -> list[HistorySearchHit]:
    """Embed ``q`` with the currently-resolved embedding provider and rank
    entries by vector distance via ``vec_entries``.

    Empty/whitespace ``q`` returns ``[]`` immediately, mirroring
    ``search_history``'s own empty-query short-circuit — never reaches the
    availability checks below and never spends an embedding API call.

    Raises ``vector_store.SemanticSearchUnavailableError`` for every
    disabled/unready state, each with its own detail string: sqlite-vec
    failed to load, embeddings disabled by the Cloud/Local eligibility
    rule, zero entries embedded yet, or any other runtime failure from
    ``provider.embed()`` itself (auth error, network failure, malformed SDK
    response). Since spec 017, the only caller is ``_semantic_lane``, which
    catches this exception and degrades to an empty lane silently (see ADR
    010) — there is no HTTP-level 503 surfaced for any of these states
    anymore. The zero-entries check happens BEFORE the (network) embed call
    so an empty index fails fast without spending an API call.

    ``highlighted_text`` is plain HTML-escaped text with no ``<mark>``
    spans — relevance here isn't token-based, so there's no single matched
    span to highlight.
    """
    # Late imports: same lazy-import discipline used elsewhere in this
    # module (get_llm_provider) — keeps embeddings/vector_store optional
    # from words.py's own import-time perspective.
    from app.core import vector_store
    from app.core.config import settings
    from app.embeddings import resolve_embedding_provider

    clamped_limit = max(1, min(int(limit), SEARCH_LIMIT_MAX))

    # Mirrors search_history's own empty-query short-circuit: an empty/
    # whitespace `q` returns an empty list immediately and never reaches
    # the availability checks below or spends a cloud embedding API call.
    if not q or not q.strip():
        return []

    if not history._vec_available:
        raise vector_store.SemanticSearchUnavailableError(
            vector_store.VEC_EXTENSION_UNAVAILABLE_DETAIL
        )

    provider, reason = await resolve_embedding_provider(
        settings.stt, settings.llm, settings.embeddings
    )
    if provider is None:
        raise vector_store.SemanticSearchUnavailableError(reason or "Semantic search is disabled")

    with history._lock:
        conn = history._ensure_conn_locked()
        indexed = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    if indexed == 0:
        raise vector_store.SemanticSearchUnavailableError(vector_store.NO_ENTRIES_EMBEDDED_DETAIL)

    # Any embedding-provider runtime failure (auth error, network failure,
    # malformed SDK response) maps to the same SemanticSearchUnavailableError
    # _semantic_lane already catches and swallows -- since spec 017, no
    # caller of this function ever lets the raw exception type reach an
    # HTTP response.
    try:
        query_vector = await provider.embed(q)
    except Exception as e:
        raise vector_store.SemanticSearchUnavailableError(
            f"Semantic search embedding failed: {type(e).__name__}"
        ) from e

    with history._lock:
        conn = history._ensure_conn_locked()
        rows = vector_store.query_similar(conn, query_vector, clamped_limit)

    return [_hit_from_row(r, []) for r in rows]


# --- Hybrid RRF search (spec 017 / ADR 010) --------------------------------

async def _semantic_lane(q: str, limit: int) -> list[HistorySearchHit]:
    """Wraps ``search_history_semantic`` so every failure mode degrades to an
    empty lane instead of propagating — see ADR 010. This is what
    structurally closes the exception-leak bug: there is no response path
    left that can carry an embedding-provider error string to the client.
    """
    from app.core import vector_store

    try:
        return await search_history_semantic(q, limit=limit)
    except vector_store.SemanticSearchUnavailableError as e:
        log.debug("Semantic lane unavailable, FTS-only: %s", e.detail)
        return []
    except Exception:
        log.warning("Semantic lane failed unexpectedly, falling back to FTS-only", exc_info=True)
        return []


def _rrf_fuse(
    fts_hits: list[HistorySearchHit],
    semantic_hits: list[HistorySearchHit],
    limit: int,
) -> list[HistorySearchHit]:
    """Reciprocal Rank Fusion: ``score(entry) = sum over lanes of
    1 / (RRF_K + rank_in_lane)``. The FTS lane is folded in first, so
    ``by_id.setdefault`` keeps its ``<mark>``-tagged ``highlighted_text``
    for any entry present in both lanes — the semantic lane's plain-escaped
    text never overwrites it.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, HistorySearchHit] = {}
    for rank, hit in enumerate(fts_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + rank)
        by_id.setdefault(hit.id, hit)  # FTS's <mark>-tagged text wins ties
    for rank, hit in enumerate(semantic_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + rank)
        by_id.setdefault(hit.id, hit)
    ranked = sorted(scores, key=lambda eid: scores[eid], reverse=True)
    return [by_id[eid] for eid in ranked[:limit]]


async def search_history_hybrid(q: str, limit: int = 20) -> list[HistorySearchHit]:
    """Always-on hybrid search: runs the FTS5/BM25+LIKE lane and the
    semantic (vector-distance) lane concurrently via ``asyncio.gather`` and
    fuses them with RRF. Both lanes fetch a fixed ``SEARCH_LIMIT_MAX``-row
    candidate pool regardless of the caller's ``limit`` so ranking has full
    context before truncation.

    ``search_history`` is synchronous/blocking (a real SQLite query under
    ``history._lock``), so it runs via ``asyncio.to_thread`` — that's what
    lets it genuinely overlap the semantic lane's own ``await`` in wall-clock
    time instead of the two lanes running sequentially.
    """
    clamped_limit = max(1, min(int(limit), SEARCH_LIMIT_MAX))
    if not q or not q.strip():
        return []
    fts_hits, semantic_hits = await asyncio.gather(
        asyncio.to_thread(search_history, q, SEARCH_LIMIT_MAX),
        _semantic_lane(q, SEARCH_LIMIT_MAX),
    )
    return _rrf_fuse(fts_hits, semantic_hits, clamped_limit)

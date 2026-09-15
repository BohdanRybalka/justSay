"""Word frequency + transcript search.

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
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import sqlite3
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from app.transcripts import history
from app.transcripts.stopwords_en import STOPWORDS_EN
from app.transcripts.stopwords_uk import STOPWORDS_UK

log = logging.getLogger(__name__)

TOP_LIMIT_MAX = 500
SEARCH_LIMIT_MAX = 100
SEARCH_SCAN_CHUNK_ROWS = 200
RRF_K = 60

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



class HistorySearchHit(history.HistoryEntry):
    highlighted_text: str = ""


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

    spans: list[tuple[int, int]] = []
    for tok in tokens:
        if not tok:
            continue
        for m in re.finditer(re.escape(tok), text, flags=re.IGNORECASE):
            if m.end() > m.start():
                spans.append((m.start(), m.end()))

    if not spans:
        return html.escape(text)

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


def _escape_like(t: str) -> str:
    """Neutralise ``LIKE``'s own wildcards so a query matches them literally.

    Paired with ``ESCAPE '\\'`` at every call site; the backslash must be doubled
    first or the two later replacements would themselves be escaped twice.
    """
    return t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _substring_page_locked(
    conn,
    like_sql: str,
    like_params: dict[str, str],
    before: tuple[int, str] | None,
    chunk: int,
):
    """Caller MUST hold ``history._lock``. One page of the substring walk.

    Reuses ``history._CURSOR_PAGE_WHERE`` and ``history._CURSOR_PAGE_ORDER``
    verbatim rather than respelling the seek, for the reason
    ``history._has_more_locked`` already gives: the row-value predicate is the
    only spelling that plans as a seek into ``entries_ts_id_idx`` instead of a
    scan from the top of it on every page.

    The projection is three small columns on purpose. ``matched`` keeps the
    ``LIKE`` evaluation inside SQLite's C implementation, and ``id``/``ts`` carry
    the cursor for the next page even when the last row of the page does not
    match -- which a page filtered down to matching rows could not do.
    """
    where = history._CURSOR_PAGE_WHERE if before is not None else ""
    params: dict[str, object] = {**like_params, "row_limit": chunk}
    if before is not None:
        params["before_ts"] = before[0]
        params["before_id"] = before[1]
    return conn.execute(
        f"SELECT id, ts, ({like_sql}) AS matched FROM entries "
        f"{where}{history._CURSOR_PAGE_ORDER}",
        params,
    ).fetchall()


def _substring_lane(tokens: list[str], exclude_ids: set[str], wanted: int) -> list[sqlite3.Row]:
    """The mid-word substring lane, walked in bounded pages.

    ``history._lock`` is taken once per page and released before the next one,
    so the hold is a function of ``SEARCH_SCAN_CHUNK_ROWS`` and never of the size
    of the user's history: a dictation finishing during a search waits for a page
    rather than for the table (ADR 061).

    ``wanted <= 0`` returns before any lock is taken. That is the case where the
    FTS lane has already filled the caller's limit, and without the guard the lane
    would spend an acquisition and a statement to learn it has nothing to collect.

    Collection order is the walk order, ``ts DESC, id DESC``, and the final
    ``id IN (...)`` fetch is re-ordered back into it in Python because ``IN`` does
    not preserve it. At most ``SEARCH_LIMIT_MAX`` ids reach that fetch.

    The walk is not one snapshot of ``entries``: a row written or deleted between
    two pages can be missed, which is the exposure ``history.get_page`` already
    carries since ADR 055. Duplicates cannot happen, because ``exclude_ids``
    de-duplicates against the FTS lane and each row is seen by one page only.
    """
    if wanted <= 0:
        return []

    like_sql = " AND ".join(
        f"cleaned_text LIKE :like_{i} ESCAPE '\\'" for i in range(len(tokens))
    )
    like_params = {f"like_{i}": f"%{_escape_like(t)}%" for i, t in enumerate(tokens)}

    collected: list[str] = []
    before: tuple[int, str] | None = None
    while True:
        with history._lock:
            conn = history._ensure_conn_locked()
            page = _substring_page_locked(
                conn, like_sql, like_params, before, SEARCH_SCAN_CHUNK_ROWS
            )
        for row in page:
            if row["matched"] and row["id"] not in exclude_ids:
                collected.append(row["id"])
                if len(collected) >= wanted:
                    break
        if len(collected) >= wanted or len(page) < SEARCH_SCAN_CHUNK_ROWS:
            break
        before = (page[-1]["ts"], page[-1]["id"])

    if not collected:
        return []

    placeholders = ",".join("?" * len(collected))
    with history._lock:
        conn = history._ensure_conn_locked()
        rows = conn.execute(
            f"SELECT {history.columns_sql(history.ENTRY_COLUMNS)} "
            f"FROM entries WHERE id IN ({placeholders})",
            collected,
        ).fetchall()
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in collected if i in by_id]


def search_history(q: str, limit: int = 20) -> list[HistorySearchHit]:
    """Two-lane search: FTS5 BM25 prefix-match (primary) + LIKE substring
    fallback (secondary).

    The FTS5 lane uses the sanitized prefix query (``прав*``-style) on
    ``entry_fts``, orders by BM25 ascending (best first), and holds
    ``history._lock`` for its single statement only.

    The substring lane catches mid-word substrings the prefix path misses
    (e.g. ``"кадабр"`` inside ``"абракадабра"``). SQLite can answer no
    leading-wildcard ``LIKE`` from an index, so it is a walk of ``entries``; it
    runs after the FTS lane's acquisition has been released, in pages of at most
    ``SEARCH_SCAN_CHUNK_ROWS`` rows with one acquisition per page, and it
    de-duplicates against the FTS rows in Python rather than with an
    ``id NOT IN (...)`` clause. The two lanes therefore no longer read one
    snapshot -- see ``_substring_lane`` and ADR 061 for what that costs and buys.

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
            f"SELECT {history.columns_sql(history.ENTRY_COLUMNS, alias='e')}, "
            "bm25(entry_fts) AS rank "
            "FROM entry_fts JOIN entries e ON e.rowid = entry_fts.rowid "
            "WHERE entry_fts MATCH ? "
            "ORDER BY rank ASC LIMIT ?",
            (fts_expr, clamped_limit),
        ).fetchall()

    rows = list(fts_rows)
    rows.extend(
        _substring_lane(
            tokens, {r["id"] for r in fts_rows}, clamped_limit - len(rows)
        )
    )

    return [_hit_from_row(r, tokens) for r in rows[:clamped_limit]]


async def search_history_semantic(q: str, limit: int = 20) -> list[HistorySearchHit]:
    """Embed ``q`` with the currently-resolved embedding provider and rank
    entries by vector distance via ``vec_entries``.

    Empty/whitespace ``q`` returns ``[]`` immediately, mirroring
    ``search_history``'s own empty-query short-circuit — never reaches the
    availability checks below and never spends an embedding API call.

    Raises ``vector_store.SemanticSearchUnavailableError`` -- a
    ``app.core.errors.ResourceUnavailableError`` -- for every
    disabled/unready state, each with its own ``message``: sqlite-vec
    failed to load, embeddings disabled by the Cloud/Local eligibility
    rule, zero entries embedded yet, or any other runtime failure from
    ``provider.embed()`` itself (auth error, network failure, malformed SDK
    response), the last of which carries the raised class name in
    ``diagnostic`` rather than in the message. Since spec 017, the only
    caller is ``_semantic_lane``, which catches this exception and degrades
    to an empty lane silently (see ADR 010) — there is no HTTP-level 503
    surfaced for any of these states anymore. The zero-entries check happens
    BEFORE the (network) embed call so an empty index fails fast without
    spending an API call.

    ``highlighted_text`` is plain HTML-escaped text with no ``<mark>``
    spans — relevance here isn't token-based, so there's no single matched
    span to highlight.
    """
    from app.core.config import settings
    from app.embeddings import resolve_embedding_provider
    from app.transcripts import vector_store

    clamped_limit = max(1, min(int(limit), SEARCH_LIMIT_MAX))

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

    try:
        query_vector = await provider.embed(q)
    except Exception as e:
        raise vector_store.SemanticSearchUnavailableError(
            "Semantic search embedding failed", diagnostic=type(e).__name__
        ) from e

    with history._lock:
        conn = history._ensure_conn_locked()
        rows = vector_store.query_similar(conn, query_vector, clamped_limit)

    return [_hit_from_row(r, []) for r in rows]



async def _semantic_lane(q: str, limit: int) -> list[HistorySearchHit]:
    """Wraps ``search_history_semantic`` so every failure mode degrades to an
    empty lane instead of propagating — see ADR 010. This is what
    structurally closes the exception-leak bug: there is no response path
    left that can carry an embedding-provider error string to the client.
    """
    from app.transcripts import vector_store

    try:
        return await search_history_semantic(q, limit=limit)
    except vector_store.SemanticSearchUnavailableError as e:
        log.debug("Semantic lane unavailable, FTS-only: %s (%s)", e.message, e.diagnostic)
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
        by_id.setdefault(hit.id, hit)
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

"""Transcript search — the FTS5 lane, the LIKE fallback, the semantic lane and RRF fusion.

``search_history`` is FTS5 BM25 prefix match, widened by words a typo or two
away, plus a paged ``LIKE`` walk; ``search_history_semantic`` is vector distance
over ``vec_entries``, and ``search_history_hybrid`` fuses both with RRF: exact
hits, close word hits, at most ``SEMANTIC_ONLY_MAX`` meaning-only hits, in order.
This module owns no connection: it borrows ``history._lock`` for every statement.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import sqlite3
import unicodedata
from typing import Literal

from app.stt.config import stt_settings
from app.transcripts import history, schema

log = logging.getLogger(__name__)

SEARCH_LIMIT_MAX = 100
SEARCH_SCAN_CHUNK_ROWS = 200
RRF_K = 60
SEMANTIC_ONLY_MAX = 3
NEAR_MARK_OPEN = '<mark class="near">'
TYPO_EXPANSIONS_MAX = 50
_VOCAB_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS temp.entry_fts_vocab "
    "USING fts5vocab(main, entry_fts, row)"
)

MatchKind = Literal["exact", "near", "meaning"]


class HistorySearchHit(history.HistoryEntry):
    highlighted_text: str = ""
    match: MatchKind = "meaning"


_SANITIZE_KEEP_RE = re.compile(r"[^\w\s'’‘]", re.UNICODE)


_WORD_RE = re.compile(r"[^\W_]+")


def _query_tokens(q: str) -> list[str]:
    """Whitelist-sanitize ``q`` into lowercased tokens, ``[]`` when nothing is left.

    No token holds a double quote, so each can sit inside an FTS5 string.
    """
    if not q:
        return []
    return [t.lower() for t in _SANITIZE_KEEP_RE.sub(" ", q).split()]


def _fts_expression(tokens: list[str], typo_terms: dict[str, list[str]]) -> str:
    """Every token as a quoted prefix OR'd with its typo terms as whole words, ANDed.

    Quoting keeps an apostrophe or ``NOT`` a term rather than FTS5 syntax.
    """
    groups = []
    for tok in tokens:
        alternatives = [f'"{tok}"*'] + [f'"{term}"' for term in typo_terms.get(tok, [])]
        groups.append(
            alternatives[0] if len(alternatives) == 1 else f"({' OR '.join(alternatives)})"
        )
    return " AND ".join(groups)


def _typo_budget(token: str) -> int:
    """Edits a token may be from a stored word, Elasticsearch's ``fuzziness: AUTO``:
    none up to 2 characters, one up to 5, two beyond. A token the FTS tokenizer
    would split (an apostrophe, an underscore) gets none."""
    if len(token) <= 2 or not _WORD_RE.fullmatch(token):
        return 0
    return 1 if len(token) <= 5 else 2


def _typo_distance(a: str, b: str, budget: int) -> int:
    """Optimal string alignment distance, where swapping two neighbours is one edit
    as in Lucene's fuzzy query; anything past ``budget`` answers ``budget + 1``."""
    if abs(len(a) - len(b)) > budget:
        return budget + 1
    before_prev: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        row = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                row[j] = min(row[j], before_prev[j - 2] + 1)
        if min(row) > budget:
            return budget + 1
        before_prev, prev = prev, row
    return min(prev[-1], budget + 1)


def _typo_terms(tokens: list[str]) -> dict[str, list[str]]:
    """Stored words within each token's typo budget that its prefix does not
    already reach, closest then most used first, at most ``TYPO_EXPANSIONS_MAX``.

    The vocabulary is read under ``history._lock``; distances run after it is freed.
    """
    budgets = {tok: _typo_budget(tok) for tok in tokens if _typo_budget(tok)}
    if not budgets:
        return {}
    shortest = min(len(tok) - budget for tok, budget in budgets.items())
    longest = max(len(tok) + budget for tok, budget in budgets.items())
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute(_VOCAB_DDL)
        vocab = conn.execute(
            "SELECT term, doc FROM temp.entry_fts_vocab WHERE length(term) BETWEEN ? AND ?",
            (shortest, longest),
        ).fetchall()
    found: dict[str, list[str]] = {}
    for tok, budget in budgets.items():
        ranked = sorted(
            (distance, -doc, term)
            for term, doc in vocab
            if not term.startswith(tok)
            and (distance := _typo_distance(tok, term, budget)) <= budget
        )
        if ranked:
            found[tok] = [term for _, _, term in ranked[:TYPO_EXPANSIONS_MAX]]
    return found


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _token_spans(text: str, tokens: list[str]) -> list[tuple[int, int, bool]]:
    """Every case-insensitive occurrence of every token as ``(start, end, whole_word)``."""
    spans: list[tuple[int, int, bool]] = []
    for tok in tokens:
        if not tok:
            continue
        for m in re.finditer(re.escape(tok), text, flags=re.IGNORECASE):
            start, end = m.start(), m.end()
            if end > start:
                whole = not (start > 0 and _is_word_char(text[start - 1])) and not (
                    end < len(text) and _is_word_char(text[end])
                )
                spans.append((start, end, whole))
    return spans


def _match_kind(text: str | None, tokens: list[str]) -> MatchKind:
    """``exact`` when every token stands in ``text`` as a whole word at least once,
    ``meaning`` for a hit found without words (the semantic lane), else ``near``."""
    if not tokens:
        return "meaning"
    if not text:
        return "near"
    missing = {tok.lower() for tok in tokens if tok}
    for start, end, whole in _token_spans(text, tokens):
        if whole:
            missing.discard(text[start:end].lower())
    return "near" if missing else "exact"


def _fold_like_fts(raw: str) -> str:
    """``raw`` as the ``unicode61 remove_diacritics 2`` tokenizer stores it: lowercased,
    a Latin letter's diacritic dropped (café → cafe), a Cyrillic й or ї kept."""
    folded = []
    for char in raw.lower():
        base = unicodedata.normalize("NFD", char)[0]
        folded.append(base if base.isascii() else char)
    return "".join(folded)


def _build_highlight(
    text: str | None, tokens: list[str], typo_terms: frozenset[str] = frozenset()
) -> str:
    """HTML-escape ``text`` and wrap any ``tokens`` and ``typo_terms`` words in a mark.

    Case-insensitive; overlapping spans merge into one mark, a plain ``<mark>``
    when any part of it is a token as a whole word and ``<mark class="near">``
    otherwise. Every segment is escaped, so no raw markup survives.
    """
    if not text:
        return ""
    typo_spans = [
        (m.start(), m.end(), False)
        for m in _WORD_RE.finditer(text)
        if _fold_like_fts(m.group()) in typo_terms
    ]
    spans = sorted(_token_spans(text, tokens) + typo_spans)
    if not spans:
        return html.escape(text)

    merged: list[list] = [list(spans[0])]
    for start, end, whole in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = merged[-1][2] or whole
        else:
            merged.append([start, end, whole])

    out: list[str] = []
    cursor = 0
    for start, end, whole in merged:
        if start > cursor:
            out.append(html.escape(text[cursor:start]))
        out.append("<mark>" if whole else NEAR_MARK_OPEN)
        out.append(html.escape(text[start:end]))
        out.append("</mark>")
        cursor = end
    if cursor < len(text):
        out.append(html.escape(text[cursor:]))
    return "".join(out)


def _hit_from_row(
    row, tokens: list[str], typo_terms: frozenset[str] = frozenset()
) -> HistorySearchHit:
    """Build a search hit from a row that includes ``cleaned_text``."""
    base = history._row_to_entry(row)
    return HistorySearchHit(
        **base.model_dump(),
        highlighted_text=_build_highlight(row["cleaned_text"], tokens, typo_terms),
        match=_match_kind(row["cleaned_text"], tokens),
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

    Reuses ``history._CURSOR_PAGE_WHERE``/``_CURSOR_PAGE_ORDER`` rather than
    respelling the seek; ``id``/``ts`` carry the next page's cursor regardless.
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


def _substring_lane(
    tokens: list[str], exclude_ids: set[str], wanted: int, starred_only: bool
) -> list[sqlite3.Row]:
    """The mid-word substring lane, walked in bounded pages, in ``ts DESC, id DESC``.

    ``history._lock`` is taken once per page, so the hold is bounded by
    ``SEARCH_SCAN_CHUNK_ROWS`` (ADR 061). The walk is not one snapshot of the table.
    """
    if wanted <= 0:
        return []

    like_sql = " AND ".join(
        [f"cleaned_text LIKE :like_{i} ESCAPE '\\'" for i in range(len(tokens))]
        + ([history._STARRED] if starred_only else [])
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
            f"SELECT {schema.columns_sql(schema.ENTRY_COLUMNS)} "
            f"FROM entries WHERE id IN ({placeholders})",
            collected,
        ).fetchall()
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in collected if i in by_id]


def _fts_rows(expression: str, limit: int, starred_only: bool) -> list[sqlite3.Row]:
    """Up to ``limit`` rows matching the FTS5 ``expression``, best BM25 rank first."""
    with history._lock:
        conn = history._ensure_conn_locked()
        return conn.execute(
            f"SELECT {schema.columns_sql(schema.ENTRY_COLUMNS, alias='e')}, "
            "bm25(entry_fts) AS rank "
            "FROM entry_fts JOIN entries e ON e.rowid = entry_fts.rowid "
            "WHERE entry_fts MATCH ? "
            f"{f'AND e.{history._STARRED} ' if starred_only else ''}"
            "ORDER BY rank ASC LIMIT ?",
            (expression, limit),
        ).fetchall()


def search_history(
    q: str, limit: int = 20, starred_only: bool = False
) -> list[HistorySearchHit]:
    """Three lanes, each filling what the one before left: FTS5 BM25 prefix match,
    the same widened by typo terms, then a ``LIKE`` substring walk. They do not read
    one snapshot and de-duplicate in Python (ADR 061). Only ``len(q)`` is logged.
    """
    clamped_limit = max(1, min(int(limit), SEARCH_LIMIT_MAX))
    log.debug("search len=%d", len(q or ""))

    tokens = _query_tokens(q or "")
    if not tokens:
        return []

    rows = _fts_rows(_fts_expression(tokens, {}), clamped_limit, starred_only)
    typo_terms = _typo_terms(tokens) if len(rows) < clamped_limit else {}
    if typo_terms:
        seen = {r["id"] for r in rows}
        widened = _fts_rows(_fts_expression(tokens, typo_terms), clamped_limit, starred_only)
        rows.extend(r for r in widened if r["id"] not in seen)
    rows = rows[:clamped_limit]
    rows.extend(
        _substring_lane(
            tokens, {r["id"] for r in rows}, clamped_limit - len(rows), starred_only
        )
    )

    marked_typos = frozenset(term for terms in typo_terms.values() for term in terms)
    return [_hit_from_row(r, tokens, marked_typos) for r in rows[:clamped_limit]]


async def search_history_semantic(q: str, limit: int = 20) -> list[HistorySearchHit]:
    """Embed ``q`` and rank entries by vector distance over ``vec_entries``.

    Empty or whitespace ``q`` returns ``[]`` without an embedding call; every
    unready state raises ``SemanticSearchUnavailableError``. No ``<mark>`` spans.
    """
    from app.embeddings import resolve_embedding_provider
    from app.embeddings.config import embedding_settings
    from app.transcripts import vector_store

    clamped_limit = max(1, min(int(limit), SEARCH_LIMIT_MAX))

    if not q or not q.strip():
        return []

    if not history._vec_available:
        raise vector_store.SemanticSearchUnavailableError(
            vector_store.VEC_EXTENSION_UNAVAILABLE_DETAIL
        )

    provider, reason = await resolve_embedding_provider(stt_settings, embedding_settings)
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
    empty lane instead of propagating (ADR 010).
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
    """Reciprocal Rank Fusion: ``score = sum over lanes of 1 / (RRF_K + rank)``.

    The FTS lane is folded in first, so ``by_id.setdefault`` keeps its
    ``<mark>``-tagged ``highlighted_text`` for an entry present in both lanes.
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


async def search_history_hybrid(
    q: str, limit: int = 20, starred_only: bool = False
) -> list[HistorySearchHit]:
    """Run the FTS/LIKE lane and the semantic lane concurrently and fuse with RRF.

    Answers exact hits, then close word hits, then at most ``SEMANTIC_ONLY_MAX``
    hits only the semantic lane found, each group in fused order. Both lanes fetch
    ``SEARCH_LIMIT_MAX`` candidates concurrently; ``starred_only`` filters both.
    """
    clamped_limit = max(1, min(int(limit), SEARCH_LIMIT_MAX))
    if not q or not q.strip():
        return []
    fts_hits, semantic_hits = await asyncio.gather(
        asyncio.to_thread(search_history, q, SEARCH_LIMIT_MAX, starred_only),
        _semantic_lane(q, SEARCH_LIMIT_MAX),
    )
    if starred_only:
        semantic_hits = [hit for hit in semantic_hits if hit.starred]
    fused = _rrf_fuse(fts_hits, semantic_hits, len(fts_hits) + len(semantic_hits))
    worded = {hit.id for hit in fts_hits}
    exact = [hit for hit in fused if hit.match == "exact"]
    near = [hit for hit in fused if hit.id in worded and hit.match == "near"]
    meaning_only = [hit for hit in fused if hit.id not in worded][:SEMANTIC_ONLY_MAX]
    return (exact + near + meaning_only)[:clamped_limit]

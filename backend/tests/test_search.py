"""Phase 1 tests — the FTS5 lane, the substring lane, semantic search and RRF fusion."""

from __future__ import annotations

import asyncio
import logging
import time
from html.parser import HTMLParser
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.errors import ResourceUnavailableError
from app.transcripts import history, search, vector_store

pytestmark = pytest.mark.usefixtures("isolated_transcript_storage")


def test_sanitize_lowercases_and_appends_star():
    expr, tokens = search._sanitize_fts_query("Я прав")
    assert expr == "я* прав*"
    assert tokens == ["я", "прав"]


def test_sanitize_lowercases_fts5_operator_keywords():
    """``NOT*``/``AND*``/``OR*`` raise FTS5 syntax errors when uppercase.
    The sanitizer must lowercase them so they become literal prefix terms."""
    expr, tokens = search._sanitize_fts_query("NOT AND OR meeting")
    assert expr == "not* and* or* meeting*"
    history.save_entry(text="meeting brief", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute(
            "SELECT count(*) FROM entry_fts WHERE entry_fts MATCH ?", (expr,)
        ).fetchone()


def test_sanitize_strips_fts5_specials_and_dash_slash():
    """`-` is FTS5 NOT, `/` is part of `NEAR/n`. Both must be stripped.
    The trailing ``*`` per token is the prefix syntax we deliberately
    add, so we only check that NO ``*`` appears inside a token."""
    expr, _tokens = search._sanitize_fts_query('"(bad:chars)*')
    assert expr == "bad* chars*"
    for bad in '"():':
        assert bad not in expr

    expr, tokens = search._sanitize_fts_query("-правив")
    assert expr == "правив*"
    assert tokens == ["правив"]

    expr, _tokens = search._sanitize_fts_query("NEAR/3 word")
    assert "/" not in expr
    assert "near*" in expr and "3*" in expr and "word*" in expr


def test_sanitize_whitespace_and_empty():
    assert search._sanitize_fts_query("") == ("", [])
    assert search._sanitize_fts_query("   ") == ("", [])
    assert search._sanitize_fts_query("\t\n") == ("", [])


def test_build_highlight_basic_match():
    out = search._build_highlight("правив у файлі", ["прав"])
    assert "<mark>прав</mark>ив у файлі" in out


def test_build_highlight_case_insensitive_cyrillic():
    out = search._build_highlight("Прав і прав", ["прав"])
    assert "<mark>Прав</mark>" in out
    assert "<mark>прав</mark>" in out


def test_build_highlight_escapes_xss_content():
    """Regression for entry-gate iter 1 RED-1 (FTS5 highlight() did not
    escape). Our Python helper MUST escape the raw text and only insert
    literal ``<mark>`` markup. No raw ``<script>`` may leak."""
    out = search._build_highlight("<script>alert(1)</script>", ["alert"])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<mark>alert</mark>" in out


def test_build_highlight_does_not_match_inside_entity():
    """Regression for entry-gate iter 1 RED-4: matching 'amp' inside the
    escaped '&amp;' would corrupt the entity. Our offsets are found on the
    raw text BEFORE escaping, so 'amp' never matches inside an entity."""
    out = search._build_highlight("AT&T", ["amp"])
    assert out == "AT&amp;T"
    assert "<mark>" not in out


def test_build_highlight_overlapping_tokens_produce_valid_html():
    """Regression for entry-gate iter 1 RED-3: iterative re.sub built
    broken nested tags. Single-pass span-merge must produce well-formed
    HTML."""
    out = search._build_highlight("mark spot", ["mark", "ar"])

    class Validator(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: list[str] = []
            self.ok = True

        def handle_starttag(self, tag, attrs):
            self.stack.append(tag)

        def handle_endtag(self, tag):
            if not self.stack or self.stack[-1] != tag:
                self.ok = False
            else:
                self.stack.pop()

    v = Validator()
    v.feed(out)
    assert v.ok and not v.stack, f"Malformed HTML: {out!r}"


def test_build_highlight_empty_text_guard():
    """Iter-2 RED-4: NULL/empty cleaned_text must not crash."""
    assert search._build_highlight("", ["x"]) == ""
    assert search._build_highlight(None, ["x"]) == ""


def test_build_highlight_token_longer_than_text():
    assert search._build_highlight("ab", ["abcdef"]) == "ab"


def test_build_highlight_empty_tokens_returns_escaped_text():
    """No tokens → no marks, but content is still HTML-escaped."""
    assert search._build_highlight("AT&T", []) == "AT&amp;T"
    assert search._build_highlight("anything", []) == "anything"


def test_search_history_prefix_match_returns_highlight():
    history.save_entry(text="правив у файлі", duration_ms=1, language="uk")
    hits = search.search_history("прав", limit=5)
    assert len(hits) == 1
    assert isinstance(hits[0], search.HistorySearchHit)
    assert "<mark>прав</mark>ив" in hits[0].highlighted_text


def test_search_history_no_results_no_crash():
    history.save_entry(text="anything", duration_ms=1)
    assert search.search_history("nonexistent_token_xyz", limit=5) == []


def test_search_history_like_fallback_catches_substring():
    """FTS5 prefix matching cannot find ``кадабр`` inside ``абракадабра``.
    The LIKE-fallback lane catches it."""
    history.save_entry(text="абракадабра", duration_ms=1)
    hits = search.search_history("кадабр", limit=5)
    assert len(hits) == 1
    assert "абракадабра" in hits[0].text
    assert "<mark>кадабр</mark>" in hits[0].highlighted_text


def test_search_history_dedup_when_both_lanes_match():
    """Row that matches both FTS5 (prefix) and LIKE (substring) appears
    exactly once."""
    history.save_entry(text="правда буде завжди прав", duration_ms=1)
    hits = search.search_history("прав", limit=5)
    assert len(hits) == 1


def test_search_history_combined_cap_enforced():
    """Iter-2 YELLOW-1: even with both lanes hitting the same row set,
    the final list length must respect ``limit``."""
    for i in range(5):
        history.save_entry(text=f"правда{i} буде", duration_ms=1)
    hits = search.search_history("прав", limit=3)
    assert len(hits) == 3


def test_search_history_empty_fts5_then_like_only_no_sql_error():
    """Iter-2 RED-3, kept after spec 125 removed the SQL guard it named.

    The substring lane no longer emits ``id NOT IN (...)`` at all -- it
    de-duplicates in Python against the FTS ids -- so the syntax error this was
    written for cannot recur. What it still pins is the lane running to a result
    when the FTS lane returned nothing at all."""
    history.save_entry(text="абракадабра", duration_ms=1)
    hits = search.search_history("кадабр", limit=5)
    assert len(hits) == 1


def test_search_history_does_not_log_query(caplog):
    """Privacy: only ``len(q)`` may appear in logs — never ``q`` and never
    ``len(sanitized_q)``."""
    history.save_entry(text="something", duration_ms=1)
    with caplog.at_level(logging.DEBUG, logger="app.transcripts.search"):
        search.search_history("secretpassword12345", limit=5)
    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "secretpassword" not in joined
    assert "12345" not in joined
    assert "len=19" in joined


@pytest.mark.asyncio
async def test_search_endpoint_returns_highlighted_text_field(client):
    """Iter-2 BLOCK-1: the ``response_model`` MUST be ``HistorySearchResponse``
    so FastAPI serializes ``highlighted_text``. Under a response model built from
    plain ``HistoryEntry`` rows the field would be silently dropped."""
    history.save_entry(text="правив у файлі", duration_ms=1, language="uk")
    resp = await client.get("/history/search?q=прав")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["entries"]) == 1
    assert "highlighted_text" in data["entries"][0]
    assert "<mark>прав</mark>" in data["entries"][0]["highlighted_text"]



@pytest.mark.asyncio
async def test_search_history_semantic_ranks_by_distance_with_plain_highlight():
    """Direct lane-level coverage of search_history_semantic's own
    ranking/highlight contract. The router-level `mode=semantic` HTTP tests
    that used to exercise this were legitimately removed by this spec (the
    `mode` param no longer exists); this restores the one piece of coverage
    that was genuinely lane-specific rather than router-specific: real
    distance-ranked ordering and the plain (non-`<mark>`) highlight shape.

    Seeds three entries with hand-placed vectors so the nearest-to-farthest
    order is unambiguous, mocking only `resolve_embedding_provider` (same
    pattern as test_vector_store.py's `test_backfill_*`/
    `test_relocate_preserves_embeddings_and_semantic_search` tests) so the
    query embed call returns a fixed vector instead of hitting a real
    provider.
    """
    near = history.save_entry(text="close match alpha", duration_ms=1)
    mid = history.save_entry(text="somewhat similar beta", duration_ms=1)
    far = history.save_entry(text="totally different gamma", duration_ms=1)

    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        for entry, vec in (
            (near, [1.0, 0.0, 0.0]),
            (mid, [0.9, 0.1, 0.0]),
            (far, [0.0, 1.0, 0.0]),
        ):
            rowid = conn.execute(
                "SELECT rowid FROM entries WHERE id = ?", (entry.id,)
            ).fetchone()[0]
            vector_store.insert_embedding(
                conn, entry.id, rowid, vec, "cloud", "text-embedding-004"
            )

    fake_provider = AsyncMock()
    fake_provider.model_name = "text-embedding-004"
    fake_provider.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])

    with patch(
        "app.embeddings.resolve_embedding_provider",
        new=AsyncMock(return_value=(fake_provider, None)),
    ):
        hits = await search.search_history_semantic("anything", limit=10)

    assert [h.id for h in hits] == [near.id, mid.id, far.id]
    assert all("<mark>" not in h.highlighted_text for h in hits)


def _seed_one_embedded_entry() -> None:
    """The zero-entries gate short-circuits before the embed call, so every test
    below that needs to reach ``provider.embed`` has to get past it first."""
    entry = history.save_entry(text="close match alpha", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        rowid = conn.execute("SELECT rowid FROM entries WHERE id = ?", (entry.id,)).fetchone()[0]
        vector_store.insert_embedding(
            conn, entry.id, rowid, [1.0, 0.0, 0.0], "cloud", "text-embedding-004"
        )


@pytest.mark.asyncio
async def test_an_embed_failure_keeps_the_class_name_out_of_the_message():
    """The provider's exception type is diagnostic material for the log, not a
    sentence a person reads. It travels in ``diagnostic``; ``message`` stays prose."""
    _seed_one_embedded_entry()
    fake_provider = AsyncMock()
    fake_provider.model_name = "text-embedding-004"
    fake_provider.embed = AsyncMock(side_effect=ZeroDivisionError("boom"))

    with patch(
        "app.embeddings.resolve_embedding_provider",
        new=AsyncMock(return_value=(fake_provider, None)),
    ):
        with pytest.raises(vector_store.SemanticSearchUnavailableError) as excinfo:
            await search.search_history_semantic("anything", limit=10)

    assert excinfo.value.message == "Semantic search embedding failed"
    assert "ZeroDivisionError" not in excinfo.value.message
    assert excinfo.value.diagnostic == "ZeroDivisionError"


@pytest.mark.asyncio
async def test_a_refused_provider_spends_no_vector_query():
    """Zero-leak: when the Cloud/Local eligibility rule hands back no provider,
    the lane refuses before anything downstream of the gate runs."""
    query_similar = Mock()
    with (
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(return_value=(None, "Semantic search is disabled")),
        ),
        patch("app.transcripts.vector_store.query_similar", query_similar),
    ):
        with pytest.raises(vector_store.SemanticSearchUnavailableError) as excinfo:
            await search.search_history_semantic("anything", limit=10)

    assert excinfo.value.message == "Semantic search is disabled"
    assert query_similar.call_count == 0


@pytest.mark.asyncio
async def test_the_only_embed_call_is_the_resolved_providers_own():
    """``search_history_semantic`` embeds through the object eligibility handed it
    and constructs no provider of its own. This pins the call site, not the
    Local/Cloud rule itself, which is decided in ``app.embeddings``."""
    _seed_one_embedded_entry()
    fake_provider = AsyncMock()
    fake_provider.model_name = "text-embedding-004"
    fake_provider.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])

    with patch(
        "app.embeddings.resolve_embedding_provider",
        new=AsyncMock(return_value=(fake_provider, None)),
    ):
        await search.search_history_semantic("anything", limit=10)

    assert fake_provider.embed.await_count == 1
    assert fake_provider.embed.await_args.args == ("anything",)


@pytest.mark.asyncio
async def test_local_dictation_mode_gets_rows_rather_than_a_refusal():
    """Spec 169 AC: a Local-mode install used to be refused with the mixed-mode
    reason on every semantic search, because the other half of the eligibility
    key was frozen at ``cloud`` and unreachable from the UI. Eligibility now
    keys on the dictation mode alone, so the same install reaches the local
    provider and gets rows back.

    ``resolve_embedding_provider`` is NOT mocked here -- that would mock away
    the rule under test. Only Ollama is stood in for: the tag probe reports the
    model pulled and the provider returns a fixed vector.
    """
    from app.core.config import settings as runtime_settings
    from app.core.types import ProviderMode
    from app.embeddings import clear_cache as clear_embeddings_cache

    entry = history.save_entry(text="local mode entry", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "local", "ollama/nomic-embed-text", 3)
        rowid = conn.execute("SELECT rowid FROM entries WHERE id = ?", (entry.id,)).fetchone()[0]
        vector_store.insert_embedding(
            conn, entry.id, rowid, [1.0, 0.0, 0.0], "local", "ollama/nomic-embed-text"
        )

    fake_local = Mock()
    fake_local.model_name = "ollama/nomic-embed-text"
    fake_local.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])

    saved_mode = runtime_settings.stt.mode
    runtime_settings.stt.mode = ProviderMode.LOCAL
    clear_embeddings_cache()
    try:
        with (
            patch("app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)),
            patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local),
        ):
            hits = await search.search_history_semantic("anything", limit=10)
    finally:
        runtime_settings.stt.mode = saved_mode
        clear_embeddings_cache()

    assert [h.id for h in hits] == [entry.id]


def test_the_semantic_refusal_is_inside_the_error_hierarchy():
    """Replaces a prose claim in docs/style-guide.md §3.1: the class is a
    ``ResourceUnavailableError``, so it answers 503 rather than becoming a 500 the
    day something does route it to a response."""
    assert issubclass(vector_store.SemanticSearchUnavailableError, ResourceUnavailableError)
    assert vector_store.SemanticSearchUnavailableError("x").status_code == 503
    assert vector_store.SemanticSearchUnavailableError("x").code == "resource_unavailable"



def _make_hit(entry_id: str, highlighted_text: str = "") -> search.HistorySearchHit:
    return search.HistorySearchHit(
        id=entry_id,
        timestamp="2024-01-01T00:00:00Z",
        language="en",
        text=entry_id,
        duration_ms=1,
        highlighted_text=highlighted_text,
    )


def test_rrf_fuse_tie_scoring_is_symmetric_and_deterministic():
    """Entry A ranked #1 in FTS / #3 in semantic and entry B ranked #3 in
    FTS / #1 in semantic must fuse to exactly equal scores
    (``1/(60+1) + 1/(60+3)`` for both), and the deterministic tiebreak (A,
    folded into the result map first via the FTS lane) sorts A before B."""
    a = _make_hit("A")
    b = _make_hit("B")
    filler_fts = _make_hit("filler_fts")
    filler_semantic = _make_hit("filler_semantic")

    fts_hits = [a, filler_fts, b]
    semantic_hits = [b, filler_semantic, a]

    fused = search._rrf_fuse(fts_hits, semantic_hits, limit=10)

    expected_score = 1.0 / (search.RRF_K + 1) + 1.0 / (search.RRF_K + 3)
    scores: dict[str, float] = {}
    for rank, hit in enumerate(fts_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (search.RRF_K + rank)
    for rank, hit in enumerate(semantic_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (search.RRF_K + rank)

    assert scores["A"] == pytest.approx(expected_score)
    assert scores["B"] == pytest.approx(expected_score)
    assert scores["A"] == scores["B"]

    ids = [h.id for h in fused]
    assert ids.index("A") < ids.index("B")


def test_rrf_fuse_dedup_combined_score_and_highlight_precedence():
    """An entry present in both lanes must contribute both lanes' RRF terms
    to ONE combined score (not double-counted as two rows, not silently
    overwritten by whichever lane ran second), and its ``highlighted_text``
    must be the FTS lane's ``<mark>``-tagged version, never the semantic
    lane's plain-escaped one."""
    fts_hit = _make_hit("shared", highlighted_text="<mark>alpha</mark> text")
    semantic_hit = _make_hit("shared", highlighted_text="alpha text")
    solo_fts_hit = _make_hit("solo")

    fused = search._rrf_fuse([fts_hit], [semantic_hit], limit=10)
    assert [h.id for h in fused] == ["shared"]
    assert fused[0].highlighted_text == "<mark>alpha</mark> text"

    fused_vs_solo = search._rrf_fuse([fts_hit, solo_fts_hit], [semantic_hit], limit=10)
    assert [h.id for h in fused_vs_solo] == ["shared", "solo"]


@pytest.mark.asyncio
async def test_search_history_hybrid_runs_lanes_concurrently():
    """Proves the two lanes actually run concurrently via asyncio.gather,
    not sequential awaits: the FTS lane (patched to a real thread-sleep, run
    through asyncio.to_thread) and the semantic lane (patched to an
    asyncio.sleep) each take measurable, deliberately different time. A
    sequential implementation's wall-clock time would be close to the SUM
    of both delays; a concurrent one is close to the LARGER delay alone.
    """
    fts_delay = 0.25
    semantic_delay = 0.5

    def slow_search_history(q, limit):
        time.sleep(fts_delay)
        return []

    async def slow_semantic_lane(q, limit):
        await asyncio.sleep(semantic_delay)
        return []

    with (
        patch("app.transcripts.search.search_history", side_effect=slow_search_history),
        patch("app.transcripts.search._semantic_lane", side_effect=slow_semantic_lane),
    ):
        start = time.monotonic()
        await search.search_history_hybrid("anything", limit=10)
        elapsed = time.monotonic() - start

    assert elapsed >= semantic_delay - 0.05
    assert elapsed < fts_delay + semantic_delay - 0.1


@pytest.mark.asyncio
async def test_search_history_hybrid_empty_query_returns_empty_without_calling_either_lane():
    """Restores the coverage the deleted router-level
    `test_search_semantic_empty_query_returns_empty_without_calling_provider`
    used to provide (spec 003) -- now at the hybrid/lane level. `words.py`'s
    `if not q or not q.strip(): return []` short-circuit (present in both
    `search_history_hybrid` and `search_history_semantic`) must fire before
    either lane runs, so an empty or whitespace-only query must never reach
    `search_history` (the FTS lane) nor the embedding provider (the semantic
    lane's `resolve_embedding_provider`/`.embed()`) -- proven here by
    patching both to raise if invoked."""
    with (
        patch(
            "app.transcripts.search.search_history",
            side_effect=AssertionError("search_history must not be called for an empty query"),
        ),
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(
                side_effect=AssertionError(
                    "resolve_embedding_provider must not be called for an empty query"
                )
            ),
        ),
    ):
        assert await search.search_history_hybrid("", limit=10) == []
        assert await search.search_history_hybrid("   ", limit=10) == []


def test_search_does_not_hold_the_store_lock_while_highlighting(monkeypatch):
    """`history._lock` is the single lock every write also takes, and
    `_build_highlight` runs `re.finditer` per token plus `html.escape` per
    segment for up to SEARCH_LIMIT_MAX rows. None of that needs the connection,
    so holding the lock across it blocks `save_entry` for its duration.

    `_lock` is a plain `threading.Lock`, not an `RLock`, so a non-blocking
    acquire from the same thread answers the question honestly.
    """
    history.save_entry(text="правив у файлі", duration_ms=1, language="uk")

    real_build_highlight = search._build_highlight
    lock_was_free: list[bool] = []

    def probing_build_highlight(text, tokens):
        acquired = history._lock.acquire(blocking=False)
        lock_was_free.append(acquired)
        if acquired:
            history._lock.release()
        return real_build_highlight(text, tokens)

    monkeypatch.setattr(search, "_build_highlight", probing_build_highlight)

    hits = search.search_history("прав", limit=5)

    assert len(hits) == 1
    assert "<mark>прав</mark>ив" in hits[0].highlighted_text
    assert lock_was_free == [True]


class _CountingLock:
    """Delegates to a real lock, counts acquisitions, and can run a one-shot
    callback immediately after a chosen release.

    ``__enter__``/``__exit__`` are the whole surface: every use of
    ``history._lock`` in ``backend/app/`` is a ``with`` statement, so delegating
    ``acquire``/``release`` as well would guard nothing. The callback fires after
    the inner lock is already released, so what it runs may take the lock itself;
    it is cleared before it runs, so a re-entry cannot fire it twice.
    """

    def __init__(self, inner):
        self._inner = inner
        self.acquisitions = 0
        self.releases = 0
        self.fire_on_release = None
        self.callback = None

    def __enter__(self):
        self.acquisitions += 1
        return self._inner.__enter__()

    def __exit__(self, *exc):
        result = self._inner.__exit__(*exc)
        self.releases += 1
        if self.releases == self.fire_on_release:
            self.fire_on_release = None
            self.callback()
        return result


def _insert_entry(entry_id: str, ts: int, text: str) -> None:
    """Seed one row with a chosen ``(ts, id)``, which ``save_entry`` cannot do.

    The ``entries_ai`` trigger keeps ``entry_fts`` in step, so both search lanes
    see the row.
    """
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute(
            "INSERT INTO entries(id, ts, language, raw_text, cleaned_text, duration_ms) "
            "VALUES (:id, :ts, 'uk', :text, :text, 0)",
            {"id": entry_id, "ts": ts, "text": text},
        )


def test_the_scan_chunk_never_exceeds_the_largest_page_the_store_already_reads():
    """The chunk size is a relationship, not a free-chosen number.

    ``get_page`` already holds the lock across ``HISTORY_LIMIT_MAX`` rows of
    ``ENTRY_READ_COLUMNS``, so a chunk of that many rows of ``(id, ts, matched)``
    is strictly less work per acquisition than a read the store already performs.
    The second assertion is what keeps the substring lane's final
    ``id IN (...)`` fetch inside the same budget as one page.
    """
    assert search.SEARCH_SCAN_CHUNK_ROWS <= history.HISTORY_LIMIT_MAX
    assert search.SEARCH_LIMIT_MAX <= search.SEARCH_SCAN_CHUNK_ROWS


def test_search_releases_the_store_lock_between_chunks(monkeypatch):
    """The defect this spec closes: one acquisition for the whole substring scan.

    With the chunk set to two rows and seven seeded, a walk that gives the lock
    back between pages takes at least four acquisitions; the single-statement
    lane takes one however large the history is.
    """
    for i in range(7):
        history.save_entry(text=f"абракадабра {i}", duration_ms=1)
    monkeypatch.setattr(search, "SEARCH_SCAN_CHUNK_ROWS", 2)

    lock = _CountingLock(history._lock)
    monkeypatch.setattr(history, "_lock", lock)

    hits = search.search_history("кадабр", limit=7)

    assert len(hits) == 7
    assert lock.acquisitions >= 4


def test_a_save_issued_between_two_chunks_lands_before_the_search_finishes(monkeypatch):
    """A dictation finishing during a search waits for a page, not for the table.

    The write is issued from the second release of the store lock -- the first
    boundary between two chunks of the substring walk -- and the assertions are a
    count and an order, never elapsed time. On a tree that holds the lock for the
    whole scan the second release never happens, so the entry is never written.
    """
    for i in range(7):
        history.save_entry(text=f"абракадабра {i}", duration_ms=1)
    monkeypatch.setattr(search, "SEARCH_SCAN_CHUNK_ROWS", 2)

    lock = _CountingLock(history._lock)
    monkeypatch.setattr(history, "_lock", lock)
    written: list = []

    def save_from_the_gap():
        written.append(history.save_entry(text="written mid scan", duration_ms=1))
        written.append(lock.acquisitions)

    lock.callback = save_from_the_gap
    lock.fire_on_release = 2

    hits = search.search_history("кадабр", limit=7)
    acquisitions_when_the_search_finished = lock.acquisitions

    assert len(written) == 2, "no chunk boundary ever released the lock"
    entry, acquisitions_once_the_write_was_done = written
    assert entry.id in {e.id for e in history.get_page(limit=50).entries}
    assert acquisitions_when_the_search_finished > acquisitions_once_the_write_was_done
    assert len(hits) == 7


def test_no_page_of_the_substring_walk_reads_more_than_one_chunk(monkeypatch):
    """The bound is per statement, not per search."""
    for i in range(7):
        history.save_entry(text=f"абракадабра {i}", duration_ms=1)
    monkeypatch.setattr(search, "SEARCH_SCAN_CHUNK_ROWS", 2)

    real_page = search._substring_page_locked
    page_sizes: list[int] = []

    def recording_page(conn, like_sql, like_params, before, chunk):
        rows = real_page(conn, like_sql, like_params, before, chunk)
        page_sizes.append(len(rows))
        return rows

    monkeypatch.setattr(search, "_substring_page_locked", recording_page)

    assert len(search.search_history("кадабр", limit=7)) == 7
    assert len(page_sizes) > 1
    assert max(page_sizes) <= 2


def test_a_filled_fts_lane_leaves_the_substring_lane_out_of_the_store(monkeypatch):
    """``residual == 0``: the lane must cost zero acquisitions, not one wasted one.

    The rows are seeded before the counting wrapper is installed, so the writes
    are not counted -- the single acquisition left is the FTS lane's.
    """
    for i in range(5):
        history.save_entry(text=f"правда{i} буде", duration_ms=1)

    lock = _CountingLock(history._lock)
    monkeypatch.setattr(history, "_lock", lock)

    hits = search.search_history("прав", limit=3)

    assert len(hits) == 3
    assert lock.acquisitions == 1


def test_an_underscore_in_a_query_matches_an_underscore_and_not_any_character():
    """``LIKE``'s own wildcards are escaped, so a query means what it says.

    ``_`` is the only wildcard that survives tokenisation -- ``_TOKEN_RE`` keeps
    ``\\w``, which includes it, and drops ``%`` and the backslash -- so it is the
    one that can be pinned end to end; ``_escape_like`` covers the other two
    directly. Both rows are mid-word matches the FTS prefix lane cannot answer.
    """
    _insert_entry("u1", 300, "абракадабра_ще")
    _insert_entry("u2", 200, "абракадабраZще")

    hits = search.search_history("кадабра_ще", limit=5)

    assert [h.id for h in hits] == ["u1"]


def test_escape_like_neutralises_every_wildcard():
    assert search._escape_like("100%") == "100\\%"
    assert search._escape_like("a_b") == "a\\_b"
    assert search._escape_like("c:\\d") == "c:\\\\d"


def test_a_substring_match_past_the_first_chunk_is_still_found(monkeypatch):
    """The walk does not stop at the first page it fails to match in."""
    _insert_entry("old", 100, "абракадабра")
    for i in range(6):
        _insert_entry(f"new{i}", 200 + i, f"нічого цікавого {i}")
    monkeypatch.setattr(search, "SEARCH_SCAN_CHUNK_ROWS", 2)

    hits = search.search_history("кадабр", limit=5)

    assert [h.id for h in hits] == ["old"]


def test_the_substring_lane_orders_its_hits_by_ts_then_id(monkeypatch):
    """``ts`` alone is not a total order -- ``save_entry`` mints milliseconds and
    two dictations can share one. The walk's cursor is ``(ts, id)``, so the lane's
    own hits come back in that order even across a chunk boundary."""
    for ts, ids in ((300, ("a1", "a2")), (200, ("b1", "b2")), (100, ("c1", "c2"))):
        for entry_id in ids:
            _insert_entry(entry_id, ts, "абракадабра")
    monkeypatch.setattr(search, "SEARCH_SCAN_CHUNK_ROWS", 2)

    hits = search.search_history("кадабр", limit=6)

    assert [h.id for h in hits] == ["a2", "a1", "b2", "b1", "c2", "c1"]


def test_a_row_both_lanes_match_appears_once_across_chunks(monkeypatch):
    """De-duplication moved from ``id NOT IN (...)`` to a Python set, and the row
    the FTS lane already returned now lies past the first chunk of the walk."""
    _insert_entry("both", 100, "правда буде завжди прав")
    for i in range(4):
        _insert_entry(f"filler{i}", 200 + i, "нічого цікавого")
    monkeypatch.setattr(search, "SEARCH_SCAN_CHUNK_ROWS", 2)

    hits = search.search_history("прав", limit=5)

    assert [h.id for h in hits] == ["both"]


def test_the_chunked_walk_stops_at_the_callers_limit(monkeypatch):
    """Six matching rows, a chunk of two and a limit of three: the walk reads two
    pages and stops, rather than reading the rest of the table and slicing."""
    for ts in range(6):
        _insert_entry(f"e{ts}", 100 + ts, "абракадабра")
    monkeypatch.setattr(search, "SEARCH_SCAN_CHUNK_ROWS", 2)

    real_page = search._substring_page_locked
    pages_read = []

    def recording_page(conn, like_sql, like_params, before, chunk):
        rows = real_page(conn, like_sql, like_params, before, chunk)
        pages_read.append(len(rows))
        return rows

    monkeypatch.setattr(search, "_substring_page_locked", recording_page)

    hits = search.search_history("кадабр", limit=3)

    assert [h.id for h in hits] == ["e5", "e4", "e3"]
    assert pages_read == [2, 2]

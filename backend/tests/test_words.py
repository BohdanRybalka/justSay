"""Phase 1 tests — word stats, tokeniser, stop-words, search."""

from __future__ import annotations

import asyncio
import logging
import time
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.errors import ResourceUnavailableError
from app.transcripts import history, vector_store, words
from app.transcripts.stopwords_en import STOPWORDS_EN
from app.transcripts.stopwords_uk import STOPWORDS_UK


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()



def test_tokenize_basic_latin():
    assert words.tokenize("hello world hello") == ["hello", "world", "hello"]


def test_tokenize_cyrillic():
    assert words.tokenize("кіт сидить на килимі") == ["кіт", "сидить", "на", "килимі"]


def test_tokenize_lowercases():
    assert words.tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_keeps_apostrophe_words():
    """Closes QA YELLOW-3: regex must keep contractions and Ukrainian
    apostrophe words as single tokens, not split them."""
    out = words.tokenize("don't say м'яко and she's right")
    assert "don't" in out
    assert "м'яко" in out
    assert "she's" in out
    assert "don" not in out
    assert "т" not in out
    assert "яко" not in out


def test_tokenize_typographic_apostrophe():
    out = words.tokenize("ім’я")
    assert "ім’я" in out


def test_tokenize_drops_punctuation():
    assert words.tokenize("hello, world!") == ["hello", "world"]



def test_stopwords_filtered_code_switching():
    """`the кіт sat на the килим` → only content words survive in both
    languages (UK + EN merged filter, always applied)."""
    history.save_entry(text="the кіт sat на the килим", duration_ms=1, language="uk")
    out = words.top_words(lang="all", limit=10)
    surviving = {item.word for item in out.items}
    assert "кіт" in surviving
    assert "килим" in surviving
    assert "sat" in surviving
    assert "the" not in surviving
    assert "на" not in surviving


def test_stopword_lists_disjoint_in_intent():
    """Sanity: 'the' is EN-only, 'на' is UK-only — no accidental
    cross-contamination."""
    assert "the" in STOPWORDS_EN and "the" not in STOPWORDS_UK
    assert "на" in STOPWORDS_UK and "на" not in STOPWORDS_EN



def test_top_words_round_trip():
    history.save_entry(text="cat dog cat fish", duration_ms=1, language="en")
    history.save_entry(text="cat house", duration_ms=1, language="en")
    out = words.top_words(lang="all", limit=10)
    by_word = {i.word: i.count for i in out.items}
    assert by_word.get("cat") == 3
    assert by_word.get("dog") == 1
    assert by_word.get("fish") == 1
    assert out.scanned == 2


def test_top_words_delete_removes_tokens():
    """Deleted entries must not contribute to the top list — the
    "derived not incremental" property."""
    e1 = history.save_entry(text="apple apple banana", duration_ms=1)
    history.save_entry(text="banana banana", duration_ms=1)
    history.delete_entry(e1.id)

    out = words.top_words(lang="all", limit=10)
    by_word = {i.word: i.count for i in out.items}
    assert by_word.get("apple") is None
    assert by_word.get("banana") == 2


def test_top_words_lang_filter_uk():
    history.save_entry(text="кіт пес рибка", duration_ms=1, language="uk")
    history.save_entry(text="cat dog fish", duration_ms=1, language="en")
    out = words.top_words(lang="uk", limit=10)
    assert out.scanned == 1
    surviving = {i.word for i in out.items}
    assert "кіт" in surviving
    assert "cat" not in surviving


def test_top_words_lang_filter_en():
    history.save_entry(text="кіт пес рибка", duration_ms=1, language="uk")
    history.save_entry(text="cat dog fish", duration_ms=1, language="en")
    out = words.top_words(lang="en", limit=10)
    assert out.scanned == 1
    surviving = {i.word for i in out.items}
    assert "cat" in surviving
    assert "кіт" not in surviving


def test_top_words_limit_clamped_to_max():
    """`limit=99999` is clamped to TOP_LIMIT_MAX so we never return more
    than 500 items even if requested."""
    for i in range(10):
        history.save_entry(text=f"word{i} word{i}", duration_ms=1)
    out = words.top_words(lang="all", limit=99999)
    assert len(out.items) <= words.TOP_LIMIT_MAX


def test_top_words_empty_db_returns_empty():
    out = words.top_words(lang="all", limit=10)
    assert out.items == []
    assert out.scanned == 0



@pytest.mark.asyncio
async def test_words_top_endpoint_smoke(client):
    history.save_entry(text="apple banana apple", duration_ms=1)
    resp = await client.get("/words/top?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scanned"] == 1
    words_list = {i["word"]: i["count"] for i in data["items"]}
    assert words_list.get("apple") == 2


@pytest.mark.asyncio
async def test_words_top_limit_validated_by_fastapi(client):
    """FastAPI's Query(le=500) constraint kicks in BEFORE the handler
    body — over-limit is a 422, not a clamped 200. Documents the
    contract."""
    resp = await client.get("/words/top?limit=99999")
    assert resp.status_code == 422


def test_sanitize_lowercases_and_appends_star():
    expr, tokens = words._sanitize_fts_query("Я прав")
    assert expr == "я* прав*"
    assert tokens == ["я", "прав"]


def test_sanitize_lowercases_fts5_operator_keywords():
    """``NOT*``/``AND*``/``OR*`` raise FTS5 syntax errors when uppercase.
    The sanitizer must lowercase them so they become literal prefix terms."""
    expr, tokens = words._sanitize_fts_query("NOT AND OR meeting")
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
    expr, _tokens = words._sanitize_fts_query('"(bad:chars)*')
    assert expr == "bad* chars*"
    for bad in '"():':
        assert bad not in expr

    expr, tokens = words._sanitize_fts_query("-правив")
    assert expr == "правив*"
    assert tokens == ["правив"]

    expr, _tokens = words._sanitize_fts_query("NEAR/3 word")
    assert "/" not in expr
    assert "near*" in expr and "3*" in expr and "word*" in expr


def test_sanitize_whitespace_and_empty():
    assert words._sanitize_fts_query("") == ("", [])
    assert words._sanitize_fts_query("   ") == ("", [])
    assert words._sanitize_fts_query("\t\n") == ("", [])


def test_build_highlight_basic_match():
    out = words._build_highlight("правив у файлі", ["прав"])
    assert "<mark>прав</mark>ив у файлі" in out


def test_build_highlight_case_insensitive_cyrillic():
    out = words._build_highlight("Прав і прав", ["прав"])
    assert "<mark>Прав</mark>" in out
    assert "<mark>прав</mark>" in out


def test_build_highlight_escapes_xss_content():
    """Regression for entry-gate iter 1 RED-1 (FTS5 highlight() did not
    escape). Our Python helper MUST escape the raw text and only insert
    literal ``<mark>`` markup. No raw ``<script>`` may leak."""
    out = words._build_highlight("<script>alert(1)</script>", ["alert"])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<mark>alert</mark>" in out


def test_build_highlight_does_not_match_inside_entity():
    """Regression for entry-gate iter 1 RED-4: matching 'amp' inside the
    escaped '&amp;' would corrupt the entity. Our offsets are found on the
    raw text BEFORE escaping, so 'amp' never matches inside an entity."""
    out = words._build_highlight("AT&T", ["amp"])
    assert out == "AT&amp;T"
    assert "<mark>" not in out


def test_build_highlight_overlapping_tokens_produce_valid_html():
    """Regression for entry-gate iter 1 RED-3: iterative re.sub built
    broken nested tags. Single-pass span-merge must produce well-formed
    HTML."""
    out = words._build_highlight("mark spot", ["mark", "ar"])

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
    assert words._build_highlight("", ["x"]) == ""
    assert words._build_highlight(None, ["x"]) == ""


def test_build_highlight_token_longer_than_text():
    assert words._build_highlight("ab", ["abcdef"]) == "ab"


def test_build_highlight_empty_tokens_returns_escaped_text():
    """No tokens → no marks, but content is still HTML-escaped."""
    assert words._build_highlight("AT&T", []) == "AT&amp;T"
    assert words._build_highlight("anything", []) == "anything"


def test_search_history_prefix_match_returns_highlight():
    history.save_entry(text="правив у файлі", duration_ms=1, language="uk")
    hits = words.search_history("прав", limit=5)
    assert len(hits) == 1
    assert isinstance(hits[0], words.HistorySearchHit)
    assert "<mark>прав</mark>ив" in hits[0].highlighted_text


def test_search_history_no_results_no_crash():
    history.save_entry(text="anything", duration_ms=1)
    assert words.search_history("nonexistent_token_xyz", limit=5) == []


def test_search_history_like_fallback_catches_substring():
    """FTS5 prefix matching cannot find ``кадабр`` inside ``абракадабра``.
    The LIKE-fallback lane catches it."""
    history.save_entry(text="абракадабра", duration_ms=1)
    hits = words.search_history("кадабр", limit=5)
    assert len(hits) == 1
    assert "абракадабра" in hits[0].text
    assert "<mark>кадабр</mark>" in hits[0].highlighted_text


def test_search_history_dedup_when_both_lanes_match():
    """Row that matches both FTS5 (prefix) and LIKE (substring) appears
    exactly once."""
    history.save_entry(text="правда буде завжди прав", duration_ms=1)
    hits = words.search_history("прав", limit=5)
    assert len(hits) == 1


def test_search_history_combined_cap_enforced():
    """Iter-2 YELLOW-1: even with both lanes hitting the same row set,
    the final list length must respect ``limit``."""
    for i in range(5):
        history.save_entry(text=f"правда{i} буде", duration_ms=1)
    hits = words.search_history("прав", limit=3)
    assert len(hits) == 3


def test_search_history_empty_fts5_then_like_only_no_sql_error():
    """Iter-2 RED-3, kept after spec 125 removed the SQL guard it named.

    The substring lane no longer emits ``id NOT IN (...)`` at all -- it
    de-duplicates in Python against the FTS ids -- so the syntax error this was
    written for cannot recur. What it still pins is the lane running to a result
    when the FTS lane returned nothing at all."""
    history.save_entry(text="абракадабра", duration_ms=1)
    hits = words.search_history("кадабр", limit=5)
    assert len(hits) == 1


def test_search_history_does_not_log_query(caplog):
    """Privacy: only ``len(q)`` may appear in logs — never ``q`` and never
    ``len(sanitized_q)``."""
    history.save_entry(text="something", duration_ms=1)
    with caplog.at_level(logging.DEBUG, logger="app.transcripts.words"):
        words.search_history("secretpassword12345", limit=5)
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
        hits = await words.search_history_semantic("anything", limit=10)

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
            await words.search_history_semantic("anything", limit=10)

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
            await words.search_history_semantic("anything", limit=10)

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
        await words.search_history_semantic("anything", limit=10)

    assert fake_provider.embed.await_count == 1
    assert fake_provider.embed.await_args.args == ("anything",)


def test_the_semantic_refusal_is_inside_the_error_hierarchy():
    """Replaces a prose claim in docs/style-guide.md §3.1: the class is a
    ``ResourceUnavailableError``, so it answers 503 rather than becoming a 500 the
    day something does route it to a response."""
    assert issubclass(vector_store.SemanticSearchUnavailableError, ResourceUnavailableError)
    assert vector_store.SemanticSearchUnavailableError("x").status_code == 503
    assert vector_store.SemanticSearchUnavailableError("x").code == "resource_unavailable"



def _make_hit(entry_id: str, highlighted_text: str = "") -> words.HistorySearchHit:
    return words.HistorySearchHit(
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

    fused = words._rrf_fuse(fts_hits, semantic_hits, limit=10)

    expected_score = 1.0 / (words.RRF_K + 1) + 1.0 / (words.RRF_K + 3)
    scores: dict[str, float] = {}
    for rank, hit in enumerate(fts_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (words.RRF_K + rank)
    for rank, hit in enumerate(semantic_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (words.RRF_K + rank)

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

    fused = words._rrf_fuse([fts_hit], [semantic_hit], limit=10)
    assert [h.id for h in fused] == ["shared"]
    assert fused[0].highlighted_text == "<mark>alpha</mark> text"

    fused_vs_solo = words._rrf_fuse([fts_hit, solo_fts_hit], [semantic_hit], limit=10)
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
        patch("app.transcripts.words.search_history", side_effect=slow_search_history),
        patch("app.transcripts.words._semantic_lane", side_effect=slow_semantic_lane),
    ):
        start = time.monotonic()
        await words.search_history_hybrid("anything", limit=10)
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
            "app.transcripts.words.search_history",
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
        assert await words.search_history_hybrid("", limit=10) == []
        assert await words.search_history_hybrid("   ", limit=10) == []


def _seed_history(count: int) -> None:
    for index in range(count):
        history.save_entry(
            text=f"запис номер {index} про кота пса рибку та ще трохи слів для сканування",
            duration_ms=1,
            language="uk",
        )


@pytest.mark.asyncio
async def test_words_top_leaves_the_event_loop_free(monkeypatch):
    """`top_words` reads every row of `entries` and regex-tokenises each one,
    and `src/settings/tabs/words.ts:186` asks for it every five seconds for as
    long as the Words tab is open — on the loop that also serves `/health` and
    the widget poll.

    The tokeniser, not the whole function, carries the known interval, so the
    blocking is the seeded rows' own scan lengthened rather than a sleep
    standing in for it — a wrapper around `top_words` itself would tick the same
    against an empty database. Shape borrowed from
    `test_audio.py::test_dictation_stop_leaves_the_event_loop_free`, after JS-97
    found `ticks > 0` satisfied by a single bare `await asyncio.sleep(0)`.
    """
    from app.transcripts import words_router

    rows = 200
    per_row_seconds = 0.001
    _seed_history(rows)

    real_tokenize = words.tokenize

    def slow_tokenize(text):
        time.sleep(per_row_seconds)
        return real_tokenize(text)

    monkeypatch.setattr(words, "tokenize", slow_tokenize)

    ticks = 0

    async def competitor():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    race = asyncio.ensure_future(competitor())
    response = await words_router.words_top(lang="all", limit=50)
    race.cancel()

    assert response.scanned == rows
    assert response.items
    assert ticks > 100, (
        f"the loop ticked {ticks} times while /words/top tokenised {rows} rows at "
        f"{per_row_seconds}s each -- the scan is still on the event loop"
    )


def test_top_words_does_not_rescan_an_unchanged_history(monkeypatch):
    """The Words tab asks every five seconds. Without a cache the whole table is
    read and re-tokenised on every tick — the gap `compute_stats` already closes
    for the cheaper half of the same poll."""
    _seed_history(20)

    scans = 0
    real_tokenize = words.tokenize

    def counting_tokenize(text):
        nonlocal scans
        scans += 1
        return real_tokenize(text)

    monkeypatch.setattr(words, "tokenize", counting_tokenize)

    first = words.top_words(lang="all", limit=5)
    after_first = scans
    second = words.top_words(lang="all", limit=5)

    assert after_first == 20
    assert scans == after_first
    assert second.items == first.items
    assert second.scanned == first.scanned


def test_a_new_entry_invalidates_the_word_counts(monkeypatch):
    """A cache that outlives a write would report yesterday's counts forever."""
    history.save_entry(text="кіт кіт кіт", duration_ms=1, language="uk")
    before = words.top_words(lang="all", limit=5)

    history.save_entry(text="пес пес пес пес", duration_ms=1, language="uk")
    after = words.top_words(lang="all", limit=5)

    assert before.scanned == 1
    assert after.scanned == 2
    assert [i.word for i in after.items][0] == "пес"


def test_a_cleared_history_invalidates_the_word_counts():
    history.save_entry(text="кіт кіт кіт", duration_ms=1, language="uk")
    assert words.top_words(lang="all", limit=5).items

    history.clear_all()

    empty = words.top_words(lang="all", limit=5)
    assert empty.items == []
    assert empty.scanned == 0


def test_search_does_not_hold_the_store_lock_while_highlighting(monkeypatch):
    """`history._lock` is the single lock every write also takes, and
    `_build_highlight` runs `re.finditer` per token plus `html.escape` per
    segment for up to SEARCH_LIMIT_MAX rows. None of that needs the connection,
    so holding the lock across it blocks `save_entry` for its duration.

    `_lock` is a plain `threading.Lock`, not an `RLock`, so a non-blocking
    acquire from the same thread answers the question honestly.
    """
    history.save_entry(text="правив у файлі", duration_ms=1, language="uk")

    real_build_highlight = words._build_highlight
    lock_was_free: list[bool] = []

    def probing_build_highlight(text, tokens):
        acquired = history._lock.acquire(blocking=False)
        lock_was_free.append(acquired)
        if acquired:
            history._lock.release()
        return real_build_highlight(text, tokens)

    monkeypatch.setattr(words, "_build_highlight", probing_build_highlight)

    hits = words.search_history("прав", limit=5)

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
    assert words.SEARCH_SCAN_CHUNK_ROWS <= history.HISTORY_LIMIT_MAX
    assert words.SEARCH_LIMIT_MAX <= words.SEARCH_SCAN_CHUNK_ROWS


def test_search_releases_the_store_lock_between_chunks(monkeypatch):
    """The defect this spec closes: one acquisition for the whole substring scan.

    With the chunk set to two rows and seven seeded, a walk that gives the lock
    back between pages takes at least four acquisitions; the single-statement
    lane takes one however large the history is.
    """
    for i in range(7):
        history.save_entry(text=f"абракадабра {i}", duration_ms=1)
    monkeypatch.setattr(words, "SEARCH_SCAN_CHUNK_ROWS", 2)

    lock = _CountingLock(history._lock)
    monkeypatch.setattr(history, "_lock", lock)

    hits = words.search_history("кадабр", limit=7)

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
    monkeypatch.setattr(words, "SEARCH_SCAN_CHUNK_ROWS", 2)

    lock = _CountingLock(history._lock)
    monkeypatch.setattr(history, "_lock", lock)
    written: list = []

    def save_from_the_gap():
        written.append(history.save_entry(text="written mid scan", duration_ms=1))
        written.append(lock.acquisitions)

    lock.callback = save_from_the_gap
    lock.fire_on_release = 2

    hits = words.search_history("кадабр", limit=7)
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
    monkeypatch.setattr(words, "SEARCH_SCAN_CHUNK_ROWS", 2)

    real_page = words._substring_page_locked
    page_sizes: list[int] = []

    def recording_page(conn, like_sql, like_params, before, chunk):
        rows = real_page(conn, like_sql, like_params, before, chunk)
        page_sizes.append(len(rows))
        return rows

    monkeypatch.setattr(words, "_substring_page_locked", recording_page)

    assert len(words.search_history("кадабр", limit=7)) == 7
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

    hits = words.search_history("прав", limit=3)

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

    hits = words.search_history("кадабра_ще", limit=5)

    assert [h.id for h in hits] == ["u1"]


def test_escape_like_neutralises_every_wildcard():
    assert words._escape_like("100%") == "100\\%"
    assert words._escape_like("a_b") == "a\\_b"
    assert words._escape_like("c:\\d") == "c:\\\\d"


def test_a_substring_match_past_the_first_chunk_is_still_found(monkeypatch):
    """The walk does not stop at the first page it fails to match in."""
    _insert_entry("old", 100, "абракадабра")
    for i in range(6):
        _insert_entry(f"new{i}", 200 + i, f"нічого цікавого {i}")
    monkeypatch.setattr(words, "SEARCH_SCAN_CHUNK_ROWS", 2)

    hits = words.search_history("кадабр", limit=5)

    assert [h.id for h in hits] == ["old"]


def test_the_substring_lane_orders_its_hits_by_ts_then_id(monkeypatch):
    """``ts`` alone is not a total order -- ``save_entry`` mints milliseconds and
    two dictations can share one. The walk's cursor is ``(ts, id)``, so the lane's
    own hits come back in that order even across a chunk boundary."""
    for ts, ids in ((300, ("a1", "a2")), (200, ("b1", "b2")), (100, ("c1", "c2"))):
        for entry_id in ids:
            _insert_entry(entry_id, ts, "абракадабра")
    monkeypatch.setattr(words, "SEARCH_SCAN_CHUNK_ROWS", 2)

    hits = words.search_history("кадабр", limit=6)

    assert [h.id for h in hits] == ["a2", "a1", "b2", "b1", "c2", "c1"]


def test_a_row_both_lanes_match_appears_once_across_chunks(monkeypatch):
    """De-duplication moved from ``id NOT IN (...)`` to a Python set, and the row
    the FTS lane already returned now lies past the first chunk of the walk."""
    _insert_entry("both", 100, "правда буде завжди прав")
    for i in range(4):
        _insert_entry(f"filler{i}", 200 + i, "нічого цікавого")
    monkeypatch.setattr(words, "SEARCH_SCAN_CHUNK_ROWS", 2)

    hits = words.search_history("прав", limit=5)

    assert [h.id for h in hits] == ["both"]


def test_the_chunked_walk_stops_at_the_callers_limit(monkeypatch):
    """Six matching rows, a chunk of two and a limit of three: the walk reads two
    pages and stops, rather than reading the rest of the table and slicing."""
    for ts in range(6):
        _insert_entry(f"e{ts}", 100 + ts, "абракадабра")
    monkeypatch.setattr(words, "SEARCH_SCAN_CHUNK_ROWS", 2)

    real_page = words._substring_page_locked
    pages_read = []

    def recording_page(conn, like_sql, like_params, before, chunk):
        rows = real_page(conn, like_sql, like_params, before, chunk)
        pages_read.append(len(rows))
        return rows

    monkeypatch.setattr(words, "_substring_page_locked", recording_page)

    hits = words.search_history("кадабр", limit=3)

    assert [h.id for h in hits] == ["e5", "e4", "e3"]
    assert pages_read == [2, 2]

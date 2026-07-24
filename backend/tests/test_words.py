"""Phase 1 tests — word stats, tokeniser, stop-words, search."""

from __future__ import annotations

import asyncio
import logging
import time
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.core import history, vector_store, words
from app.core.stopwords_en import STOPWORDS_EN
from app.core.stopwords_uk import STOPWORDS_UK


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
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
    """Iter-2 RED-3: ``id NOT IN ()`` would be a SQL syntax error if the
    LIKE-fallback lane ran without any FTS5 results. The guard must skip
    the NOT IN clause when ``fts_rows`` is empty."""
    history.save_entry(text="абракадабра", duration_ms=1)
    hits = words.search_history("кадабр", limit=5)
    assert len(hits) == 1


def test_search_history_does_not_log_query(caplog):
    """Privacy: only ``len(q)`` may appear in logs — never ``q`` and never
    ``len(sanitized_q)``."""
    history.save_entry(text="something", duration_ms=1)
    with caplog.at_level(logging.DEBUG, logger="app.core.words"):
        words.search_history("secretpassword12345", limit=5)
    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "secretpassword" not in joined
    assert "12345" not in joined
    assert "len=19" in joined


@pytest.mark.asyncio
async def test_search_endpoint_returns_highlighted_text_field(client):
    """Iter-2 BLOCK-1: the ``response_model`` MUST be ``HistorySearchResponse``
    so FastAPI serializes ``highlighted_text``. With the old
    ``HistoryListResponse`` model the field would be silently dropped."""
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



def _make_hit(entry_id: str, highlighted_text: str = "") -> words.HistorySearchHit:
    return words.HistorySearchHit(
        id=entry_id,
        timestamp="2024-01-01T00:00:00Z",
        language="en",
        style="normal",
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
        patch("app.core.words.search_history", side_effect=slow_search_history),
        patch("app.core.words._semantic_lane", side_effect=slow_semantic_lane),
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
            "app.core.words.search_history",
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

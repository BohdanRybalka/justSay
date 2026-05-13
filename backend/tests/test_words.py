"""Phase 1 tests — word stats, tokeniser, stop-words, insights cache + privacy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.core import history, words
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
    # Insights cache is a separate module variable — wipe between tests so
    # one test's payload doesn't survive into the next.
    monkeypatch.setattr(words, "_insights_cache", None)

    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()


# --- Tokeniser -----------------------------------------------------------

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
    # Confirm we did NOT produce the broken stems "don", "t", "м", "яко".
    assert "don" not in out
    assert "т" not in out
    assert "яко" not in out


def test_tokenize_typographic_apostrophe():
    out = words.tokenize("ім’я")  # ’ U+2019
    assert "ім’я" in out


def test_tokenize_drops_punctuation():
    assert words.tokenize("hello, world!") == ["hello", "world"]


# --- Stop-word filtering -------------------------------------------------

def test_stopwords_filtered_code_switching():
    """`the кіт sat на the килим` → only content words survive in both
    languages (UK + EN merged filter, always applied)."""
    history.save_entry(text="the кіт sat на the килим", duration_ms=1, language="uk")
    out = words.top_words(lang="all", limit=10)
    surviving = {item.word for item in out.items}
    assert "кіт" in surviving
    assert "килим" in surviving
    assert "sat" in surviving
    assert "the" not in surviving  # EN stop-word
    assert "на" not in surviving   # UK stop-word


def test_stopword_lists_disjoint_in_intent():
    """Sanity: 'the' is EN-only, 'на' is UK-only — no accidental
    cross-contamination."""
    assert "the" in STOPWORDS_EN and "the" not in STOPWORDS_UK
    assert "на" in STOPWORDS_UK and "на" not in STOPWORDS_EN


# --- top_words behaviour -------------------------------------------------

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
    assert by_word.get("apple") is None  # all apple entries gone
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


# --- Insights cache + privacy --------------------------------------------

@pytest.mark.asyncio
async def test_insights_cache_hits_within_ttl():
    """Second call within TTL must NOT re-invoke the LLM."""
    history.save_entry(text="cat cat cat dog", duration_ms=1)

    fake_provider = AsyncMock()
    fake_provider.model_name = "ollama/test"
    fake_provider.process = AsyncMock(return_value="Insight one\nInsight two")

    with patch("app.llm.get_llm_provider", return_value=fake_provider):
        first = await words.get_insights()
        second = await words.get_insights()

    assert first == second
    assert fake_provider.process.call_count == 1


@pytest.mark.asyncio
async def test_insights_cache_invalidated_by_save():
    """Closes Plan 013 invalidation contract — save_entry fires the
    mutation-listener registry which clears the insights cache."""
    history.save_entry(text="cat cat dog", duration_ms=1)

    fake_provider = AsyncMock()
    fake_provider.model_name = "ollama/test"
    fake_provider.process = AsyncMock(return_value="Insight A")

    with patch("app.llm.get_llm_provider", return_value=fake_provider):
        await words.get_insights()
        # Mutate — listener must wipe the cache.
        history.save_entry(text="more text", duration_ms=1)
        await words.get_insights()

    assert fake_provider.process.call_count == 2


@pytest.mark.asyncio
async def test_insights_cache_invalidated_by_delete():
    # Two entries so the second get_insights still has content to feed
    # the LLM after the delete (empty DB short-circuits before the LLM).
    e = history.save_entry(text="cat cat dog", duration_ms=1)
    history.save_entry(text="apple banana", duration_ms=1)

    fake_provider = AsyncMock()
    fake_provider.model_name = "ollama/test"
    fake_provider.process = AsyncMock(return_value="Insight A")

    with patch("app.llm.get_llm_provider", return_value=fake_provider):
        await words.get_insights()
        history.delete_entry(e.id)
        await words.get_insights()

    assert fake_provider.process.call_count == 2


@pytest.mark.asyncio
async def test_insights_cache_invalidated_by_clear():
    history.save_entry(text="cat cat dog", duration_ms=1)

    fake_provider = AsyncMock()
    fake_provider.model_name = "ollama/test"
    fake_provider.process = AsyncMock(return_value="Insight A")

    with patch("app.llm.get_llm_provider", return_value=fake_provider):
        await words.get_insights()
        history.clear_all()
        # Empty DB → no LLM call (early return) — assert separately.
        result = await words.get_insights()

    assert result.insights == []
    assert result.scanned_words == 0
    # The clear_all call invalidated the cache; the second get_insights
    # short-circuits on empty top-words and never calls the provider.
    assert fake_provider.process.call_count == 1


@pytest.mark.asyncio
async def test_insights_error_not_cached():
    """Closes QA YELLOW-5: LLM failures must NOT poison the cache. The
    next call retries the provider."""
    history.save_entry(text="cat cat dog", duration_ms=1)

    fake_provider = AsyncMock()
    fake_provider.model_name = "ollama/test"
    fake_provider.process = AsyncMock(side_effect=RuntimeError("ollama down"))

    with patch("app.llm.get_llm_provider", return_value=fake_provider):
        with pytest.raises(RuntimeError):
            await words.get_insights()
        # Recover the provider; the cache must be empty so this call hits it.
        fake_provider.process = AsyncMock(return_value="Insight A")
        recovered = await words.get_insights()

    assert recovered.insights == ["Insight A"]


@pytest.mark.asyncio
async def test_insights_privacy_uses_active_provider_factory():
    """Closes QA YELLOW-1: /words/insights MUST route through
    ``get_llm_provider(settings.llm)`` so it inherits ``llm_mode``. In
    Local mode the cloud provider must never appear.

    The test mocks ``get_llm_provider`` itself (not the HTTP client) so a
    direct cloud-SDK bypass would still fail the assertion below.
    """
    history.save_entry(text="cat cat dog", duration_ms=1)

    fake_local = AsyncMock()
    fake_local.model_name = "ollama/gemma3:4b"
    fake_local.process = AsyncMock(return_value="Insight A")

    with patch("app.llm.get_llm_provider", return_value=fake_local) as factory:
        result = await words.get_insights()

    factory.assert_called_once()  # words.py uses the factory, no SDK bypass
    assert result.model == "ollama/gemma3:4b"


# --- Router smoke tests --------------------------------------------------

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


@pytest.mark.asyncio
async def test_words_insights_503_when_llm_fails(client):
    history.save_entry(text="cat cat dog", duration_ms=1)
    fake_provider = AsyncMock()
    fake_provider.model_name = "ollama/test"
    fake_provider.process = AsyncMock(side_effect=RuntimeError("ollama down"))

    with patch("app.llm.get_llm_provider", return_value=fake_provider):
        resp = await client.get("/words/insights")

    assert resp.status_code == 503
    assert "Insights unavailable" in resp.json()["detail"]


# --- Plan 021: search sanitization + highlight ---------------------------

import logging
from html.parser import HTMLParser


def test_sanitize_lowercases_and_appends_star():
    expr, tokens = words._sanitize_fts_query("Я прав")
    assert expr == "я* прав*"
    assert tokens == ["я", "прав"]


def test_sanitize_lowercases_fts5_operator_keywords():
    """``NOT*``/``AND*``/``OR*`` raise FTS5 syntax errors when uppercase.
    The sanitizer must lowercase them so they become literal prefix terms."""
    expr, tokens = words._sanitize_fts_query("NOT AND OR meeting")
    assert expr == "not* and* or* meeting*"
    # Regression: the rewritten expression must not raise FTS5 syntax error.
    history.save_entry(text="meeting brief", duration_ms=1)
    # Direct conn check to prove the expression itself is FTS5-valid.
    with history._lock:
        conn = history._ensure_conn_locked()
        # Run a MATCH with the expression — should not OperationalError.
        conn.execute(
            "SELECT count(*) FROM entry_fts WHERE entry_fts MATCH ?", (expr,)
        ).fetchone()


def test_sanitize_strips_fts5_specials_and_dash_slash():
    """`-` is FTS5 NOT, `/` is part of `NEAR/n`. Both must be stripped.
    The trailing ``*`` per token is the prefix syntax we deliberately
    add, so we only check that NO ``*`` appears inside a token."""
    expr, _tokens = words._sanitize_fts_query('"(bad:chars)*')
    # Specials stripped — only sanitized tokens with trailing `*` survive.
    assert expr == "bad* chars*"
    for bad in '"():':
        assert bad not in expr

    expr, tokens = words._sanitize_fts_query("-правив")
    assert expr == "правив*"
    assert tokens == ["правив"]

    expr, _tokens = words._sanitize_fts_query("NEAR/3 word")
    assert "/" not in expr
    # 'NEAR' is no longer an operator after lowercase, and '/' is stripped.
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

    # Round-trip through the HTML parser to assert well-formedness.
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
    # Substring 'кадабр' does NOT prefix-match any token; FTS5 returns 0.
    history.save_entry(text="абракадабра", duration_ms=1)
    # Should not raise OperationalError("near ')': syntax error").
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

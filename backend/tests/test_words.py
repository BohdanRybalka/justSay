"""Favourite words: the tokeniser, the stop-words, dictations only, the cache."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.transcripts import history, words
from app.transcripts.stopwords_en import STOPWORDS_EN
from app.transcripts.stopwords_uk import STOPWORDS_UK

pytestmark = pytest.mark.usefixtures("isolated_transcript_storage")


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
    out = words.top_words(limit=10)
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
    out = words.top_words(limit=10)
    by_word = {i.word: i.count for i in out.items}
    assert by_word.get("cat") == 3
    assert by_word.get("dog") == 1
    assert by_word.get("fish") == 1


def test_top_words_delete_removes_tokens():
    """Deleted entries must not contribute to the top list — the
    "derived not incremental" property."""
    e1 = history.save_entry(text="apple apple banana", duration_ms=1)
    history.save_entry(text="banana banana", duration_ms=1)
    history.delete_entry(e1.id)

    out = words.top_words(limit=10)
    by_word = {i.word: i.count for i in out.items}
    assert by_word.get("apple") is None
    assert by_word.get("banana") == 2


def test_top_words_limit_clamped_to_max():
    """`limit=99999` is clamped to TOP_LIMIT_MAX so we never return more
    than 500 items even if requested."""
    for i in range(10):
        history.save_entry(text=f"word{i} word{i}", duration_ms=1)
    out = words.top_words(limit=99999)
    assert len(out.items) <= words.TOP_LIMIT_MAX


def test_top_words_empty_db_returns_empty():
    assert words.top_words(limit=10).items == []


def test_files_and_meetings_carry_other_voices_and_count_nowhere():
    history.save_entry(text="кіт кіт", duration_ms=1)
    history.save_entry(text="podcast podcast podcast", duration_ms=1, source="file")
    history.save_entry(text="colleague colleague colleague", duration_ms=1, source="meeting")

    assert [(i.word, i.count) for i in words.top_words(limit=10).items] == [("кіт", 2)]
    assert set(words.dictation_tokens().counts) == {"кіт"}



@pytest.mark.asyncio
async def test_words_top_endpoint_smoke(client):
    history.save_entry(text="apple banana apple", duration_ms=1)
    resp = await client.get("/words/top?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    words_list = {i["word"]: i["count"] for i in data["items"]}
    assert words_list.get("apple") == 2


@pytest.mark.asyncio
async def test_words_top_limit_validated_by_fastapi(client):
    """FastAPI's Query(le=500) constraint kicks in BEFORE the handler
    body — over-limit is a 422, not a clamped 200. Documents the
    contract."""
    resp = await client.get("/words/top?limit=99999")
    assert resp.status_code == 422


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
    and the Insights panel asks for it each time the window comes back — on the
    loop that also serves `/health` and the widget poll.

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
    response = await words_router.words_top(limit=50)
    race.cancel()

    assert response.items
    assert ticks > 100, (
        f"the loop ticked {ticks} times while /words/top tokenised {rows} rows at "
        f"{per_row_seconds}s each -- the scan is still on the event loop"
    )


def test_top_words_does_not_rescan_an_unchanged_history(monkeypatch):
    """Insights reads the words and the vocabulary on every open. Without a cache
    the whole table is read and re-tokenised on every read."""
    _seed_history(20)

    scans = 0
    real_tokenize = words.tokenize

    def counting_tokenize(text):
        nonlocal scans
        scans += 1
        return real_tokenize(text)

    monkeypatch.setattr(words, "tokenize", counting_tokenize)

    first = words.top_words(limit=5)
    after_first = scans
    second = words.top_words(limit=5)

    assert after_first == 20
    assert scans == after_first
    assert second.items == first.items


def test_a_new_entry_invalidates_the_word_counts(monkeypatch):
    """A cache that outlives a write would report yesterday's counts forever."""
    history.save_entry(text="кіт кіт кіт", duration_ms=1, language="uk")
    before = words.top_words(limit=5)

    history.save_entry(text="пес пес пес пес", duration_ms=1, language="uk")
    after = words.top_words(limit=5)

    assert [i.word for i in before.items] == ["кіт"]
    assert [i.word for i in after.items][0] == "пес"


def test_a_cleared_history_invalidates_the_word_counts():
    history.save_entry(text="кіт кіт кіт", duration_ms=1, language="uk")
    assert words.top_words(limit=5).items

    history.clear_all()

    assert words.top_words(limit=5).items == []

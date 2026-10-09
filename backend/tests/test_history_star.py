"""Starring an entry, and History and search narrowed by kind and to starred entries."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from app.transcripts import history, search


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_page_total_cache", None)
    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()


def _save_at(
    minute: int,
    text: str = "note",
    words: int = 10,
    source: history.EntrySource = "dictation",
) -> str:
    entry_id = history.save_entry(
        text=text, duration_ms=1, word_count=words, source=source
    ).id
    ts = int(datetime(2026, 7, 28, 9, minute).timestamp() * 1000)
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        conn.execute("UPDATE entries SET ts = ? WHERE id = ?", (ts, entry_id))
        conn.execute("COMMIT")
        history.invalidate_derived_caches_locked()
    return entry_id


def _one_of_each() -> dict[str, str]:
    return {
        "dictation": _save_at(0, text="budget dictated", words=3, source="dictation"),
        "meeting": _save_at(1, text="budget discussed", words=5, source="meeting"),
        "file": _save_at(2, text="budget recorded", words=7, source="file"),
    }


@pytest.mark.asyncio
async def test_a_star_set_through_the_route_is_read_back_and_can_be_cleared(client):
    entry_id = _save_at(0)

    starred = await client.put(f"/history/{entry_id}/star", json={"starred": True})
    assert starred.status_code == 200
    assert starred.json() == {"starred": True}
    assert (await client.get("/history")).json()["entries"][0]["starred"] is True

    await client.put(f"/history/{entry_id}/star", json={"starred": False})
    assert (await client.get("/history")).json()["entries"][0]["starred"] is False


@pytest.mark.asyncio
async def test_starring_an_unknown_entry_is_404(client):
    response = await client.put("/history/missing/star", json={"starred": True})

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_the_starred_filter_pages_over_starred_entries_only(client):
    ids = [_save_at(minute) for minute in range(6)]
    for entry_id in (ids[0], ids[2], ids[5]):
        history.set_starred(entry_id, True)

    first = (await client.get("/history?limit=2&starred=true")).json()
    cursor = first["next_cursor"]
    second = (
        await client.get(
            f"/history?limit=2&starred=true&before_ts={cursor['ts']}&before_id={cursor['id']}"
        )
    ).json()

    assert [e["id"] for e in first["entries"]] == [ids[5], ids[2]]
    assert [e["id"] for e in second["entries"]] == [ids[0]]
    assert second["next_cursor"] is None
    assert first["total"] == 3
    assert first["days"] == [{"date": "2026-07-28", "recordings": 3, "words": 30}]


def test_the_newer_read_under_the_starred_filter_skips_unstarred_rows():
    ids = [_save_at(minute) for minute in range(3)]
    history.set_starred(ids[0], True)
    history.set_starred(ids[2], True)
    oldest = history.HistoryCursor(
        ts=int(datetime(2026, 7, 28, 9, 0).timestamp() * 1000), id=ids[0]
    )

    newer = history.get_page(limit=50, after=oldest, shown=history.HistoryFilter(starred=True))

    assert [e.id for e in newer.entries] == [ids[2]]


@pytest.mark.asyncio
async def test_search_under_the_starred_filter_finds_only_starred_matches():
    kept = _save_at(0, text="budget review for July")
    _save_at(1, text="budget draft nobody starred")
    _save_at(2, text="mid-sentence subbudget note")
    history.set_starred(kept, True)

    hits = await search.search_history_hybrid(
        "budget", limit=10, shown=history.HistoryFilter(starred=True)
    )

    assert [hit.id for hit in hits] == [kept]


@pytest.mark.asyncio
async def test_the_substring_lane_also_keeps_to_starred_entries():
    kept = _save_at(0, text="mid-sentence subbudget kept")
    _save_at(1, text="mid-sentence subbudget dropped")
    history.set_starred(kept, True)

    hits = search.search_history("dget", limit=10, shown=history.HistoryFilter(starred=True))

    assert [hit.id for hit in hits] == [kept]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["dictation", "meeting", "file"])
async def test_each_kind_alone_narrows_the_rows_the_total_and_the_days(client, source):
    ids = _one_of_each()
    words = {"dictation": 3, "meeting": 5, "file": 7}[source]

    page = (await client.get(f"/history?source={source}")).json()

    assert [e["id"] for e in page["entries"]] == [ids[source]]
    assert page["total"] == 1
    assert page["days"] == [{"date": "2026-07-28", "recordings": 1, "words": words}]


@pytest.mark.asyncio
async def test_no_kind_keeps_every_entry(client):
    ids = _one_of_each()

    page = (await client.get("/history")).json()

    assert [e["id"] for e in page["entries"]] == [ids["file"], ids["meeting"], ids["dictation"]]
    assert page["total"] == 3


@pytest.mark.asyncio
async def test_a_kind_and_starred_together_keep_only_starred_entries_of_that_kind(client):
    starred_meeting = _save_at(0, source="meeting")
    _save_at(1, source="meeting")
    history.set_starred(_save_at(2, source="dictation"), True)
    history.set_starred(starred_meeting, True)

    page = (await client.get("/history?source=meeting&starred=true")).json()

    assert [e["id"] for e in page["entries"]] == [starred_meeting]
    assert page["total"] == 1


@pytest.mark.asyncio
async def test_a_kind_pages_with_its_cursor_and_reads_newer_rows_of_that_kind_only(client):
    meetings = [_save_at(minute, source="meeting") for minute in (0, 2, 4)]
    for minute in (1, 3, 5):
        _save_at(minute, source="file")

    first = (await client.get("/history?limit=2&source=meeting")).json()
    cursor = first["next_cursor"]
    second = (
        await client.get(
            f"/history?limit=2&source=meeting&before_ts={cursor['ts']}&before_id={cursor['id']}"
        )
    ).json()
    oldest = second["newest_cursor"]
    newer = (
        await client.get(
            f"/history?limit=50&source=meeting&after_ts={oldest['ts']}&after_id={oldest['id']}"
        )
    ).json()

    assert [e["id"] for e in first["entries"]] == [meetings[2], meetings[1]]
    assert [e["id"] for e in second["entries"]] == [meetings[0]]
    assert second["next_cursor"] is None
    assert [e["id"] for e in newer["entries"]] == [meetings[1], meetings[2]]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["source=meeting", "starred=true"])
async def test_a_full_page_says_no_more_when_only_other_rows_lie_past_it(client, query):
    _save_at(0, source="file")
    for minute in (2, 4):
        history.set_starred(_save_at(minute, source="meeting"), True)

    page = (await client.get(f"/history?limit=2&{query}")).json()

    assert len(page["entries"]) == 2
    assert page["next_cursor"] is None


@pytest.mark.asyncio
async def test_search_under_a_kind_finds_only_that_kind_in_every_lane(client):
    ids = _one_of_each()
    _save_at(3, text="mid-sentence subbudget meeting", source="meeting")
    elsewhere = _save_at(4, text="mid-sentence subbudget file", source="file")

    exact = (await client.get("/history/search?q=budget&source=file")).json()
    substring = search.search_history(
        "dget", limit=10, shown=history.HistoryFilter(source="file")
    )

    assert {e["id"] for e in exact["entries"]} == {ids["file"], elsewhere}
    assert [hit.id for hit in substring] == [elsewhere, ids["file"]]


@pytest.mark.asyncio
async def test_the_meaning_lane_drops_hits_of_another_kind():
    ids = _one_of_each()
    meaning_hits = [
        search.HistorySearchHit(**history.get_page().entries[index].model_dump())
        for index in range(3)
    ]

    async def every_kind_by_meaning(q, limit):
        return meaning_hits

    with patch.object(search, "_semantic_lane", every_kind_by_meaning):
        hits = await search.search_history_hybrid(
            "nothing typed matches", limit=10, shown=history.HistoryFilter(source="meeting")
        )

    assert [hit.id for hit in hits] == [ids["meeting"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/history?source=voice", "/history/search?q=a&source=voice"])
async def test_an_unknown_kind_is_422(client, path):
    response = await client.get(path)

    assert response.status_code == 422

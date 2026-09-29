"""Starring an entry, and History and search narrowed to starred entries."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.transcripts import history, search


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)
    monkeypatch.setattr(history, "_page_total_cache", None)
    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()


def _save_at(minute: int, text: str = "note", words: int = 10) -> str:
    entry_id = history.save_entry(text=text, duration_ms=1, word_count=words).id
    ts = int(datetime(2026, 7, 28, 9, minute).timestamp() * 1000)
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        conn.execute("UPDATE entries SET ts = ? WHERE id = ?", (ts, entry_id))
        conn.execute("COMMIT")
        history.invalidate_derived_caches_locked()
    return entry_id


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

    newer = history.get_page(limit=50, after=oldest, starred_only=True)

    assert [e.id for e in newer.entries] == [ids[2]]


@pytest.mark.asyncio
async def test_search_under_the_starred_filter_finds_only_starred_matches():
    kept = _save_at(0, text="budget review for July")
    _save_at(1, text="budget draft nobody starred")
    _save_at(2, text="mid-sentence subbudget note")
    history.set_starred(kept, True)

    hits = await search.search_history_hybrid("budget", limit=10, starred_only=True)

    assert [hit.id for hit in hits] == [kept]


@pytest.mark.asyncio
async def test_the_substring_lane_also_keeps_to_starred_entries():
    kept = _save_at(0, text="mid-sentence subbudget kept")
    _save_at(1, text="mid-sentence subbudget dropped")
    history.set_starred(kept, True)

    hits = search.search_history("dget", limit=10, starred_only=True)

    assert [hit.id for hit in hits] == [kept]

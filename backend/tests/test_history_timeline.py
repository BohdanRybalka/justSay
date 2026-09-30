"""What the History timeline reads: whole-day totals beside each page, and the newer-rows read.

Local days are the machine's own, the rule SQLite's ``'localtime'`` applies, so every
timestamp here is built from a naive local ``datetime``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.transcripts import history, schema


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_page_total_cache", None)
    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()


def _local_ms(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, second).timestamp() * 1000)


def _save(ts: int, words: int | None) -> str:
    entry_id = history.save_entry(text=f"said at {ts}", duration_ms=1, word_count=words).id
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        conn.execute("UPDATE entries SET ts = ? WHERE id = ?", (ts, entry_id))
        conn.execute("COMMIT")
        history.invalidate_derived_caches_locked()
    return entry_id


def _days(page: history.HistoryPage) -> list[tuple[str | None, int, int]]:
    return [(day.date, day.recordings, day.words) for day in page.days]


def test_a_day_header_counts_the_whole_day_not_the_rows_on_the_page():
    for minute in range(5):
        _save(_local_ms(2026, 7, 28, 10, minute), 100)

    page = history.get_page(limit=2)

    assert len(page.entries) == 2
    assert _days(page) == [("2026-07-28", 5, 500)]


def test_a_page_across_midnight_names_both_local_days():
    _save(_local_ms(2026, 7, 27, 23, 59, 59), 10)
    _save(_local_ms(2026, 7, 28, 0, 0, 1), 20)
    _save(_local_ms(2026, 7, 28, 0, 30), None)

    page = history.get_page(limit=50)

    assert _days(page) == [("2026-07-28", 2, 20), ("2026-07-27", 1, 10)]


def test_days_outside_the_page_are_not_reported():
    _save(_local_ms(2026, 7, 26, 9, 0), 7)
    _save(_local_ms(2026, 7, 28, 9, 0), 1)
    _save(_local_ms(2026, 7, 28, 9, 5), 2)

    page = history.get_page(limit=2)

    assert _days(page) == [("2026-07-28", 2, 3)]


def test_rows_with_no_known_time_share_one_day_counted_over_all_of_them():
    _save(_local_ms(2026, 7, 28, 9, 0), 4)
    _save(schema.UNKNOWN_TS, 5)
    _save(schema.UNKNOWN_TS, 6)

    first = history.get_page(limit=2)
    assert _days(first) == [("2026-07-28", 1, 4), (None, 2, 11)]
    assert [e.timestamp for e in first.entries][1] is None


def test_the_newer_read_returns_only_rows_after_the_position_oldest_first():
    ids = [_save(_local_ms(2026, 7, 28, 9, minute), 1) for minute in range(4)]
    anchor = history.get_page(limit=2, before=None)
    assert anchor.newest_cursor is not None
    assert anchor.newest_cursor.id == ids[3]

    older_anchor = history.HistoryCursor(ts=_local_ms(2026, 7, 28, 9, 1), id=ids[1])
    newer = history.get_page(limit=50, after=older_anchor)

    assert [e.id for e in newer.entries] == ids[2:]
    assert newer.next_cursor is None
    assert newer.newest_cursor is not None and newer.newest_cursor.id == ids[3]
    assert _days(newer) == [("2026-07-28", 4, 4)]


def test_the_newer_read_past_the_newest_row_is_empty():
    _save(_local_ms(2026, 7, 28, 9, 0), 1)
    newest = history.get_page(limit=50).newest_cursor

    page = history.get_page(limit=50, after=newest)

    assert page.entries == []
    assert page.newest_cursor is None
    assert page.days == []


def test_the_newer_read_is_limited_and_resumes_from_its_own_newest_row():
    ids = [_save(_local_ms(2026, 7, 28, 9, minute), 1) for minute in range(5)]
    start = history.HistoryCursor(ts=_local_ms(2026, 7, 28, 9, 0), id=ids[0])

    first = history.get_page(limit=2, after=start)
    second = history.get_page(limit=2, after=first.newest_cursor)

    assert [e.id for e in first.entries] == ids[1:3]
    assert [e.id for e in second.entries] == ids[3:5]


def test_the_newer_read_seeks_the_composite_index():
    with history._lock:
        conn = history._ensure_conn_locked()
        plan = " ".join(
            str(row[-1])
            for row in conn.execute(
                f"EXPLAIN QUERY PLAN SELECT id FROM entries "
                f"{history._where(history._AFTER_CURSOR, None)}{history._NEWER_PAGE_ORDER}",
                {"after_ts": 0, "after_id": "", "row_limit": 1},
            ).fetchall()
        )
    assert "SEARCH" in plan and "entries_ts_id_idx" in plan, plan
    assert "TEMP B-TREE" not in plan.upper(), plan


@pytest.mark.asyncio
async def test_the_route_serves_newer_rows_and_refuses_mixed_cursors(client):
    ids = [_save(_local_ms(2026, 7, 28, 9, minute), 3) for minute in range(3)]
    anchor_ts = _local_ms(2026, 7, 28, 9, 0)

    response = await client.get(f"/history?limit=30&after_ts={anchor_ts}&after_id={ids[0]}")

    assert response.status_code == 200
    body = response.json()
    assert [e["id"] for e in body["entries"]] == ids[1:]
    assert body["days"] == [{"date": "2026-07-28", "recordings": 3, "words": 9}]
    assert body["newest_cursor"]["id"] == ids[2]
    assert (await client.get(f"/history?after_ts={anchor_ts}")).status_code == 422
    assert (await client.get(f"/history?after_id={ids[0]}")).status_code == 422
    both = f"/history?before_ts={anchor_ts}&before_id=x&after_ts={anchor_ts}&after_id=y"
    assert (await client.get(both)).status_code == 422

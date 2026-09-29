"""Insights figures: today, the month, the streak — dictations only, local days.

Every timestamp is built from a naive local ``datetime``, the rule SQLite's
``'localtime'`` applies to the rows.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.transcripts import history, insights


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(insights, "_cache", None)
    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()


NOW = datetime(2026, 8, 20, 21, 30)


def _save(
    when: datetime,
    words: int,
    seconds: float | None = None,
    source: history.EntrySource = "dictation",
) -> None:
    entry_id = history.save_entry(
        text="said", duration_ms=1, word_count=words,
        audio_duration_seconds=seconds, source=source,
    ).id
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute(
            "UPDATE entries SET ts = ? WHERE id = ?", (int(when.timestamp() * 1000), entry_id)
        )
        history.invalidate_derived_caches_locked()


def test_today_counts_dictations_since_local_midnight():
    _save(datetime(2026, 8, 19, 23, 59, 59), 500)
    _save(datetime(2026, 8, 20, 0, 0, 1), 100)
    _save(datetime(2026, 8, 20, 9, 0), 20)

    today = insights.compute_insights(now=NOW).today

    assert (today.words, today.recordings) == (120, 2)


def test_the_month_starts_on_the_first_and_ends_now():
    _save(datetime(2026, 7, 31, 23, 59, 59), 1000, 60)
    _save(datetime(2026, 8, 1, 0, 0, 1), 400, 120)
    _save(datetime(2026, 8, 20, 9, 0), 200, 60)
    _save(datetime(2026, 8, 21, 9, 0), 3000, 60)

    month = insights.compute_insights(now=NOW).month

    assert (month.words, month.recordings) == (600, 2)
    assert month.speaking_seconds == 180.0


def test_files_and_meetings_count_nowhere():
    _save(datetime(2026, 8, 20, 9, 0), 80, 60)
    _save(datetime(2026, 8, 20, 10, 0), 6000, 2400, source="meeting")
    _save(datetime(2026, 8, 19, 10, 0), 5000, 2400, source="file")

    figures = insights.compute_insights(now=NOW)

    assert (figures.today.words, figures.today.recordings) == (80, 1)
    assert (figures.month.words, figures.month.speaking_seconds) == (80, 60.0)
    assert (figures.streak.current_days, figures.streak.longest_days) == (1, 1)


def test_typing_time_is_the_same_words_at_forty_wpm():
    _save(datetime(2026, 8, 20, 9, 0), 800, 300)

    month = insights.compute_insights(now=NOW).month

    assert (month.typing_seconds, month.speaking_seconds) == (1200.0, 300.0)
    assert month.pace_wpm == 160


def test_rows_without_audio_length_keep_their_words_out_of_time_figures():
    _save(datetime(2026, 8, 20, 9, 0), 400, 120)
    _save(datetime(2026, 8, 20, 10, 0), 4000)

    month = insights.compute_insights(now=NOW).month

    assert month.words == 4400
    assert (month.speaking_seconds, month.typing_seconds) == (120.0, 600.0)
    assert month.pace_wpm == 200


def test_no_speaking_time_has_no_pace():
    _save(datetime(2026, 8, 20, 9, 0), 10)
    _save(datetime(2026, 8, 20, 10, 0), 10, 0.04)

    month = insights.compute_insights(now=NOW).month

    assert (month.speaking_seconds, month.pace_wpm) == (0.0, None)


def test_the_streak_holds_through_today_until_the_first_dictation():
    for day in (16, 17, 18, 19):
        _save(datetime(2026, 8, day, 12, 0), 10)

    streak = insights.compute_insights(now=NOW).streak

    assert (streak.current_days, streak.longest_days) == (4, 4)


def test_a_missed_day_ends_the_streak_and_the_longest_run_is_remembered():
    for day in (1, 2, 3, 4, 5, 18, 20):
        _save(datetime(2026, 8, day, 12, 0), 10)

    streak = insights.compute_insights(now=NOW).streak

    assert (streak.current_days, streak.longest_days) == (1, 5)


def test_a_streak_runs_across_a_month_end_and_skips_future_rows():
    for when in (
        datetime(2026, 7, 31, 23, 0), datetime(2026, 8, 1, 0, 30), datetime(2026, 8, 2, 8, 0),
    ):
        _save(when, 10)
    _save(datetime(2026, 8, 3, 8, 0), 10)

    streak = insights.compute_insights(now=datetime(2026, 8, 2, 9, 0)).streak

    assert (streak.current_days, streak.longest_days) == (3, 3)


def test_a_write_refreshes_the_figures():
    _save(datetime(2026, 8, 20, 9, 0), 10)
    assert insights.compute_insights(now=NOW).today.words == 10

    _save(datetime(2026, 8, 20, 10, 0), 5)

    assert insights.compute_insights(now=NOW).today.words == 15


def test_the_figures_move_to_a_new_day_without_a_write():
    _save(datetime(2026, 8, 20, 9, 0), 10)
    assert insights.compute_insights(now=NOW).today.words == 10

    tomorrow = insights.compute_insights(now=datetime(2026, 8, 21, 0, 5))

    assert tomorrow.today.words == 0
    assert tomorrow.streak.current_days == 1


@pytest.mark.asyncio
async def test_insights_endpoint_answers_with_the_figures(client):
    _save(datetime.now(), 42, 30)

    resp = await client.get("/insights")

    assert resp.status_code == 200
    data = resp.json()
    assert data["today"] == {"words": 42, "recordings": 1}
    assert data["streak"] == {"current_days": 1, "longest_days": 1}
    assert set(data["month"]) == {
        "words", "recordings", "speaking_seconds", "typing_seconds", "pace_wpm",
    }

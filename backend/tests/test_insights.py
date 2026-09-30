"""Insights figures: today, the month, the streak, the voice — dictations only, local days.

Every timestamp is built from a naive local ``datetime``, the rule SQLite's
``'localtime'`` applies to the rows.
"""

from __future__ import annotations

import asyncio
import time
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
    text: str = "said",
) -> None:
    entry_id = history.save_entry(
        text=text, duration_ms=1, word_count=words,
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


def test_chart_days_end_today_oldest_first_and_zero_filled():
    _save(datetime(2026, 8, 13, 23, 59, 59), 7)
    _save(datetime(2026, 8, 14, 0, 0, 1), 30)
    _save(datetime(2026, 8, 14, 9, 0), 12)
    _save(datetime(2026, 8, 20, 9, 0), 5)
    _save(datetime(2026, 8, 20, 10, 0), 900, source="meeting")
    _save(datetime(2026, 8, 21, 9, 0), 400)

    days = insights.compute_insights(7, now=NOW).days

    assert [(d.date.isoformat(), d.words) for d in days] == [
        ("2026-08-14", 42), ("2026-08-15", 0), ("2026-08-16", 0), ("2026-08-17", 0),
        ("2026-08-18", 0), ("2026-08-19", 0), ("2026-08-20", 5),
    ]


def test_thirty_chart_days_reach_back_across_the_month_end():
    _save(datetime(2026, 7, 22, 12, 0), 11)
    _save(datetime(2026, 7, 21, 12, 0), 99)

    days = insights.compute_insights(30, now=NOW).days

    assert len(days) == 30
    assert (days[0].date.isoformat(), days[0].words) == ("2026-07-22", 11)
    assert days[-1].date.isoformat() == "2026-08-20"


def test_the_previous_period_is_the_same_number_of_days_just_before():
    _save(datetime(2026, 8, 6, 23, 59, 59), 1000)
    _save(datetime(2026, 8, 7, 0, 0, 1), 20)
    _save(datetime(2026, 8, 13, 23, 59, 59), 3)
    _save(datetime(2026, 8, 13, 12, 0), 900, source="file")
    _save(datetime(2026, 8, 14, 0, 0, 1), 50)

    figures = insights.compute_insights(7, now=NOW)

    assert figures.previous_period_words == 23
    assert sum(d.words for d in figures.days) == 50


def test_each_span_reads_its_own_days():
    _save(datetime(2026, 8, 1, 9, 0), 10)

    assert len(insights.compute_insights(30, now=NOW).days) == 30
    week = insights.compute_insights(7, now=NOW)

    assert len(week.days) == 7
    assert week.previous_period_words == 0


def test_vocabulary_counts_every_distinct_dictated_word_stopwords_included():
    _save(datetime(2019, 3, 1, 9, 0), 5, text="The cat and THE dog")
    _save(datetime(2026, 8, 20, 9, 0), 3, text="the кіт і cat")
    _save(datetime(2026, 8, 20, 10, 0), 3, source="meeting", text="colleagues say different things")

    assert insights.compute_insights(now=NOW).vocabulary == 6


def test_the_peak_hour_is_the_busiest_over_thirty_days_with_its_neighbours_share():
    _save(datetime(2026, 7, 21, 18, 0), 5000)
    _save(datetime(2026, 7, 22, 17, 59), 100)
    _save(datetime(2026, 8, 20, 18, 30), 400)
    _save(datetime(2026, 8, 19, 19, 5), 100)
    _save(datetime(2026, 8, 18, 9, 0), 200)
    _save(datetime(2026, 8, 20, 18, 0), 9000, source="meeting")

    peak = insights.compute_insights(now=NOW).peak_hour

    assert peak is not None
    assert (peak.hour, peak.share) == (18, 0.75)


def test_the_peak_hour_wraps_midnight_and_takes_the_earliest_on_a_tie():
    _save(datetime(2026, 8, 20, 0, 10), 100)
    _save(datetime(2026, 8, 19, 23, 50), 50)
    _save(datetime(2026, 8, 19, 12, 0), 100)
    _save(datetime(2026, 8, 19, 22, 0), 50)

    peak = insights.compute_insights(now=NOW).peak_hour

    assert peak is not None
    assert (peak.hour, peak.share) == (0, 0.5)


def test_no_words_in_thirty_days_has_no_peak_hour():
    _save(datetime(2026, 7, 21, 18, 0), 500)

    assert insights.compute_insights(now=NOW).peak_hour is None


def test_the_longest_run_is_the_longest_dictation_ever():
    _save(datetime(2019, 3, 1, 9, 0), 715, 393.04)
    _save(datetime(2026, 8, 20, 9, 0), 40, 60)
    _save(datetime(2026, 8, 20, 10, 0), 9000)
    _save(datetime(2026, 8, 20, 11, 0), 6000, 2892, source="meeting")

    longest = insights.compute_insights(now=NOW).longest

    assert longest is not None
    assert (longest.seconds, longest.words) == (393.0, 715)


def test_no_timed_dictation_has_no_longest_run():
    _save(datetime(2026, 8, 20, 9, 0), 40)

    assert insights.compute_insights(now=NOW).longest is None


def test_meetings_this_week_are_the_last_seven_local_days():
    _save(datetime(2026, 8, 13, 23, 59, 59), 100, 600, source="meeting")
    _save(datetime(2026, 8, 14, 0, 0, 1), 100, 2892, source="meeting")
    _save(datetime(2026, 8, 20, 9, 0), 100, 1600.04, source="meeting")
    _save(datetime(2026, 8, 20, 12, 0), 100, source="meeting")
    _save(datetime(2026, 8, 21, 9, 0), 100, 600, source="meeting")
    _save(datetime(2026, 8, 20, 10, 0), 100, 600)

    meetings = insights.compute_insights(now=NOW).meetings_week

    assert (meetings.count, meetings.seconds) == (3, 4492.0)


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
    assert len(data["days"]) == 30
    assert data["days"][-1]["words"] == 42
    assert data["previous_period_words"] == 0
    assert data["vocabulary"] == 1
    assert data["peak_hour"]["share"] == 1.0
    assert data["longest"] == {"seconds": 30.0, "words": 42}
    assert data["meetings_week"] == {"count": 0, "seconds": 0.0}


@pytest.mark.asyncio
async def test_insights_leave_the_event_loop_free(monkeypatch):
    """A miss tokenises every dictation for the vocabulary, on the loop that also
    serves ``/health`` and the widget poll unless it runs in a thread."""
    from app.transcripts import insights_router, words

    rows = 100
    for index in range(rows):
        _save(datetime(2026, 8, 20, 9, 0), 5, text=f"word {index}")
    real_tokenize = words.tokenize

    def slow_tokenize(text):
        time.sleep(0.001)
        return real_tokenize(text)

    monkeypatch.setattr(words, "tokenize", slow_tokenize)
    ticks = 0

    async def competitor():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    race = asyncio.ensure_future(competitor())
    figures = await insights_router.get_insights()
    race.cancel()

    assert figures.vocabulary == rows + 1
    assert ticks > 50, f"the loop ticked {ticks} times while /insights tokenised {rows} rows"


@pytest.mark.asyncio
async def test_insights_endpoint_takes_seven_or_thirty_days_only(client):
    week = await client.get("/insights?days=7")
    other = await client.get("/insights?days=10")

    assert week.status_code == 200
    assert len(week.json()["days"]) == 7
    assert other.status_code == 422

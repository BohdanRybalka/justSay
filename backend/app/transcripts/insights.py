"""Insights — the launch panel's personal figures, over dictations only.

Files and meetings carry other voices, so every figure reads rows whose ``source``
is ``dictation``. Days are the machine's local days, the rule ``/history`` applies.
Time figures skip rows that do not know their audio length; word figures keep them.
Meetings this week is the one figure that reads ``meeting`` rows. Figures cache on
``history.derived_generation_locked``, the local date and the chart span.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import IntEnum
from typing import NamedTuple

from pydantic import BaseModel

from app.transcripts import history, words

TYPING_WPM = 40
PEAK_HOUR_DAYS = 30
MEETINGS_WEEK_DAYS = 7


class ChartSpan(IntEnum):
    WEEK = 7
    MONTH = 30

_DAYS_SQL = (
    f"SELECT {history._LOCAL_DAY} AS day, COUNT(*), COALESCE(SUM(word_count), 0), "
    "COALESCE(SUM(CASE WHEN audio_duration_seconds IS NOT NULL THEN word_count END), 0), "
    "COALESCE(SUM(audio_duration_seconds), 0.0) "
    f"FROM entries WHERE source = 'dictation' AND {history._KNOWN_TS} GROUP BY day"
)
_LOCAL_HOUR = "CAST(strftime('%H', ts / 1000, 'unixepoch', 'localtime') AS INTEGER)"
_HOURS_SQL = (
    f"SELECT {_LOCAL_HOUR} AS hour, COALESCE(SUM(word_count), 0) FROM entries "
    f"WHERE source = 'dictation' AND {history._KNOWN_TS} "
    f"AND {history._LOCAL_DAY} BETWEEN ? AND ? GROUP BY hour"
)
_LONGEST_SQL = (
    "SELECT audio_duration_seconds, COALESCE(word_count, 0) FROM entries "
    "WHERE source = 'dictation' AND audio_duration_seconds > 0 "
    "ORDER BY audio_duration_seconds DESC LIMIT 1"
)
_MEETINGS_SQL = (
    "SELECT COUNT(*), COALESCE(SUM(audio_duration_seconds), 0.0) FROM entries "
    f"WHERE source = 'meeting' AND {history._KNOWN_TS} AND {history._LOCAL_DAY} BETWEEN ? AND ?"
)


class TodayFigures(BaseModel):
    words: int
    recordings: int


class MonthFigures(BaseModel):
    """The calendar month to date. ``typing_seconds`` is the same words typed at
    ``TYPING_WPM``; the time saved is it minus ``speaking_seconds``, never below zero.
    ``pace_wpm`` is ``None`` without speaking time.
    """

    words: int
    recordings: int
    speaking_seconds: float
    typing_seconds: float
    pace_wpm: int | None


class StreakFigures(BaseModel):
    """Days in a row with a dictation, ending today or, before today's first, yesterday."""

    current_days: int
    longest_days: int


class DayWords(BaseModel):
    date: date
    words: int


class PeakHour(BaseModel):
    """The local hour with most dictated words over the last ``PEAK_HOUR_DAYS`` days, the
    earliest on a tie; ``share`` is the part of those words said from the hour before it
    through the hour after it.
    """

    hour: int
    share: float


class LongestRun(BaseModel):
    """The dictation with the longest known audio: its length and its words."""

    seconds: float
    words: int


class MeetingsWeek(BaseModel):
    """Meetings recorded over the last ``MEETINGS_WEEK_DAYS`` local days, today included."""

    count: int
    seconds: float


class Insights(BaseModel):
    """``days`` are the chart's last days up to today, oldest first, zero-filled;
    ``previous_period_words`` sums the same number of days just before them.
    ``vocabulary`` counts the distinct tokens of every dictation, stop-words included.
    """

    today: TodayFigures
    month: MonthFigures
    streak: StreakFigures
    days: list[DayWords]
    previous_period_words: int
    vocabulary: int
    peak_hour: PeakHour | None
    longest: LongestRun | None
    meetings_week: MeetingsWeek


class _Day(NamedTuple):
    recordings: int
    words: int
    timed_words: int
    speaking_seconds: float


class _Voice(NamedTuple):
    vocabulary: int
    peak_hour: PeakHour | None
    longest: LongestRun | None
    meetings_week: MeetingsWeek


_NO_DICTATION = _Day(0, 0, 0, 0.0)

_cache: tuple[int, date, int, Insights] | None = None


def compute_insights(span: int = ChartSpan.MONTH, now: datetime | None = None) -> Insights:
    """Every figure of the panel with ``span`` chart days, as of ``now`` (local clock).

    A miss may tokenise every dictation, so run it off the event loop.
    """
    global _cache
    today = (now or datetime.now().astimezone()).date()
    with history._lock:
        if _cache is not None and _cache[:3] == (history.derived_generation_locked(), today, span):
            return _cache[3]
    tokens = words.dictation_tokens()
    with history._lock:
        generation = history.derived_generation_locked()
        conn = history._ensure_conn_locked()
        rows = conn.execute(_DAYS_SQL).fetchall()
        days = {date.fromisoformat(row[0]): _Day(*row[1:]) for row in rows}
        voice = _Voice(
            vocabulary=len(tokens.counts),
            peak_hour=_peak_hour(dict(conn.execute(_HOURS_SQL, _last_days(today, PEAK_HOUR_DAYS)))),
            longest=_longest(conn.execute(_LONGEST_SQL).fetchone()),
            meetings_week=_meetings(
                conn.execute(_MEETINGS_SQL, _last_days(today, MEETINGS_WEEK_DAYS)).fetchone()
            ),
        )
        insights = _figures({day: d for day, d in days.items() if day <= today}, today, span, voice)
        if tokens.generation == generation:
            _cache = (generation, today, span, insights)
        return insights


def _last_days(today: date, count: int) -> tuple[str, str]:
    return (today - timedelta(days=count - 1)).isoformat(), today.isoformat()


def _peak_hour(words_by_hour: dict[int, int]) -> PeakHour | None:
    total = sum(words_by_hour.values())
    if total == 0:
        return None
    hour = max(range(24), key=lambda h: words_by_hour.get(h, 0))
    around = sum(words_by_hour.get((hour + step) % 24, 0) for step in (-1, 0, 1))
    return PeakHour(hour=hour, share=round(around / total, 3))


def _longest(row: tuple[float, int] | None) -> LongestRun | None:
    return None if row is None else LongestRun(seconds=round(row[0], 1), words=row[1])


def _meetings(row: tuple[int, float]) -> MeetingsWeek:
    return MeetingsWeek(count=row[0], seconds=round(row[1], 1))


def _figures(days: dict[date, _Day], today: date, span: int, voice: _Voice) -> Insights:
    month = [d for day, d in days.items() if day >= today.replace(day=1)]
    speaking = round(sum(d.speaking_seconds for d in month), 1)
    timed_words = sum(d.timed_words for d in month)
    typing = timed_words * 60 / TYPING_WPM
    today_day = days.get(today, _NO_DICTATION)
    return Insights(
        today=TodayFigures(words=today_day.words, recordings=today_day.recordings),
        month=MonthFigures(
            words=sum(d.words for d in month),
            recordings=sum(d.recordings for d in month),
            speaking_seconds=speaking,
            typing_seconds=round(typing, 1),
            pace_wpm=round(timed_words * 60 / speaking) if speaking > 0 else None,
        ),
        streak=_streak(set(days), today),
        days=[DayWords(date=day, words=_words(days, day)) for day in _span_days(today, span)],
        previous_period_words=sum(
            _words(days, day) for day in _span_days(today - timedelta(days=span), span)
        ),
        **voice._asdict(),
    )


def _span_days(last: date, span: int) -> list[date]:
    return [last - timedelta(days=back) for back in range(span - 1, -1, -1)]


def _words(days: dict[date, _Day], day: date) -> int:
    return days.get(day, _NO_DICTATION).words


def _streak(days: set[date], today: date) -> StreakFigures:
    end = today if today in days else today - timedelta(days=1)
    current = 0
    while end - timedelta(days=current) in days:
        current += 1
    longest = 0
    for first in days - {day + timedelta(days=1) for day in days}:
        run = 1
        while first + timedelta(days=run) in days:
            run += 1
        longest = max(longest, run)
    return StreakFigures(current_days=current, longest_days=longest)

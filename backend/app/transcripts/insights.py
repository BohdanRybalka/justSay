"""Insights — the launch panel's personal figures, over dictations only.

Files and meetings carry other voices, so every figure reads rows whose ``source``
is ``dictation``. Days are the machine's local days, the rule ``/history`` applies.
Time figures skip rows that do not know their audio length; word figures keep them.
Figures cache on ``history.derived_generation_locked`` and the local date.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import NamedTuple

from pydantic import BaseModel

from app.transcripts import history

TYPING_WPM = 40

_DAYS_SQL = (
    f"SELECT {history._LOCAL_DAY} AS day, COUNT(*), COALESCE(SUM(word_count), 0), "
    "COALESCE(SUM(CASE WHEN audio_duration_seconds IS NOT NULL THEN word_count END), 0), "
    "COALESCE(SUM(audio_duration_seconds), 0.0) "
    f"FROM entries WHERE source = 'dictation' AND {history._KNOWN_TS} GROUP BY day"
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


class Insights(BaseModel):
    today: TodayFigures
    month: MonthFigures
    streak: StreakFigures


class _Day(NamedTuple):
    recordings: int
    words: int
    timed_words: int
    speaking_seconds: float


_NO_DICTATION = _Day(0, 0, 0, 0.0)

_cache: tuple[int, date, Insights] | None = None


def compute_insights(now: datetime | None = None) -> Insights:
    """Today, this month and the streak, as of ``now`` (the local clock by default)."""
    global _cache
    today = (now or datetime.now().astimezone()).date()
    with history._lock:
        generation = history.derived_generation_locked()
        if _cache is not None and _cache[:2] == (generation, today):
            return _cache[2]
        rows = history._ensure_conn_locked().execute(_DAYS_SQL).fetchall()
        days = {date.fromisoformat(row[0]): _Day(*row[1:]) for row in rows}
        insights = _figures({day: d for day, d in days.items() if day <= today}, today)
        _cache = (generation, today, insights)
        return insights


def _figures(days: dict[date, _Day], today: date) -> Insights:
    month = [d for day, d in days.items() if day >= today.replace(day=1)]
    speaking = sum(d.speaking_seconds for d in month)
    timed_words = sum(d.timed_words for d in month)
    typing = timed_words * 60 / TYPING_WPM
    today_day = days.get(today, _NO_DICTATION)
    return Insights(
        today=TodayFigures(words=today_day.words, recordings=today_day.recordings),
        month=MonthFigures(
            words=sum(d.words for d in month),
            recordings=sum(d.recordings for d in month),
            speaking_seconds=round(speaking, 1),
            typing_seconds=round(typing, 1),
            pace_wpm=round(timed_words * 60 / speaking) if speaking > 0 else None,
        ),
        streak=_streak(set(days), today),
    )


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

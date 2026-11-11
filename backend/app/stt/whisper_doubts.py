"""Signs that a Whisper ``verbose_json`` answer for one dictation piece went wrong.

A segment is non-speech when its ``no_speech_prob`` is over `NO_SPEECH_PROB` and its
``avg_logprob`` under `LOW_AVG_LOGPROB`, Whisper's own skip rule; it loops when its
``compression_ratio`` is over `LOOPING_COMPRESSION_RATIO`. Ukrainian text holding a letter
only Russian uses has drifted into Russian. A field missing from a segment raises no doubt.
"""

from __future__ import annotations

from dataclasses import dataclass

NO_SPEECH_PROB = 0.6
LOW_AVG_LOGPROB = -1.0
LOOPING_COMPRESSION_RATIO = 2.4
RUSSIAN_ONLY_LETTERS = frozenset("ыэъёЫЭЪЁ")


@dataclass(frozen=True)
class AnswerDoubts:
    """How many segments an answer has, how many are non-speech, and whether it loops or drifts."""

    segments: int
    non_speech_segments: int
    is_looping: bool
    is_russian: bool

    @property
    def is_non_speech(self) -> bool:
        return self.segments > 0 and self.non_speech_segments == self.segments

    @property
    def signals(self) -> int:
        return (self.non_speech_segments > 0) + self.is_looping + self.is_russian


def _number(segment: object, name: str) -> float | None:
    value = segment.get(name) if isinstance(segment, dict) else getattr(segment, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _is_non_speech(segment: object) -> bool:
    no_speech, logprob = _number(segment, "no_speech_prob"), _number(segment, "avg_logprob")
    return (
        no_speech is not None
        and logprob is not None
        and no_speech > NO_SPEECH_PROB
        and logprob < LOW_AVG_LOGPROB
    )


def _is_looping(segment: object) -> bool:
    ratio = _number(segment, "compression_ratio")
    return ratio is not None and ratio > LOOPING_COMPRESSION_RATIO


def read_doubts(text: str, segments: object, language: str) -> AnswerDoubts:
    """The doubts ``text`` and its ``segments`` raise for a piece asked in ``language``.

    Only a ``list`` or ``tuple`` of segments is read; anything else counts as none.
    """
    listed = list(segments) if isinstance(segments, (list, tuple)) else []
    return AnswerDoubts(
        segments=len(listed),
        non_speech_segments=sum(_is_non_speech(segment) for segment in listed),
        is_looping=any(_is_looping(segment) for segment in listed),
        is_russian=language == "uk" and not RUSSIAN_ONLY_LETTERS.isdisjoint(text),
    )

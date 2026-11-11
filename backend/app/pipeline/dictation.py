"""A Cloud dictation sent in Whisper-window pieces: silent edges trimmed, long speech cut in pauses.

Only what is surely silent before the first sound and after the last is cut off. Up to
``WHISPER_WINDOW_SECONDS`` the rest goes whole; longer, it is cut at its longest pause between
``SHORTEST_PIECE_SECONDS`` and ``DICTATION_PIECE_SECONDS`` into the piece, else hard there, never
overlapping. A doubted piece is asked once more and the surer answer kept; a piece either answer
hears as no speech adds nothing. Texts join with a space, or none in a language written without.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.audio.config import audio_settings
from app.audio.pauses import sound_span
from app.core.scratch import discard_scratch_file
from app.pipeline.chunking import (
    SHORT_MAX_PAUSE_SECONDS,
    SHORT_MAX_PAUSES,
    PiecePacer,
    RouteStoppedError,
    _pause_relative_to,
    _read_seconds,
    _write_flac,
    on_mono_audio,
    reported_language,
    transcribe_with_pauses,
)
from app.stt.base import WHISPER_WINDOW_SECONDS, STTProvider, TranscriptionResult
from app.stt.languages import UNSPACED_LANGUAGES

log = logging.getLogger(__name__)

DICTATION_PIECE_SECONDS = 28.0
SHORTEST_PIECE_SECONDS = 10.0


def plan_dictation_pieces(
    duration: float, find_pause: Callable[[float, float], float | None]
) -> list[tuple[float, float]]:
    """``(start, end)`` pieces covering ``duration``; one up to ``WHISPER_WINDOW_SECONDS``.

    Longer, a piece ends at ``find_pause(start, end)``, the longest pause from
    ``SHORTEST_PIECE_SECONDS`` in up to ``DICTATION_PIECE_SECONDS``, or hard at that limit when it
    answers ``None``. A cut always leaves at least ``SHORTEST_PIECE_SECONDS`` after it.
    """
    pieces: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > WHISPER_WINDOW_SECONDS:
        limit = min(start + DICTATION_PIECE_SECONDS, duration - SHORTEST_PIECE_SECONDS)
        pause = find_pause(start + SHORTEST_PIECE_SECONDS, limit)
        end = limit if pause is None else pause
        pieces.append((start, end))
        start = end
    pieces.append((start, duration))
    return pieces


def _sound_of(mono: Path, total: float) -> tuple[float, float]:
    """Where the sound of ``mono`` begins and ends; all of it when none is heard."""
    levels = _read_seconds(mono, 0.0, total).astype(np.float32) / 32768.0
    return sound_span(levels, audio_settings) or (0.0, total)


@dataclass(frozen=True)
class _PieceAnswer:
    result: TranscriptionResult
    is_heard: bool


def _is_unlikely_speech(result: TranscriptionResult, no_speech_threshold: float) -> bool:
    prob = result.no_speech_prob
    return prob is not None and prob > no_speech_threshold


def _hears_no_speech(result: TranscriptionResult, no_speech_threshold: float) -> bool:
    return not result.text or _is_unlikely_speech(result, no_speech_threshold)


def _doubts(result: TranscriptionResult, no_speech_threshold: float) -> int:
    return (result.doubts or 0) + _is_unlikely_speech(result, no_speech_threshold)


async def _send(
    stt: STTProvider,
    piece: Path,
    language: str,
    seconds: float | None,
    pacer: PiecePacer,
    *,
    second_opinion: bool,
) -> TranscriptionResult:
    return await transcribe_with_pauses(
        stt,
        piece,
        language,
        seconds,
        pacer,
        max_pause=SHORT_MAX_PAUSE_SECONDS,
        max_pauses=SHORT_MAX_PAUSES,
        provider_options={"dictation": True, "second_opinion": second_opinion},
    )


async def _transcribe_piece(
    stt: STTProvider,
    piece: Path,
    language: str,
    seconds: float | None,
    pacer: PiecePacer,
    no_speech_threshold: float,
) -> _PieceAnswer:
    """One piece's answer, asked again when it raises doubts or sounds like no speech.

    A piece adds nothing when its first answer has no text, or when the second hears no speech
    either; else the answer with fewer doubts is kept, the first on a tie or a failure.
    """
    first = await _send(stt, piece, language, seconds, pacer, second_opinion=False)
    doubts = _doubts(first, no_speech_threshold)
    if not first.text or not doubts:
        return _PieceAnswer(first, bool(first.text))
    log.warning("%s: a dictation piece raised %d doubts; again", stt.model_name, doubts)
    try:
        second = await _send(stt, piece, language, seconds, pacer, second_opinion=True)
    except RouteStoppedError:
        raise
    except Exception:
        log.warning("Asking again failed; keeping the first answer", exc_info=True)
        return _PieceAnswer(first, True)
    if _hears_no_speech(second, no_speech_threshold):
        log.info("The second answer hears no speech either; the piece adds nothing")
        return _PieceAnswer(second, False)
    if _doubts(second, no_speech_threshold) < doubts:
        return _PieceAnswer(second, True)
    log.info("The second answer is no surer; keeping the first")
    return _PieceAnswer(first, True)


def _joined(
    answers: list[_PieceAnswer], language: str, no_speech_threshold: float
) -> TranscriptionResult:
    """The heard pieces' texts joined; with none heard, ``no_speech_prob`` says so."""
    heard = [answer.result for answer in answers if answer.is_heard]
    unheard = [
        answer.result.no_speech_prob
        for answer in answers
        if _is_unlikely_speech(answer.result, no_speech_threshold)
    ]
    tokens = [a.result.tokens_used for a in answers if a.result.tokens_used is not None]
    languages = Counter(result.detected_language for result in heard if result.detected_language)
    spoken = reported_language(language, languages)
    return TranscriptionResult(
        text=("" if spoken in UNSPACED_LANGUAGES else " ").join(result.text for result in heard),
        tokens_used=sum(tokens) if tokens else None,
        detected_language=spoken,
        no_speech_prob=None if heard else max(unheard, default=1.0),
    )


async def transcribe_dictation(
    stt: STTProvider,
    audio_path: Path,
    language: str,
    duration: float | None,
    pacer: PiecePacer,
    *,
    no_speech_threshold: float,
) -> TranscriptionResult:
    """Transcribe one dictation in pieces; scratch files sit beside ``audio_path`` and are removed.

    A piece adds nothing when it has no text, or when both answers hear no speech: a
    ``no_speech_prob`` over ``no_speech_threshold``. The route is checked before every send.
    """

    async def piece_answer(piece: Path, seconds: float | None) -> _PieceAnswer:
        return await _transcribe_piece(stt, piece, language, seconds, pacer, no_speech_threshold)

    async def whole() -> TranscriptionResult:
        return _joined([await piece_answer(audio_path, duration)], language, no_speech_threshold)

    async def cut(mono: Path, total: float) -> TranscriptionResult:
        begin, end = await asyncio.to_thread(_sound_of, mono, total)
        pieces = await asyncio.to_thread(
            plan_dictation_pieces,
            end - begin,
            lambda start, stop: _pause_relative_to(mono, begin, start, stop),
        )
        log.info("Dictation: %.1fs of %.1fs sent in %d pieces", end - begin, total, len(pieces))
        answers: list[_PieceAnswer] = []
        for index, (start, stop) in enumerate(pieces):
            piece = audio_path.with_name(f"{audio_path.stem}-piece{index}.flac")
            try:
                samples = await asyncio.to_thread(
                    _read_seconds, mono, begin + start, begin + stop
                )
                await asyncio.to_thread(_write_flac, piece, samples)
                answers.append(await piece_answer(piece, stop - start))
            finally:
                discard_scratch_file(piece)
        return _joined(answers, language, no_speech_threshold)

    return await on_mono_audio(audio_path, whole, cut)

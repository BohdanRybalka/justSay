"""A Cloud dictation sent in Whisper-window pieces: edges trimmed, long speech cut in its pauses.

The silence before the first speech and after the last is cut off. Speech longer than
``DICTATION_PIECE_SECONDS`` is cut at its longest pause that leaves both sides at least
``SHORTEST_PIECE_SECONDS``, else hard at the limit; pieces never overlap and their texts are
joined with a space. A piece with doubts is asked once for a second opinion and the answer with
fewer doubts is kept; a piece heard as non-speech adds nothing. Undecodable audio goes whole.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.audio.config import audio_settings
from app.audio.pauses import longest_pause, speech_profile
from app.core.audio_formats import UndecodableAudioError, decode_to_mono_wav
from app.core.scratch import discard_scratch_file
from app.pipeline.chunking import (
    PIECE_RATE,
    SHORT_MAX_PAUSE_SECONDS,
    SHORT_MAX_PAUSES,
    PiecePacer,
    RouteStoppedError,
    reported_language,
    transcribe_with_pauses,
)
from app.stt.base import STTProvider, TranscriptionResult

log = logging.getLogger(__name__)

DICTATION_PIECE_SECONDS = 28.0
SHORTEST_PIECE_SECONDS = 10.0


def plan_dictation_pieces(
    duration: float, find_pause: Callable[[float, float], float | None]
) -> list[tuple[float, float]]:
    """``(start, end)`` pieces covering ``duration``, none longer than ``DICTATION_PIECE_SECONDS``.

    ``find_pause(start, end)`` answers the time of the longest pause between them, or ``None``.
    A cut leaves at least ``SHORTEST_PIECE_SECONDS`` on both sides of it.
    """
    pieces: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > DICTATION_PIECE_SECONDS:
        limit = min(start + DICTATION_PIECE_SECONDS, duration - SHORTEST_PIECE_SECONDS)
        pause = find_pause(start + SHORTEST_PIECE_SECONDS, limit)
        end = limit if pause is None else pause
        pieces.append((start, end))
        start = end
    pieces.append((start, duration))
    return pieces


def _speech_between_edges(mono: Path) -> np.ndarray:
    """The samples of ``mono`` from its first speech to its last; all of them if none is heard."""
    import soundfile as sf

    samples, _rate = sf.read(str(mono), dtype="float32")
    spans = speech_profile(samples, audio_settings).spans
    if not spans:
        return samples
    return samples[round(spans[0][0] * PIECE_RATE) : round(spans[-1][1] * PIECE_RATE)]


def _pause_in(samples: np.ndarray, start: float, end: float) -> float | None:
    window = samples[round(start * PIECE_RATE) : round(end * PIECE_RATE)]
    offset = longest_pause(window, audio_settings)
    return None if offset is None else start + offset


def _write_piece(piece: Path, samples: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(piece), samples, PIECE_RATE, format="FLAC", subtype="PCM_16")


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
    stt: STTProvider, piece: Path, language: str, seconds: float | None, pacer: PiecePacer
) -> TranscriptionResult:
    """One piece's answer; one with text and doubts is asked again and the surer answer kept.

    A second answer is kept only when it has text and fewer doubts; one that fails keeps the first.
    """
    first = await _send(stt, piece, language, seconds, pacer, second_opinion=False)
    if not (first.text and first.doubts):
        return first
    log.warning("%s: a dictation piece raised %d doubts; again", stt.model_name, first.doubts)
    try:
        second = await _send(stt, piece, language, seconds, pacer, second_opinion=True)
    except RouteStoppedError:
        raise
    except Exception:
        log.warning("Asking again failed; keeping the first answer", exc_info=True)
        return first
    if second.text and (second.doubts or 0) < first.doubts:
        return second
    log.info("The second answer is no surer (%s doubts); keeping the first", second.doubts)
    return first


def _joined(results: list[TranscriptionResult], language: str) -> TranscriptionResult:
    """The pieces' texts joined with a space; ``no_speech_prob`` is 1.0 when none has text."""
    spoken = [result for result in results if result.text]
    tokens = [result.tokens_used for result in results if result.tokens_used is not None]
    heard = Counter(result.detected_language for result in spoken if result.detected_language)
    return TranscriptionResult(
        text=" ".join(result.text for result in spoken),
        tokens_used=sum(tokens) if tokens else None,
        detected_language=reported_language(language, heard),
        no_speech_prob=None if spoken else 1.0,
    )


async def transcribe_dictation(
    stt: STTProvider,
    audio_path: Path,
    language: str,
    duration: float | None,
    pacer: PiecePacer,
) -> TranscriptionResult:
    """Transcribe one dictation in pieces; scratch files sit beside ``audio_path`` and are removed.

    The route is checked before every send, and a provider's wait is paced as for a short file.
    """
    mono = audio_path.with_name(f"{audio_path.stem}-16k.wav")
    try:
        await asyncio.to_thread(decode_to_mono_wav, audio_path, mono, PIECE_RATE)
    except UndecodableAudioError:
        log.info("%s cannot be cut here; sending the dictation whole", audio_path.suffix)
        whole = await _transcribe_piece(stt, audio_path, language, duration, pacer)
        return _joined([whole], language)

    results: list[TranscriptionResult] = []
    try:
        speech = await asyncio.to_thread(_speech_between_edges, mono)
        pieces = await asyncio.to_thread(
            plan_dictation_pieces,
            speech.size / PIECE_RATE,
            lambda start, end: _pause_in(speech, start, end),
        )
        log.info(
            "Dictation: %.1fs of speech in %d pieces", speech.size / PIECE_RATE, len(pieces)
        )
        for index, (start, end) in enumerate(pieces):
            piece = audio_path.with_name(f"{audio_path.stem}-piece{index}.flac")
            try:
                samples = speech[round(start * PIECE_RATE) : round(end * PIECE_RATE)]
                await asyncio.to_thread(_write_piece, piece, samples)
                results.append(await _transcribe_piece(stt, piece, language, end - start, pacer))
            finally:
                discard_scratch_file(piece)
    finally:
        discard_scratch_file(mono)
    return _joined(results, language)

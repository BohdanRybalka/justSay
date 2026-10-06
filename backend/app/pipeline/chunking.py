"""Long audio as overlapping pieces every provider accepts, joined back into one transcript.

Audio soundfile decodes becomes 16 kHz mono FLAC pieces (Groq's recommended input) overlapping by
``OVERLAP_SECONDS``; the words both pieces heard are kept once. A format it cannot decode goes as
one request. A refusal carrying ``Retry-After`` pauses the work and retries the same piece.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from app.core.audio_formats import UndecodableAudioError, decode_to_mono_wav
from app.core.errors import ResourceUnavailableError
from app.core.scratch import discard_scratch_file
from app.stt import routing
from app.stt.base import STTProvider, TranscriptionResult

log = logging.getLogger(__name__)

PIECE_RATE = 16000
CLOUD_PIECE_SECONDS = 600.0
LOCAL_PIECE_SECONDS = 120.0
OVERLAP_SECONDS = 10.0
MAX_PAUSE_SECONDS = 300.0
MAX_PAUSES_PER_PIECE = 5
MAX_OVERLAP_WORDS = 45
MIN_SEAM_RUN = 3

_TOKEN = re.compile(r"\S+\s*")


class PiecePacer(Protocol):
    """What the caller hears around each piece; it may hold a piece back or stop the work."""

    async def before_piece(self, index: int, total: int) -> None: ...

    async def pause(self, seconds: float) -> None: ...

    def piece_done(self, done: int, total: int) -> None: ...


def piece_spans(duration: float, piece_seconds: float) -> list[tuple[float, float]]:
    """Equal ``(start, end)`` spans covering ``duration``, none longer than ``piece_seconds``."""
    if duration <= piece_seconds:
        return [(0.0, duration)]
    count = math.ceil((duration - OVERLAP_SECONDS) / (piece_seconds - OVERLAP_SECONDS))
    step = (duration - OVERLAP_SECONDS) / count
    return [(i * step, min(duration, i * step + step + OVERLAP_SECONDS)) for i in range(count)]


def _normalised(token: str) -> str:
    return "".join(ch for ch in token.casefold() if ch.isalnum())


def _seam_words(tokens: list[str]) -> list[object]:
    return [_normalised(token) or object() for token in tokens]


def _overlap_run(tail: list[object], head: list[object]) -> tuple[int, int, int] | None:
    """``(a, b, size)``: the longest shared run of ``tail`` and ``head`` that fits the overlap.

    A run fits when it, the words after it in ``tail`` and those before it in ``head`` add up to
    at most ``MAX_OVERLAP_WORDS``; among equally long runs the one nearest the seam wins.
    """
    best: tuple[tuple[int, int], int, int, int] | None = None
    previous = [0] * (len(head) + 1)
    for i in range(1, len(tail) + 1):
        current = [0] * (len(head) + 1)
        for j in range(1, len(head) + 1):
            if tail[i - 1] != head[j - 1]:
                continue
            size = current[j] = previous[j - 1] + 1
            span = (len(tail) - i) + size + (j - size)
            if size >= MIN_SEAM_RUN and span <= MAX_OVERLAP_WORDS:
                rank = (size, -span)
                if best is None or rank > best[0]:
                    best = (rank, i - size, j - size, size)
        previous = current
    return None if best is None else best[1:]


def join_at_seam(left: str, right: str) -> str:
    """``left`` then ``right``, with the words both heard in the overlap kept once.

    The overlap is marked by a run of at least three equal words that can sit inside it (see
    ``_overlap_run``); each side is cut at its middle, so a word one piece added beside it does
    not hide it. Without such a run, as when the overlap held no speech, nothing is cut.
    """
    left_tokens, right_tokens = _TOKEN.findall(left), _TOKEN.findall(right)
    tail_start = max(0, len(left_tokens) - MAX_OVERLAP_WORDS)
    run = _overlap_run(
        _seam_words(left_tokens[tail_start:]), _seam_words(right_tokens[:MAX_OVERLAP_WORDS])
    )
    if run is None:
        return f"{left.rstrip()} {right.lstrip()}".strip()
    a, b, size = run
    half = size // 2
    return "".join(left_tokens[: tail_start + a + half] + right_tokens[b + half :]).strip()


def _write_piece(mono: Path, piece: Path, start: float, end: float) -> None:
    import soundfile as sf

    with sf.SoundFile(str(mono)) as source:
        first = round(start * PIECE_RATE)
        source.seek(first)
        samples = source.read(round(end * PIECE_RATE) - first, dtype="int16")
    sf.write(str(piece), samples, PIECE_RATE, format="FLAC", subtype="PCM_16")


def _frames(mono: Path) -> int:
    import soundfile as sf

    return sf.info(str(mono)).frames


def _retry_after(refusal: ResourceUnavailableError) -> float | None:
    value = (refusal.headers or {}).get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


async def _transcribe_with_pauses(
    stt: STTProvider, path: Path, language: str, duration: float | None, pacer: PiecePacer
) -> TranscriptionResult:
    pauses = 0
    while True:
        try:
            return await stt.transcribe(path, language=language, audio_duration=duration)
        except ResourceUnavailableError as refusal:
            wait = _retry_after(refusal)
            pauses += 1
            if wait is None or wait > MAX_PAUSE_SECONDS or pauses > MAX_PAUSES_PER_PIECE:
                raise
            log.info("%s asked to wait %.0fs; pausing (pause %d)", stt.model_name, wait, pauses)
            await pacer.pause(wait)


async def transcribe_in_pieces(
    stt: STTProvider,
    audio_path: Path,
    *,
    language: str,
    duration: float | None,
    no_speech_threshold: float,
    holds_no_speech: Callable[[Path], bool],
    pacer: PiecePacer,
) -> TranscriptionResult:
    """Transcribe ``audio_path`` piece by piece; scratch files sit beside it and are removed.

    A piece ``holds_no_speech`` judges silent is never sent; one whose ``no_speech_prob`` exceeds
    ``no_speech_threshold`` adds no text. The first piece with speech fixes ``"auto"`` for the
    rest. With no speech in any piece, ``no_speech_prob`` is 1.0, so the caller discards it.
    """
    mono = audio_path.with_name(f"{audio_path.stem}-16k.wav")
    try:
        await asyncio.to_thread(decode_to_mono_wav, audio_path, mono, PIECE_RATE)
    except UndecodableAudioError:
        log.info("%s cannot be cut here; sending it whole", audio_path.suffix)
        await pacer.before_piece(0, 1)
        result = await _transcribe_with_pauses(stt, audio_path, language, duration, pacer)
        pacer.piece_done(1, 1)
        return result

    try:
        local = routing.is_local_provider(stt)
        piece_seconds = LOCAL_PIECE_SECONDS if local else CLOUD_PIECE_SECONDS
        spans = piece_spans(await asyncio.to_thread(_frames, mono) / PIECE_RATE, piece_seconds)
        texts: list[str] = []
        silent = 0
        tokens: list[int] = []
        for index, (start, end) in enumerate(spans):
            await pacer.before_piece(index, len(spans))
            piece = audio_path.with_name(f"{audio_path.stem}-piece{index}.flac")
            try:
                await asyncio.to_thread(_write_piece, mono, piece, start, end)
                if await asyncio.to_thread(holds_no_speech, piece):
                    result = None
                else:
                    result = await _transcribe_with_pauses(
                        stt, piece, language, end - start, pacer
                    )
            finally:
                discard_scratch_file(piece)
            pacer.piece_done(index + 1, len(spans))
            if result is None:
                silent += 1
                continue
            if result.tokens_used is not None:
                tokens.append(result.tokens_used)
            prob = result.no_speech_prob
            if prob is not None and prob > no_speech_threshold:
                silent += 1
                continue
            if language == "auto" and result.detected_language:
                language = result.detected_language
            if result.text:
                texts.append(result.text)
    finally:
        discard_scratch_file(mono)

    text = texts[0] if texts else ""
    for following in texts[1:]:
        text = join_at_seam(text, following)
    return TranscriptionResult(
        text=text,
        tokens_used=sum(tokens) if tokens else None,
        detected_language=None if language == "auto" else language,
        no_speech_prob=1.0 if silent == len(spans) else None,
    )

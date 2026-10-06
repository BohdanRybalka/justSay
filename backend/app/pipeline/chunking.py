"""Long audio as pieces every provider accepts, cut in pauses and joined back into one transcript.

Decodable audio becomes 16 kHz mono FLAC pieces no longer than the provider's
``longest_piece_seconds``, each ending in a pause; where none is found, pieces overlap and the
words both heard are kept once. A transcript stuck in a loop or far too sparse is asked for again.
``Retry-After`` pauses the work; audio nothing here decodes goes whole.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from app.audio.config import audio_settings
from app.audio.pauses import longest_pause
from app.core.audio_formats import UndecodableAudioError, decode_to_mono_wav
from app.core.errors import ResourceUnavailableError
from app.core.scratch import discard_scratch_file
from app.stt.base import STTProvider, TranscriptionResult

log = logging.getLogger(__name__)

PIECE_RATE = 16000
PAUSE_SEARCH_SHARE = 0.1
MIN_PAUSE_SEARCH_SECONDS = 10.0
OVERLAP_SECONDS = 10.0
OVERLAP_SHARE = 1 / 6
MAX_PAUSE_SECONDS = 300.0
MAX_PAUSES_PER_PIECE = 5
MAX_OVERLAP_WORDS = 45
MIN_SEAM_RUN = 3
LOOP_NGRAM_CHARS = 32
VARIETY_NGRAM_CHARS = 4
LOOP_REPEATED_SHARE = 0.3
MIN_CHARS_PER_MINUTE = 60
SPARSE_CHECK_FROM_SECONDS = 120.0

_TOKEN = re.compile(r"\S+\s*")


class PiecePacer(Protocol):
    """What the caller hears around each piece; it may hold a piece back or stop the work."""

    async def before_piece(self, index: int, total: int) -> None: ...

    async def pause(self, seconds: float) -> None: ...

    def piece_done(self, done: int, total: int) -> None: ...


@dataclass(frozen=True)
class Piece:
    start: float
    end: float
    overlaps_previous: bool


def plan_pieces(
    duration: float, piece_seconds: float, find_pause: Callable[[float, float], float | None]
) -> list[Piece]:
    """Pieces covering ``duration``, none longer than ``piece_seconds``, each ending in a pause.

    ``find_pause(start, end)`` answers the time of the longest pause in the last tenth of a piece
    (10 s at least), never leaving less than that for the last one. Without a pause the piece ends
    there and the next starts up to ``OVERLAP_SECONDS`` earlier.
    """
    if piece_seconds < 2 * MIN_PAUSE_SEARCH_SECONDS:
        raise ValueError(f"A piece of {piece_seconds}s leaves no room to search for a pause")
    pieces: list[Piece] = []
    search = max(piece_seconds * PAUSE_SEARCH_SHARE, MIN_PAUSE_SEARCH_SECONDS)
    overlap = min(OVERLAP_SECONDS, piece_seconds * OVERLAP_SHARE)
    start, overlaps = 0.0, False
    while duration - start > piece_seconds:
        limit = min(start + piece_seconds, duration - search)
        pause = find_pause(limit - search, limit)
        if pause is None:
            pieces.append(Piece(start, limit, overlaps))
            start, overlaps = limit - overlap, True
        else:
            pieces.append(Piece(start, pause, overlaps))
            start, overlaps = pause, False
    pieces.append(Piece(start, duration, overlaps))
    return pieces


def _normalised(token: str) -> str:
    return "".join(ch for ch in token.casefold() if ch.isalnum())


def _distinct_stretches(letters: str, size: int = LOOP_NGRAM_CHARS) -> int:
    return len({letters[i : i + size] for i in range(len(letters) - size + 1)})


def looks_broken(text: str, seconds: float) -> bool:
    """Too little text for ``seconds`` of audio with speech in it, or the same text over and over.

    Counted in letters, so languages written without spaces are measured alike. An empty answer
    always is; below ``SPARSE_CHECK_FROM_SECONDS`` a piece may hold one word and is not sparse.
    """
    letters = _normalised(text)
    if not letters:
        return True
    sparse_possible = seconds >= SPARSE_CHECK_FROM_SECONDS
    if sparse_possible and len(letters) < MIN_CHARS_PER_MINUTE * seconds / 60:
        return True
    stretches = len(letters) - LOOP_NGRAM_CHARS + 1
    return stretches > 0 and 1 - _distinct_stretches(letters) / stretches > LOOP_REPEATED_SHARE


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
    not hide it. Without such a run nothing is cut; a phrase repeated close to the seam on both
    sides of a speechless overlap can still be taken for it.
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


def _pause_in(mono: Path, start: float, end: float) -> float | None:
    import soundfile as sf

    first, last = round(start * PIECE_RATE), round(end * PIECE_RATE)
    samples, _rate = sf.read(str(mono), start=first, stop=last, dtype="float32")
    offset = longest_pause(samples, audio_settings)
    return None if offset is None else start + offset


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


async def _transcribe_checked(
    stt: STTProvider,
    piece: Path,
    language: str,
    seconds: float,
    no_speech_threshold: float,
    pacer: PiecePacer,
    turn: Callable[[], Awaitable[None]],
) -> TranscriptionResult:
    first = await _transcribe_with_pauses(stt, piece, language, seconds, pacer)
    prob = first.no_speech_prob
    if (prob is not None and prob > no_speech_threshold) or not looks_broken(first.text, seconds):
        return first
    log.warning("%s: a %.0fs piece looked looping or sparse; again", stt.model_name, seconds)
    await turn()
    try:
        second = await _transcribe_with_pauses(stt, piece, language, seconds, pacer)
    except Exception:
        log.warning("Asking again failed; keeping the first answer", exc_info=True)
        return first
    kept = max(
        (first, second),
        key=lambda r: (
            not looks_broken(r.text, seconds),
            _distinct_stretches(_normalised(r.text), VARIETY_NGRAM_CHARS),
            len(_normalised(r.text)),
        ),
    )
    spent = [r.tokens_used for r in (first, second) if r.tokens_used is not None]
    return replace(kept, tokens_used=sum(spent) if spent else None)


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
        total = await asyncio.to_thread(_frames, mono) / PIECE_RATE
        pieces = await asyncio.to_thread(
            plan_pieces,
            total,
            stt.longest_piece_seconds,
            lambda start, end: _pause_in(mono, start, end),
        )
        texts: list[tuple[str, bool]] = []
        silent = 0
        tokens: list[int] = []
        for index, planned in enumerate(pieces):

            async def turn(index: int = index) -> None:
                await pacer.before_piece(index, len(pieces))

            await turn()
            piece = audio_path.with_name(f"{audio_path.stem}-piece{index}.flac")
            seconds = planned.end - planned.start
            try:
                await asyncio.to_thread(_write_piece, mono, piece, planned.start, planned.end)
                if await asyncio.to_thread(holds_no_speech, piece):
                    result = None
                else:
                    result = await _transcribe_checked(
                        stt, piece, language, seconds, no_speech_threshold, pacer, turn
                    )
            finally:
                discard_scratch_file(piece)
            pacer.piece_done(index + 1, len(pieces))
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
                texts.append((result.text, planned.overlaps_previous))
    finally:
        discard_scratch_file(mono)

    text = texts[0][0] if texts else ""
    for following, overlaps in texts[1:]:
        text = join_at_seam(text, following) if overlaps else f"{text} {following}"
    return TranscriptionResult(
        text=text,
        tokens_used=sum(tokens) if tokens else None,
        detected_language=None if language == "auto" else language,
        no_speech_prob=1.0 if silent == len(pieces) else None,
    )

"""A two-channel meeting as turns, yours and the others', in the order they were spoken.

Each channel is cut into turns at its own pauses and every turn goes to the engine on its own.
A microphone turn that only repeats what the others said in the same seconds is cut, which is
what a microphone hears when the far side plays through speakers. The turns are merged by start
time; with speech on one side only, the text carries no labels.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path

from app.audio.config import audio_settings
from app.audio.pauses import MIN_SILENCE_SECONDS, SPEECH_PAD_SECONDS, speech_profile
from app.audio.timeline import MICROPHONE_TRACK, SYSTEM_TRACK, TRACK_COUNT
from app.core.scratch import discard_scratch_file
from app.pipeline.chunking import (
    PIECE_RATE,
    PiecePacer,
    comparable_words,
    reported_language,
    transcribe_stretch,
)
from app.stt.base import STTProvider, TranscriptionResult

LABELS = {MICROPHONE_TRACK: "You", SYSTEM_TRACK: "Others"}
TRACK_SUFFIXES = {MICROPHONE_TRACK: "you", SYSTEM_TRACK: "others"}
WINDOW_SECONDS = 60.0
FLOAT_SLACK = 1e-6
TURN_JOIN_GAP_SECONDS = MIN_SILENCE_SECONDS - 2 * SPEECH_PAD_SECONDS - FLOAT_SLACK
ECHO_WINDOW_SECONDS = 2.0
ECHO_COPY_MIN_RUN = 2
ECHO_COPY_SHARE = 0.75
ECHO_CUT_MIN_RUN = 4

_LABEL = re.compile("^(?:" + "|".join(LABELS.values()) + "): ", re.MULTILINE)


@dataclass(frozen=True)
class Spoken:
    start: float
    end: float
    track: int
    text: str


class _TurnPacer:
    """Shows the pieces of every turn to a pacer as the pieces of one file."""

    def __init__(self, pacer: PiecePacer, turns: int) -> None:
        self._pacer = pacer
        self._finished_pieces = 0
        self._pieces = 1
        self._turns_left = turns

    def _total(self) -> int:
        return self._finished_pieces + self._pieces + max(self._turns_left - 1, 0)

    async def before_piece(self, index: int, total: int) -> None:
        self._pieces = total
        await self._pacer.before_piece(self._finished_pieces + index, self._total())

    async def pause(self, seconds: float) -> None:
        await self._pacer.pause(seconds)

    def piece_done(self, done: int, total: int) -> None:
        self._pieces = total
        self._pacer.piece_done(self._finished_pieces + done, self._total())

    def finish_turn(self) -> None:
        self._finished_pieces += self._pieces
        self._pieces = 1
        self._turns_left -= 1


def has_two_tracks(audio_path: Path) -> bool:
    """Whether ``audio_path`` is a meeting file with a channel for each side, at the piece rate."""
    import soundfile as sf

    try:
        info = sf.info(str(audio_path))
    except RuntimeError:
        return False
    return info.channels == TRACK_COUNT and info.samplerate == PIECE_RATE


def spoken_word_count(text: str) -> int:
    """The words in a meeting's ``text``, without its labels."""
    return len(_LABEL.sub("", text).split())


def _split_tracks(stereo: Path, targets: dict[int, Path]) -> None:
    import soundfile as sf

    with ExitStack() as sinks:
        writers = {
            track: sinks.enter_context(
                sf.SoundFile(
                    str(path),
                    "w",
                    samplerate=PIECE_RATE,
                    channels=1,
                    format="WAV",
                    subtype="PCM_16",
                )
            )
            for track, path in targets.items()
        }
        for block in sf.blocks(
            str(stereo), blocksize=round(WINDOW_SECONDS * PIECE_RATE), dtype="int16", always_2d=True
        ):
            for track, writer in writers.items():
                writer.write(block[:, track])


def _turns(mono: Path) -> list[tuple[float, float]]:
    """The ``(start, end)`` seconds of each stretch of speech in ``mono``, split by 2 s pauses."""
    import soundfile as sf

    turns: list[tuple[float, float]] = []
    offset = 0.0
    with sf.SoundFile(str(mono)) as source:
        while True:
            window = source.read(round(WINDOW_SECONDS * PIECE_RATE), dtype="float32")
            if not window.size:
                return turns
            for start, end in speech_profile(window, audio_settings).spans:
                if turns and offset + start - turns[-1][1] < TURN_JOIN_GAP_SECONDS:
                    turns[-1] = (turns[-1][0], offset + end)
                else:
                    turns.append((offset + start, offset + end))
            offset += window.size / PIECE_RATE


def without_repeats(text: str, heard: str) -> str:
    """``text`` without the runs of words ``heard`` has too; nothing if it is mostly such runs."""
    words = text.split()
    blocks = SequenceMatcher(
        None, comparable_words(words), comparable_words(heard.split()), autojunk=False
    ).get_matching_blocks()
    if sum(b.size for b in blocks if b.size >= ECHO_COPY_MIN_RUN) >= ECHO_COPY_SHARE * len(words):
        return ""
    cut = {
        index
        for b in blocks
        if b.size >= ECHO_CUT_MIN_RUN
        for index in range(b.a, b.a + b.size)
    }
    return " ".join(word for index, word in enumerate(words) if index not in cut)


def _without_echo(yours: list[Spoken], others: list[Spoken]) -> list[Spoken]:
    kept = []
    for turn in yours:
        nearby = " ".join(
            other.text
            for other in others
            if other.start - ECHO_WINDOW_SECONDS < turn.end
            and turn.start - ECHO_WINDOW_SECONDS < other.end
        )
        text = without_repeats(turn.text, nearby) if nearby else turn.text
        if text:
            kept.append(replace(turn, text=text))
    return kept


def _as_text(turns: list[Spoken]) -> str:
    """``turns`` by start time; labelled lines when both sides spoke, else the words alone."""
    ordered = sorted(turns, key=lambda turn: (turn.start, turn.track))
    if len({turn.track for turn in ordered}) < TRACK_COUNT:
        return " ".join(turn.text for turn in ordered)
    lines: list[str] = []
    previous: int | None = None
    for turn in ordered:
        if turn.track == previous:
            lines[-1] += f" {turn.text}"
        else:
            lines.append(f"{LABELS[turn.track]}: {turn.text}")
        previous = turn.track
    return "\n".join(lines)


async def transcribe_meeting(
    stt: STTProvider,
    audio_path: Path,
    *,
    language: str,
    no_speech_threshold: float,
    holds_no_speech: Callable[[Path], bool],
    pacer: PiecePacer,
) -> TranscriptionResult:
    """Transcribe the two channels of ``audio_path`` turn by turn and merge them by time.

    Scratch files sit beside ``audio_path`` and are removed. With no speech in either
    channel, ``no_speech_prob`` is 1.0, so the caller discards it.
    """
    tracks = {
        track: audio_path.with_name(f"{audio_path.stem}-{suffix}.wav")
        for track, suffix in TRACK_SUFFIXES.items()
    }
    spoken: dict[int, list[Spoken]] = {track: [] for track in tracks}
    heard: Counter[str] = Counter()
    tokens: list[int] = []
    try:
        await asyncio.to_thread(_split_tracks, audio_path, tracks)
        plan = [
            (track, start, end)
            for track, mono in tracks.items()
            for start, end in await asyncio.to_thread(_turns, mono)
        ]
        turn_pacer = _TurnPacer(pacer, len(plan))
        for track, start, end in plan:
            result = await transcribe_stretch(
                stt,
                tracks[track],
                tracks[track],
                start,
                end,
                language=language,
                no_speech_threshold=no_speech_threshold,
                holds_no_speech=holds_no_speech,
                pacer=turn_pacer,
            )
            turn_pacer.finish_turn()
            if result.tokens_used is not None:
                tokens.append(result.tokens_used)
            if result.detected_language:
                heard[result.detected_language] += 1
            if result.text:
                spoken[track].append(Spoken(start, end, track, result.text))
    finally:
        for mono in tracks.values():
            discard_scratch_file(mono)

    turns = spoken[SYSTEM_TRACK] + _without_echo(spoken[MICROPHONE_TRACK], spoken[SYSTEM_TRACK])
    return TranscriptionResult(
        text=_as_text(turns),
        tokens_used=sum(tokens) if tokens else None,
        detected_language=reported_language(language, heard),
        no_speech_prob=None if turns else 1.0,
    )

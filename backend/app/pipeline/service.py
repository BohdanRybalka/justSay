"""Audio processing pipeline: Audio -> STT -> cleanup -> History.

Cloud mode transcribes on Groq Whisper, Local mode on this machine's engine.
A Cloud-mode dictation is cleaned of fillers before it is returned; history keeps both texts.
A dictation shorter than a word, or with less voice than one by the VAD, is silence; a short one
loses the phrases engines make up over silence.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import BackgroundTasks

from app.audio.analysis import analyze_silence
from app.audio.config import audio_settings
from app.audio.vad import HOP_SECONDS, analyze_vad
from app.core.types import ProviderMode
from app.pipeline.chunking import (
    LITTLE_SPEECH_SECONDS,
    SHORT_MAX_PAUSE_SECONDS,
    SHORT_MAX_PAUSES,
    FinishedPieces,
    PiecePacer,
    RouteStoppedError,
    UnpacedPieces,
    transcribe_in_pieces,
    transcribe_with_pauses,
)
from app.pipeline.cleanup import clean_dictation
from app.pipeline.dictation import transcribe_dictation
from app.pipeline.invented_phrases import SHORTEST_WORD_SECONDS, without_invented_phrases
from app.pipeline.meeting_text import has_two_tracks, spoken_word_count, transcribe_meeting
from app.pipeline.utils import detect_duration
from app.stt.base import SINGLE_ATTEMPT
from app.stt.config import stt_settings
from app.stt.routing import get_provider, is_local_provider
from app.transcripts.history import EntrySource, save_entry

log = logging.getLogger(__name__)

LONG_DICTATION_SECONDS = 600.0
DICTATION_SPEECH_HOPS = math.ceil(SHORTEST_WORD_SECONDS / HOP_SECONDS)


@dataclass
class ProcessingResult:
    """Domain model for pipeline output."""

    text: str
    duration_ms: int
    model_name: str = ""
    discarded_reason: str | None = None
    entry_id: str | None = None


class PipelineObserver(PiecePacer, Protocol):
    """What a caller showing progress hears: the route, each piece, then the save."""

    async def before_transcribe(self, model_name: str, audio_duration: float | None) -> None: ...

    def before_save(self) -> None: ...

    def saved(self, entry_id: str) -> None: ...


def _silence_note(audio_path: Path, min_speech_hops: int = 0) -> tuple[str, tuple] | None:
    """Why ``audio_path`` holds no speech, as a log format and its arguments; ``None`` if it does.

    The VAD decides when it can run, needing at least ``min_speech_hops`` of speech; the energy
    check decides only when the VAD cannot.
    """
    vad = (
        analyze_vad(audio_path, audio_settings, min_speech_hops=min_speech_hops)
        if audio_settings.silence_vad_enabled
        else None
    )
    if vad is not None:
        if not vad.is_silent:
            return None
        return (
            "Discarding no-speech audio (layer=vad): speech_hops=%d/%d, max_prob=%.3f",
            (vad.speech_hop_count, vad.total_hop_count, vad.max_probability),
        )
    analysis = analyze_silence(audio_path, audio_settings)
    if analysis is None or not analysis.is_silent:
        return None
    return (
        "Discarding silent audio (layer=energy): peak=%.1f dBFS, speech_frames=%d/%d",
        (analysis.peak_dbfs, analysis.speech_frame_count, analysis.total_frame_count),
    )


def _cloud_stays_on() -> bool:
    return stt_settings.mode == ProviderMode.CLOUD


def _holds_no_speech(audio_path: Path) -> bool:
    note = _silence_note(audio_path)
    if note is not None:
        log.info(note[0], *note[1])
    return note is not None


def _shortest(*seconds: float | None) -> float | None:
    known = [value for value in seconds if value is not None]
    return min(known) if known else None


def _discarded(start: float, model_name: str = "") -> ProcessingResult:
    return ProcessingResult(
        text="",
        duration_ms=int((time.perf_counter() - start) * 1000),
        model_name=model_name,
        discarded_reason="silence",
    )


async def process_audio(
    audio_path: Path,
    language: str = "uk",
    audio_duration: float | None = None,
    background_tasks: BackgroundTasks | None = None,
    *,
    source: EntrySource,
    source_name: str | None = None,
    observer: PipelineObserver | None = None,
    mode: ProviderMode | None = None,
    finished: FinishedPieces | None = None,
) -> ProcessingResult:
    """Full pipeline on ``mode``'s provider, else the user's mode: transcribe, clean up, save.

    ``background_tasks`` schedules embedding generation after the response is sent. ``source``
    and ``source_name`` say where the history entry came from. Audio with an ``observer``, or
    longer than ``LONG_DICTATION_SECONDS``, goes in pieces; those ``finished`` already holds are
    not sent again (not for a two-channel meeting). A shorter Cloud dictation has its own pieces.
    """
    start = time.perf_counter()

    duration = audio_duration
    if duration is None:
        duration = detect_duration(audio_path)

    two_tracks = (
        source == "meeting"
        and observer is not None
        and await asyncio.to_thread(has_two_tracks, audio_path)
    )
    dictation = source == "dictation"
    recorded = (
        _shortest(duration, await asyncio.to_thread(detect_duration, audio_path))
        if dictation
        else None
    )
    if recorded is not None and recorded < SHORTEST_WORD_SECONDS:
        log.warning("Discarding a %.3fs dictation: too short to hold a word", recorded)
        return _discarded(start)
    discard_log = (
        None
        if two_tracks
        else await asyncio.to_thread(
            _silence_note, audio_path, DICTATION_SPEECH_HOPS if dictation else 0
        )
    )
    if discard_log is not None:
        log.warning(discard_log[0], *discard_log[1])
        return _discarded(start)

    route_settings = stt_settings
    if mode is not None:
        route_settings = stt_settings.model_copy(update={"mode": mode})
    stt = get_provider(route_settings.mode, stt_settings)

    log.info(
        "Pipeline route: %s, duration=%.2fs",
        stt.model_name,
        duration if duration is not None else -1.0,
    )

    if is_local_provider(stt):
        from app.stt.local_setup import await_local_ready

        await await_local_ready(route_settings)

    if observer is not None:
        await observer.before_transcribe(stt.model_name, duration)

    cloud_dictation = dictation and route_settings.mode == ProviderMode.CLOUD
    stays_on = _cloud_stays_on if cloud_dictation else None
    pacer: PiecePacer = observer if observer is not None else UnpacedPieces(stays_on)
    attempt_token = SINGLE_ATTEMPT.set(route_settings.mode == ProviderMode.CLOUD)
    short = observer is None and not (duration is not None and duration > LONG_DICTATION_SECONDS)
    try:
        if short and cloud_dictation:
            result = await transcribe_dictation(
                stt,
                audio_path,
                language,
                duration,
                pacer,
                no_speech_threshold=stt_settings.no_speech_prob_threshold,
            )
        elif short:
            result = await transcribe_with_pauses(
                stt,
                audio_path,
                language,
                duration,
                pacer,
                max_pause=SHORT_MAX_PAUSE_SECONDS,
                max_pauses=SHORT_MAX_PAUSES,
            )
        elif two_tracks:
            result = await transcribe_meeting(
                stt,
                audio_path,
                language=language,
                no_speech_threshold=stt_settings.no_speech_prob_threshold,
                holds_no_speech=_holds_no_speech,
                pacer=pacer,
            )
        else:
            result = await transcribe_in_pieces(
                stt,
                audio_path,
                language=language,
                duration=duration,
                no_speech_threshold=stt_settings.no_speech_prob_threshold,
                holds_no_speech=_holds_no_speech,
                pacer=pacer,
                finished=finished,
            )
    except RouteStoppedError:
        log.info("Stopped %s: the route was switched off mid-run", stt.model_name)
        raise
    except Exception:
        log.exception("STT transcribe failed (%s)", stt.model_name)
        raise
    finally:
        SINGLE_ATTEMPT.reset(attempt_token)

    if (
        result.no_speech_prob is not None
        and result.no_speech_prob > stt_settings.no_speech_prob_threshold
    ):
        log.warning(
            "Discarding transcription (layer=provider-metadata): no_speech_prob=%.3f > %.3f "
            "(%s, %d chars discarded)",
            result.no_speech_prob, stt_settings.no_speech_prob_threshold,
            stt.model_name,
            len(result.text),
        )
        return _discarded(start, stt.model_name)

    raw_text = result.text
    if recorded is not None and recorded < LITTLE_SPEECH_SECONDS:
        raw_text = without_invented_phrases(result.text)
        if result.text and not raw_text:
            log.warning(
                "Discarding a %.1fs dictation: %s wrote only a phrase made up over silence",
                recorded, stt.model_name,
            )
            return _discarded(start, stt.model_name)
        if raw_text != result.text:
            log.info(
                "Dropped made-up phrases from a %.1fs dictation (%d of %d chars kept)",
                recorded, len(raw_text), len(result.text),
            )
    text = raw_text
    if dictation and not is_local_provider(stt):
        text = await clean_dictation(raw_text, stt_settings)

    log.info(
        "Pipeline result: %s produced %d chars in %dms",
        stt.model_name,
        len(text),
        int((time.perf_counter() - start) * 1000),
    )

    duration_ms = int((time.perf_counter() - start) * 1000)
    word_count = spoken_word_count(text) if source == "meeting" else len(text.split())

    effective_language = language
    if language == "auto" and result.detected_language:
        effective_language = result.detected_language

    if observer is not None:
        observer.before_save()

    entry_id: str | None = None
    try:
        entry = save_entry(
            text=text,
            raw_text=raw_text,
            duration_ms=duration_ms,
            language=effective_language,
            model_name=stt.model_name,
            tokens_used=result.tokens_used,
            audio_duration_seconds=duration,
            word_count=word_count,
            source=source,
            source_name=source_name,
        )
        entry_id = entry.id
        if observer is not None:
            observer.saved(entry.id)
        if background_tasks is not None and text:
            from app.transcripts import vector_store

            background_tasks.add_task(vector_store.embed_entry_background, entry.id, text)
            background_tasks.add_task(vector_store.run_background_indexer)
    except Exception:
        log.exception("Saving the history entry failed — this transcript is not in history")

    return ProcessingResult(
        text=text,
        duration_ms=duration_ms,
        model_name=stt.model_name,
        entry_id=entry_id,
    )

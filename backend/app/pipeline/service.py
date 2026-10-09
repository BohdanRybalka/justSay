"""Audio processing pipeline: Audio -> STT -> cleanup -> Clipboard.

Cloud mode transcribes on Groq Whisper, Local mode on this machine's engine.
A Cloud-mode dictation is cleaned of fillers before the clipboard; history keeps both texts.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyperclip
from fastapi import BackgroundTasks

from app.audio.analysis import analyze_silence
from app.audio.config import audio_settings
from app.audio.vad import analyze_vad
from app.core.types import ProviderMode
from app.pipeline.chunking import PiecePacer, transcribe_in_pieces
from app.pipeline.cleanup import clean_dictation
from app.pipeline.utils import detect_duration
from app.stt.config import stt_settings
from app.stt.routing import get_provider, is_local_provider
from app.transcripts.history import EntrySource, save_entry

log = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Domain model for pipeline output."""

    text: str
    duration_ms: int
    copied_to_clipboard: bool
    model_name: str = ""
    discarded_reason: str | None = None


class PipelineObserver(PiecePacer, Protocol):
    """What a caller showing progress hears: the route, each piece, then the save."""

    async def before_transcribe(self, model_name: str, audio_duration: float | None) -> None: ...

    def before_save(self) -> None: ...

    def saved(self, entry_id: str) -> None: ...


def _silence_note(audio_path: Path) -> tuple[str, tuple] | None:
    """Why ``audio_path`` holds no speech, as a log format and its arguments; ``None`` if it does.

    The VAD decides when it can run; the energy check only when it cannot.
    """
    vad = analyze_vad(audio_path, audio_settings) if audio_settings.silence_vad_enabled else None
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


def _holds_no_speech(audio_path: Path) -> bool:
    note = _silence_note(audio_path)
    if note is not None:
        log.info(note[0], *note[1])
    return note is not None


async def process_audio(
    audio_path: Path,
    language: str = "uk",
    copy_to_clipboard: bool = True,
    audio_duration: float | None = None,
    background_tasks: BackgroundTasks | None = None,
    *,
    source: EntrySource,
    source_name: str | None = None,
    observer: PipelineObserver | None = None,
    mode: ProviderMode | None = None,
) -> ProcessingResult:
    """Full pipeline: transcribe -> clipboard, on ``mode``'s provider, else the user's mode.

    ``background_tasks`` schedules embedding generation after the response is sent. ``source``
    and ``source_name`` say where the history entry came from; with an ``observer`` the audio
    goes in pieces it paces.
    """
    start = time.perf_counter()

    duration = audio_duration
    if duration is None:
        duration = detect_duration(audio_path)

    discard_log = await asyncio.to_thread(_silence_note, audio_path)
    if discard_log is not None:
        log.warning(discard_log[0], *discard_log[1])
        return ProcessingResult(
            text="",
            duration_ms=int((time.perf_counter() - start) * 1000),
            copied_to_clipboard=False,
            discarded_reason="silence",
        )

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

    try:
        if observer is None:
            result = await stt.transcribe(audio_path, language=language, audio_duration=duration)
        else:
            result = await transcribe_in_pieces(
                stt,
                audio_path,
                language=language,
                duration=duration,
                no_speech_threshold=stt_settings.no_speech_prob_threshold,
                holds_no_speech=_holds_no_speech,
                pacer=observer,
            )
    except Exception:
        log.exception("STT transcribe failed (%s)", stt.model_name)
        raise

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
        return ProcessingResult(
            text="",
            duration_ms=int((time.perf_counter() - start) * 1000),
            copied_to_clipboard=False,
            model_name=stt.model_name,
            discarded_reason="silence",
        )

    raw_text = result.text
    text = raw_text
    if source == "dictation" and not is_local_provider(stt):
        text = await clean_dictation(raw_text, stt_settings)

    log.info(
        "Pipeline result: %s produced %d chars in %dms",
        stt.model_name,
        len(text),
        int((time.perf_counter() - start) * 1000),
    )

    copied = False
    if copy_to_clipboard and text:
        try:
            pyperclip.copy(text)
            copied = True
        except Exception:
            log.warning("Copying the transcript to the clipboard failed", exc_info=True)

    duration_ms = int((time.perf_counter() - start) * 1000)
    word_count = len(text.split()) if text else 0

    effective_language = language
    if language == "auto" and result.detected_language:
        effective_language = result.detected_language

    if observer is not None:
        observer.before_save()

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
        copied_to_clipboard=copied,
        model_name=stt.model_name,
    )

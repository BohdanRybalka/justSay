"""Audio processing pipeline: Audio -> smart-routed STT -> Clipboard.

Gemini's ai_prompt style handles transcription and structuring in a single call;
short normal-style audio goes through Groq Whisper for minimum latency.
``/llm/process`` remains as a standalone endpoint for explicit cleanup.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pyperclip
from fastapi import BackgroundTasks

from app.audio.analysis import analyze_silence
from app.core.config import settings
from app.core.history import save_entry
from app.pipeline.utils import detect_duration
from app.stt import get_routed_provider, is_local_provider

log = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Domain model for pipeline output."""

    text: str
    duration_ms: int
    copied_to_clipboard: bool
    model_name: str = ""               # which provider actually handled the call
    fallback_reason: str | None = None  # set when a pinned engine was overridden
    # Set when process_audio short-circuited without calling a provider at
    # all -- currently only "silence". Distinct from fallback_reason, which
    # means something else entirely (a pinned engine was overridden) --
    # see docs/adr/015-pipeline-level-silence-guard.md.
    discarded_reason: str | None = None


async def process_audio(
    audio_path: Path,
    language: str = "uk",
    style: str = "normal",
    copy_to_clipboard: bool = True,
    audio_duration: float | None = None,
    background_tasks: BackgroundTasks | None = None,
) -> ProcessingResult:
    """Full pipeline: route STT by duration+style -> transcribe -> clipboard.

    ``background_tasks``, when provided, schedules embedding generation via
    ``BackgroundTasks.add_task`` — FastAPI guarantees these run AFTER the
    response is sent, which is the actual mechanism that keeps embedding
    latency off the Instant Prompt budget. Never awaited synchronously here.
    """
    start = time.perf_counter()

    duration = audio_duration
    if duration is None:
        duration = detect_duration(audio_path)

    # Silence guard (spec 029 / docs/adr/015-pipeline-level-silence-guard.md):
    # sits between duration detection and provider routing so a silent clip
    # never reaches a provider at all -- saves STT latency locally and real
    # Cloud API spend. `analyze_silence` fails open (returns None) on any
    # decode failure -- e.g. .m4a/.webm uploads libsndfile can't open --
    # so an unreadable file always falls through to normal transcription,
    # never gets mistaken for silence. Run via asyncio.to_thread -- it's
    # synchronous CPU/IO work on EVERY call, not only silent ones (measured
    # 71ms/40s, 239ms/150s), and every comparable call in this codebase
    # (local.py's _transcribe, local_mlx.py's _run_mlx) is already wrapped
    # the same way (Stage 3 review YELLOW finding).
    analysis = await asyncio.to_thread(analyze_silence, audio_path, settings.audio)
    if analysis is not None and analysis.is_silent:
        log.warning(
            "Discarding silent audio: peak=%.1f dBFS, speech_frames=%d/%d",
            analysis.peak_dbfs, analysis.speech_frame_count, analysis.total_frame_count,
        )
        # No STT call, no clipboard write, no save_entry -- a discarded
        # accidental hotkey press must be a silent no-op, not a raised
        # exception (the widget's error toast is wired to thrown exceptions).
        return ProcessingResult(
            text="",
            duration_ms=int((time.perf_counter() - start) * 1000),
            copied_to_clipboard=False,
            discarded_reason="silence",
        )

    file_ext = audio_path.suffix.lower() if audio_path.suffix else None
    stt, fallback_reason = get_routed_provider(
        settings.stt,
        audio_duration=duration,
        style=style,
        file_extension=file_ext,
    )

    log.info(
        "Pipeline route: %s, duration=%.2fs, style=%s, ext=%s, fallback=%s",
        stt.model_name,
        duration if duration is not None else -1.0,
        style,
        file_ext,
        fallback_reason or "no",
    )

    # Spec 028 Item 2: wait for the local model to be ready before the
    # transcribe call instead of racing it. Skipped entirely for cloud
    # providers. See app.stt.local_setup.await_local_ready's docstring for
    # why a plain "not ready" (False) is NOT treated as fatal here -- only a
    # genuine LocalReadinessTimeout is; transcribe() keeps its own lazy
    # _get_model() fallback either way.
    if is_local_provider(stt):
        from app.stt.local_setup import LocalReadinessTimeout, await_local_ready

        try:
            await await_local_ready(settings.stt)
        except LocalReadinessTimeout as e:
            log.error("Local STT readiness wait failed: %s", e)
            raise RuntimeError(str(e)) from e

    try:
        result = await stt.transcribe(
            audio_path,
            language=language,
            style=style,
            audio_duration=duration,
        )
    except Exception:
        log.exception("STT transcribe failed (%s)", stt.model_name)
        raise

    text = result.text

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
        except Exception as e:
            log.warning("Clipboard copy failed: %s", e)

    duration_ms = int((time.perf_counter() - start) * 1000)
    word_count = len(text.split()) if text else 0

    # An explicit user language choice is NEVER overridden by a provider's
    # guess -- only the "auto" sentinel gets substituted, and only when the
    # provider actually reported something (otherwise "auto" stays, current
    # behaviour). See docs/adr/016-detected-language-on-stt-contract.md.
    effective_language = language
    if language == "auto" and result.detected_language:
        effective_language = result.detected_language

    try:
        entry = save_entry(
            text=text,
            duration_ms=duration_ms,
            language=effective_language,
            style=style,
            model_name=stt.model_name,
            tokens_used=result.tokens_used,
            audio_duration_seconds=duration,
            word_count=word_count,
        )
        if background_tasks is not None and text:
            from app.core import vector_store

            background_tasks.add_task(vector_store.embed_entry_background, entry.id, text)
            background_tasks.add_task(vector_store.run_background_indexer)
    except Exception as e:
        log.warning("Failed to save history entry: %s", e)

    return ProcessingResult(
        text=text,
        duration_ms=duration_ms,
        copied_to_clipboard=copied,
        model_name=stt.model_name,
        fallback_reason=fallback_reason,
    )

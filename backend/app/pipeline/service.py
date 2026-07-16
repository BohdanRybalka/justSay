"""Audio processing pipeline: Audio -> smart-routed STT -> Clipboard.

Gemini's ai_prompt style handles transcription and structuring in a single call;
short normal-style audio goes through Groq Whisper for minimum latency.
``/llm/process`` remains as a standalone endpoint for explicit cleanup.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pyperclip
from fastapi import BackgroundTasks

from app.core.config import settings
from app.core.history import save_entry
from app.pipeline.utils import detect_duration
from app.stt import get_routed_provider

log = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Domain model for pipeline output."""

    text: str
    duration_ms: int
    copied_to_clipboard: bool
    model_name: str = ""               # which provider actually handled the call
    fallback_reason: str | None = None  # set when a pinned engine was overridden


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

    try:
        entry = save_entry(
            text=text,
            duration_ms=duration_ms,
            language=language,
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

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
from app.audio.vad import analyze_vad
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

    # Pre-model silence gate (spec 029 -> 033 -> 034; see
    # docs/adr/015-pipeline-level-silence-guard.md, 019-ten-vad-neural-
    # silence-gate.md and 020-lazy-energy-guard-fallback.md). It sits between
    # duration detection and provider routing so a silent clip never reaches a
    # provider at all -- saves STT latency locally and real Cloud API spend.
    #
    # NOTE ON NUMBERING: the layer numbers below are ADR 019's and are NOT the
    # execution order. ADR 019 fixes layer 1 = energy guard, layer 2 = TEN VAD,
    # layer 3 = provider metadata, and those identities are stable — they name
    # WHICH detector, and the WARNING logs (`layer=vad` / `layer=energy`) key
    # off the names, not the numbers. ADR 020 changed only the order in which
    # layers 1 and 2 execute (2 first, 1 lazily on abstention); it deliberately
    # did not renumber them, so a reader following either ADR pointer from this
    # file finds the same layer identities it uses.
    #
    # Layer 2, the neural front gate, EXECUTES FIRST and OUTRANKS everything
    # below it: when the VAD produced a verdict, that verdict decides, in both
    # directions. It catches loud non-speech (clicks, hum, noise) that a
    # loudness threshold passes by definition, and it does not share the
    # energy guard's residual false-positive zone on quiet speech.
    # `analyze_vad` fails open (returns None) on a missing binary, a load
    # failure, or an undecodable file -- and on every non-Windows platform,
    # where no binary ships at all.
    #
    # Run via asyncio.to_thread -- synchronous CPU/IO work, like every
    # comparable call in this codebase (local.py's _transcribe, local_mlx.py's
    # _run_mlx).
    vad = None
    if settings.audio.silence_vad_enabled:
        vad = await asyncio.to_thread(analyze_vad, audio_path, settings.audio)

    # Layer 1, the energy guard -- a LAZY fallback that now EXECUTES SECOND
    # (ADR 020 amends ADR 019's "runs first, cheapest"). It is invoked
    # ONLY when the VAD abstained or is disabled, because whenever a VAD
    # verdict exists the energy verdict is overridden anyway: computing it was
    # paying a full-file decode (measured 1707ms on a 6.4min upload, vs the
    # VAD's 15.9ms early exit) to decide nothing. When the VAD abstains this
    # is the SOLE authority and behaves bit-identically to shipped spec 029 --
    # that is what "demoted, not deleted" means mechanically, and it is the
    # only silence gate that exists on macOS and on un-fetched checkouts.
    analysis = None
    if vad is None:
        analysis = await asyncio.to_thread(analyze_silence, audio_path, settings.audio)

    discard_log: tuple[str, tuple] | None = None
    if vad is not None and vad.is_silent:
        discard_log = (
            "Discarding no-speech audio (layer=vad): speech_hops=%d/%d, max_prob=%.3f",
            (vad.speech_frame_count, vad.total_frame_count, vad.max_probability),
        )
    elif analysis is not None and analysis.is_silent:
        discard_log = (
            "Discarding silent audio (layer=energy): peak=%.1f dBFS, speech_frames=%d/%d",
            (analysis.peak_dbfs, analysis.speech_frame_count, analysis.total_frame_count),
        )

    if discard_log is not None:
        log.warning(discard_log[0], *discard_log[1])
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

    # Layer 3, post-model (spec 033 / ADR 019). Whisper's own suppression is
    # `no_speech_prob > 0.6 AND avg_logprob < -1.0`, so a CONFIDENTLY-decoded
    # hallucination is never suppressed by the library -- thresholding
    # no_speech_prob alone is strictly stronger. `result.no_speech_prob` is
    # the MINIMUM across segments, so this discards only when EVERY segment
    # looks like non-speech; one confident-speech segment keeps the whole
    # result. None (no signal on this provider/path) always keeps it.
    if (
        result.no_speech_prob is not None
        and result.no_speech_prob > settings.stt.no_speech_prob_threshold
    ):
        log.warning(
            "Discarding transcription (layer=provider-metadata): no_speech_prob=%.3f > %.3f "
            "(%s, %d chars discarded)",
            result.no_speech_prob, settings.stt.no_speech_prob_threshold,
            stt.model_name,
            # Length only -- never the hallucinated text itself at WARNING.
            len(result.text),
        )
        return ProcessingResult(
            text="",
            duration_ms=int((time.perf_counter() - start) * 1000),
            copied_to_clipboard=False,
            model_name=stt.model_name,
            discarded_reason="silence",
        )

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

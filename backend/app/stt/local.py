"""Local STT provider — Faster-Whisper Large-v3.

Selected by `app.stt.local_factory.get_local_provider_class()` on
Windows / Linux / macOS-Intel. On macOS Apple Silicon the factory returns
`MLXWhisperSTTProvider` instead, which is Metal-accelerated via mlx-whisper.
"""

import asyncio
import gc
import logging
import threading
from pathlib import Path

from app.stt.base import (
    STTProvider,
    TranscriptionResult,
    coerce_no_speech_prob,
    normalize_detected_language,
)
from app.stt.config import STTSettings

log = logging.getLogger(__name__)


class LocalSTTProvider(STTProvider):
    """Faster-Whisper Large-v3 — local privacy-first STT provider.

    Model is auto-downloaded on first use (~3GB).
    GPU (CUDA) is used if available, falls back to CPU.
    Requires: pip install justsay-backend[local]
    """

    is_local = True

    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._model = None
        self._last_load_error: str | None = None
        self._load_lock: threading.Lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return f"whisper/{self._settings.whisper_model_size}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_load_error(self) -> str | None:
        return self._last_load_error

    def _get_model(self):
        with self._load_lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel

                    device = self._settings.whisper_device
                    if device == "auto":
                        device = self._detect_device()

                    compute_type = "float16" if device == "cuda" else "int8"

                    log.info(
                        "Loading whisper: model=%s device=%s compute=%s",
                        self._settings.whisper_model_size, device, compute_type,
                    )
                    self._model = WhisperModel(
                        self._settings.whisper_model_size,
                        device=device,
                        compute_type=compute_type,
                    )
                    self._last_load_error = None
                    log.info("Whisper loaded successfully")
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    self._last_load_error = msg
                    log.exception("Whisper load failed: %s", msg)
                    raise
        return self._model

    @staticmethod
    def _detect_device() -> str:
        """Resolve to "cuda"/"cpu" — faster-whisper (CTranslate2) has no
        third device string, so AMD/Intel still fall back to CPU even though
        `gpu_probe` now identifies them by name instead of staying silent.
        """
        from app.core.gpu_probe import GpuVendor, probe_gpu

        result = probe_gpu()
        if result.vendor == GpuVendor.NVIDIA:
            log.info("GPU probe: NVIDIA detected (%s) — using cuda", result.name or "unknown")
            return "cuda"
        if result.vendor in (GpuVendor.AMD, GpuVendor.INTEL):
            log.info(
                "GPU probe: %s GPU detected (%s) but faster-whisper has no %s "
                "backend — falling back to CPU",
                result.vendor.value.upper(), result.name or "unknown device",
                result.vendor.value.upper(),
            )
        else:
            log.info("GPU probe: no GPU detected — using cpu")
        return "cpu"

    async def transcribe(
        self, audio_path: Path, language: str = "uk", **kwargs
    ) -> TranscriptionResult:
        """Transcribe locally. ``style`` kwarg is accepted for interface parity but ignored.

        ``audio_duration`` (kwarg, seconds) — when provided, drives a
        latency-vs-accuracy decision: short clips get ``beam_size=1`` and
        ``condition_on_previous_text=False`` (kills silence-hallucination
        cascade); long clips keep ``beam_size=5`` and cross-segment context.
        """
        model = await asyncio.to_thread(self._get_model)
        audio_duration = kwargs.get("audio_duration")

        threshold = self._settings.cloud_routing_threshold
        is_short = audio_duration is not None and audio_duration <= threshold

        beam_size = 1 if is_short else 5
        condition_on_previous_text = not is_short
        glossary = self._settings.initial_prompt.strip() or None
        whisper_language = None if language == "auto" else language

        log.info(
            "faster-whisper: transcribe model=%s file=%s lang=%s "
            "duration=%s beam_size=%d cond_prev=%s glossary=%s",
            self._settings.whisper_model_size, audio_path.name, language,
            f"{audio_duration:.1f}s" if audio_duration is not None else "?",
            beam_size, condition_on_previous_text,
            f"{len(glossary)}chars" if glossary else "none",
        )

        def _transcribe() -> tuple[str, str | None, float | None]:
            segments, info = model.transcribe(
                str(audio_path),
                language=whisper_language,
                beam_size=beam_size,
                vad_filter=True,
                condition_on_previous_text=condition_on_previous_text,
                no_repeat_ngram_size=3,
                initial_prompt=glossary,
            )
            parts: list[str] = []
            no_speech_probs: list[float] = []
            for segment in segments:
                parts.append(segment.text.strip())
                probability = coerce_no_speech_prob(
                    getattr(segment, "no_speech_prob", None)
                )
                if probability is not None:
                    no_speech_probs.append(probability)
            text = " ".join(parts)
            return text, info.language, min(no_speech_probs) if no_speech_probs else None

        text, detected_raw, no_speech_prob = await asyncio.to_thread(_transcribe)
        return TranscriptionResult(
            text=text,
            tokens_used=None,
            detected_language=normalize_detected_language(detected_raw),
            no_speech_prob=no_speech_prob,
        )

    def cleanup(self) -> None:
        """Release whisper model and GPU memory.

        `cleanup()` is reachable synchronously from `PUT /stt/mode`'s
        `clear_cache()` on the FastAPI event-loop thread, so it must never
        block on `_load_lock` for the length of a multi-minute first-run
        model download — that would stall the entire event loop. A
        non-blocking acquire lets an in-flight `_get_model()` call win: if
        the lock is busy, log and return without touching `self._model`,
        `gc.collect()`, or `torch.cuda` (the load's own caller is
        responsible for cleaning up an orphaned load after the fact, e.g.
        `ensure_local_ready()`'s post-load mode recheck).
        """
        if not self._load_lock.acquire(blocking=False):
            log.info("cleanup() skipped: a model load is in flight (lock busy)")
            return
        try:
            if self._model is not None:
                log.info("Releasing whisper model from memory")
                del self._model
                self._model = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        log.info("CUDA cache cleared")
                except ImportError:
                    pass
        finally:
            self._load_lock.release()

"""Local STT provider — Faster-Whisper Large-v3.

`app.stt.local_factory.get_local_provider_class` selects this provider on
Windows NVIDIA/no-GPU, Linux and macOS-Intel, and returns
`WhisperCppServerSTTProvider` elsewhere. That factory is imported inside
`_get_model` and not at module level: it imports this module back, so a
module-level import here would close an import cycle.
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
    latched_load_error,
    normalize_detected_language,
)
from app.stt.config import STTSettings

log = logging.getLogger(__name__)

SHORT_CLIP_SECONDS = 30.0


class LocalSTTProvider(STTProvider):
    """Faster-Whisper Large-v3 — local privacy-first STT provider.

    The model is auto-downloaded on first use (~3GB) and runs on CUDA when
    available, CPU otherwise. Requires `pip install justsay-backend[local]`.
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

                    from app.stt.local_factory import (
                        LocalProviderKind,
                        compute_type_for_device,
                    )

                    device = self._settings.whisper_device
                    if device == "auto":
                        device = self._detect_device()

                    compute_type = compute_type_for_device(
                        device, LocalProviderKind.FASTER_WHISPER
                    )

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
                    msg = latched_load_error(e)
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
        """Transcribe locally. ``audio_duration`` (kwarg, seconds) picks beam 1
        without cross-segment context at or below `SHORT_CLIP_SECONDS`, beam 5
        with it otherwise -- never `cloud_routing_threshold` (ADR 073). An unknown
        duration takes the long path.
        """
        model = await asyncio.to_thread(self._get_model)
        audio_duration = kwargs.get("audio_duration")

        is_short = audio_duration is not None and audio_duration <= SHORT_CLIP_SECONDS

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

        Never blocks: an in-flight load wins the lock and this returns having
        touched nothing, leaving that load's own caller to release it.
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

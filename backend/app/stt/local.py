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

from app.stt.base import STTProvider, TranscriptionResult
from app.stt.config import STTSettings

log = logging.getLogger(__name__)


class LocalSTTProvider(STTProvider):
    """Faster-Whisper Large-v3 — local privacy-first STT provider.

    Model is auto-downloaded on first use (~3GB).
    GPU (CUDA) is used if available, falls back to CPU.
    Requires: pip install justsay-backend[local]
    """

    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._model = None
        self._last_load_error: str | None = None
        # Sync primitive — `_get_model` is called both directly on the event
        # loop thread (from `transcribe`) and via `asyncio.to_thread` (from
        # `router.py`'s `/stt/local/load`), so a genuine OS-thread race is
        # possible; `asyncio.Lock` would not serialise the latter call path.
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
                    # Format with type so the UI shows e.g. "OSError: [WinError 126] ..."
                    msg = f"{type(e).__name__}: {e}"
                    self._last_load_error = msg
                    log.exception("Whisper load failed: %s", msg)
                    raise
        return self._model

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    async def transcribe(self, audio_path: Path, language: str = "uk", **kwargs) -> TranscriptionResult:
        """Transcribe locally. ``style`` kwarg is accepted for interface parity but ignored.

        ``audio_duration`` (kwarg, seconds) — when provided, drives a
        latency-vs-accuracy decision: short clips get ``beam_size=1`` and
        ``condition_on_previous_text=False`` (kills silence-hallucination
        cascade); long clips keep ``beam_size=5`` and cross-segment context.
        """
        model = self._get_model()
        audio_duration = kwargs.get("audio_duration")

        # Reuse the cloud routing threshold so "short" means the same thing
        # everywhere in the pipeline. Drift between two magic-30.0 constants
        # was caught by entry-gate /architect.
        threshold = self._settings.cloud_routing_threshold
        is_short = audio_duration is not None and audio_duration <= threshold

        beam_size = 1 if is_short else 5
        # Cross-segment context helps long meetings stay coherent but caused
        # silence-hallucination cascades on short dictation (research:
        # docs/research/whisper-llm-need.md). Only disable for short clips.
        condition_on_previous_text = not is_short
        glossary = self._settings.initial_prompt.strip() or None

        log.info(
            "faster-whisper: transcribe model=%s file=%s lang=%s "
            "duration=%s beam_size=%d cond_prev=%s glossary=%s",
            self._settings.whisper_model_size, audio_path.name, language,
            f"{audio_duration:.1f}s" if audio_duration is not None else "?",
            beam_size, condition_on_previous_text,
            # Never log glossary content — could leak PII / a stray API key
            # the user pasted by mistake. Only the bool/length is observable.
            f"{len(glossary)}chars" if glossary else "none",
        )

        def _transcribe():
            segments, _info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                condition_on_previous_text=condition_on_previous_text,
                no_repeat_ngram_size=3,
                initial_prompt=glossary,
            )
            return " ".join(segment.text.strip() for segment in segments)

        text = await asyncio.to_thread(_transcribe)
        return TranscriptionResult(text=text, tokens_used=None)

    def cleanup(self) -> None:
        """Release whisper model and GPU memory."""
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

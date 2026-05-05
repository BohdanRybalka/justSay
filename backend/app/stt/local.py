"""Local STT provider — Faster-Whisper Large-v3."""

import asyncio
import gc
import logging
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

    @property
    def model_name(self) -> str:
        return f"whisper/{self._settings.whisper_model_size}"

    @property
    def last_load_error(self) -> str | None:
        return self._last_load_error

    def _get_model(self):
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
        """Transcribe locally. ``style`` kwarg is accepted for interface parity but ignored."""
        model = self._get_model()
        log.info(
            "faster-whisper: transcribe model=%s file=%s lang=%s",
            self._settings.whisper_model_size, audio_path.name, language,
        )

        def _transcribe():
            segments, _info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                vad_filter=True,
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

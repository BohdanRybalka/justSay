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
from collections.abc import Callable
from pathlib import Path

from app.core.errors import ResourceUnavailableError
from app.stt.base import (
    WHISPER_WINDOW_SECONDS,
    STTProvider,
    TranscriptionResult,
    coerce_no_speech_prob,
    latched_load_error,
    normalize_detected_language,
)
from app.stt.config import STTSettings
from app.stt.engine_use import EngineUse
from app.stt.glossary import glossary_summary, whisper_glossary
from app.stt.model_download import UNVERIFIED_SIZE

log = logging.getLogger(__name__)

SHORT_CLIP_SECONDS = 30.0

FASTER_WHISPER_REVISIONS: dict[str, str] = {
    "tiny": "d90ca5fe260221311c53c58e660288d3deb8d356",
    "base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    "small": "536b0662742c02347bc0e980a01041f333bce120",
    "medium": "08e178d48790749d25932bbc082711ddcfdfbc4f",
    "large-v2": "f0fe81560cb8b68660e564f55dd99207059c092e",
    "large-v3": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    "large-v3-turbo": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
}


def model_folder(model_size: str) -> str:
    """The folder holding ``model_size``, from the local cache without any network call.

    A missing model is downloaded once at its pinned revision; a size with no pinned
    revision is refused with ``ResourceUnavailableError`` instead of fetched from a branch.
    """
    from faster_whisper.utils import download_model

    try:
        return download_model(model_size, local_files_only=True)
    except Exception:
        revision = FASTER_WHISPER_REVISIONS.get(model_size)
        if revision is None:
            raise ResourceUnavailableError(UNVERIFIED_SIZE) from None
        return download_model(model_size, revision=revision)


class LocalSTTProvider(STTProvider):
    """Faster-Whisper Large-v3 — local privacy-first STT provider.

    The model is downloaded once on first use (~3GB, pinned revision), loads from the
    local cache without network afterwards, and runs on CUDA when available, CPU
    otherwise. Requires `pip install justsay-backend[local]`.
    """

    is_local = True
    longest_piece_seconds = WHISPER_WINDOW_SECONDS

    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._model = None
        self._loaded_with: tuple[str, str] | None = None
        self._last_load_error: str | None = None
        self._load_lock: threading.Lock = threading.Lock()
        self._use = EngineUse(
            is_stale=self._is_stale, is_ready=lambda: not self._is_stale(), unload=self._unload
        )

    @property
    def model_name(self) -> str:
        return f"whisper/{self._settings.whisper_model_size}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and not self._is_stale()

    @property
    def last_load_error(self) -> str | None:
        return self._last_load_error

    def _wanted(self) -> tuple[str, str]:
        return self._settings.whisper_model_size, self._settings.whisper_device

    def _is_stale(self) -> bool:
        """Whether the loaded model was made for other settings than the ones chosen now."""
        return self._loaded_with is not None and self._loaded_with != self._wanted()

    def _get_model(self):
        """The loaded model; one made for other settings is replaced once nothing runs on it."""
        self._use.settle()
        with self._load_lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel

                    from app.stt.local_factory import (
                        LocalProviderKind,
                        compute_type_for_device,
                    )

                    wanted = self._wanted()
                    model_size, device = wanted
                    if device == "auto":
                        device = self._detect_device()

                    compute_type = compute_type_for_device(
                        device, LocalProviderKind.FASTER_WHISPER
                    )

                    log.info(
                        "Loading whisper: model=%s device=%s compute=%s",
                        model_size, device, compute_type,
                    )
                    self._model = WhisperModel(
                        model_folder(model_size),
                        device=device,
                        compute_type=compute_type,
                    )
                    self._loaded_with = wanted
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
        with it otherwise. An unknown duration takes the long path.
        """
        model = await self._use.enter(self._get_model)
        try:
            return await self._transcribe_with(model, audio_path, language, kwargs)
        finally:
            self._use.leave()

    async def _transcribe_with(
        self, model, audio_path: Path, language: str, kwargs: dict
    ) -> TranscriptionResult:
        audio_duration = kwargs.get("audio_duration")

        is_short = audio_duration is not None and audio_duration <= SHORT_CLIP_SECONDS

        beam_size = 1 if is_short else 5
        condition_on_previous_text = not is_short
        glossary, dropped_terms = whisper_glossary(self._settings.initial_prompt)
        whisper_language = None if language == "auto" else language

        log.info(
            "faster-whisper: transcribe model=%s file=%s lang=%s "
            "duration=%s beam_size=%d cond_prev=%s glossary=%s",
            self._settings.whisper_model_size, audio_path.name, language,
            f"{audio_duration:.1f}s" if audio_duration is not None else "?",
            beam_size, condition_on_previous_text,
            glossary_summary(glossary, dropped_terms),
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
            self._release_model()
        finally:
            self._load_lock.release()

    def release_when_idle(self, still_unused: Callable[[], bool]) -> None:
        self._use.release_when_idle(still_unused)

    def _unload(self, only_if: Callable[[], bool] | None = None) -> None:
        """Release the model, waiting for a load in flight instead of skipping it.

        With ``only_if`` it releases only while that is still true after the wait.
        """
        with self._load_lock:
            if only_if is None or only_if():
                self._release_model()

    def _release_model(self) -> None:
        if self._model is None:
            return
        log.info("Releasing whisper model from memory")
        del self._model
        self._model = None
        self._loaded_with = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log.info("CUDA cache cleared")
        except ImportError:
            pass

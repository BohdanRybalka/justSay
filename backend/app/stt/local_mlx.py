"""MLX-Whisper local STT provider for macOS Apple Silicon.

Selected by `app.stt.local_factory.get_local_provider_class()` when running
natively on M1+ hardware. Provides Metal-accelerated transcription via the
`mlx-whisper` package while honouring the same `STTProvider` contract as the
faster-whisper based `LocalSTTProvider` used on Windows/Linux/macOS-Intel.

Privacy: after a model has been downloaded once, `HF_HUB_OFFLINE=1` is set
before every subsequent `mlx_whisper.load_models.load_model` call so the
process never re-contacts huggingface.co for ETag checks. Switching
`whisper_model_size` to a model that is not yet cached pops the flag for the
duration of that load so the first-time download can proceed.
"""

import asyncio
import gc
import logging
import os
import threading
from pathlib import Path

from huggingface_hub import scan_cache_dir

from app.stt.base import STTProvider, TranscriptionResult
from app.stt.config import STTSettings

log = logging.getLogger(__name__)


# Mapping from `STTSettings.whisper_model_size` to the actual mlx-community
# Hugging Face repo ID. The naming convention is NOT a clean prefix:
# `large-v3-turbo` has no `-mlx` suffix; all other sizes do. Hard-coded by
# choice — see plan 019, RED-2 (iteration 1) for the audit trail.
MLX_REPO_BY_SIZE: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _hf_cache_has_snapshot(repo_id: str) -> bool:
    """True when `scan_cache_dir` reports at least one valid revision for repo.

    Uses the public `huggingface_hub.scan_cache_dir` API rather than reading
    the on-disk `models--<slug>/snapshots/` layout directly: scan_cache_dir
    filters out empty / corrupt snapshots, so a partial download cannot trick
    us into setting `HF_HUB_OFFLINE=1` and then failing the next load.
    """
    try:
        info = scan_cache_dir()
    except Exception as e:
        # Brand-new machine: cache dir does not exist yet, or is unreadable.
        log.debug("scan_cache_dir failed, treating cache as empty: %s", e)
        return False
    return any(repo.repo_id == repo_id for repo in info.repos)


class MLXWhisperSTTProvider(STTProvider):
    """MLX-Whisper local STT provider — macOS Apple Silicon (M1+).

    Model is auto-downloaded on first use from mlx-community on Hugging Face
    (~1.6 GB for `large-v3-turbo`). Subsequent loads in the same process are
    offline; subsequent process starts re-enable offline mode once the cache
    is populated.

    Requires: `pip install justsay-backend[local-mac]`.
    """

    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._loaded: bool = False
        self._last_load_error: str | None = None
        # Sync primitive — `_get_model` runs on a worker thread via
        # `asyncio.to_thread`, so `asyncio.Lock` would not serialise it.
        self._load_lock: threading.Lock = threading.Lock()
        # Async primitive — guards the async `transcribe` body.
        self._transcribe_lock: asyncio.Lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return f"whisper-mlx/{self._settings.whisper_model_size}"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def last_load_error(self) -> str | None:
        return self._last_load_error

    def _get_repo_id(self) -> str:
        """Resolve the HF repo ID for the configured model size.

        Pure lookup. Raises `ValueError` for unsupported sizes; latching of
        the error onto `_last_load_error` is owned by `_get_model`'s
        `try/except` so the order is resolution → short-circuit → load.
        """
        size = self._settings.whisper_model_size
        try:
            return MLX_REPO_BY_SIZE[size]
        except KeyError:
            supported = ", ".join(sorted(MLX_REPO_BY_SIZE))
            raise ValueError(
                f"Unsupported MLX whisper_model_size: {size!r}. "
                f"Supported sizes: {supported}"
            )

    def _get_model(self) -> str:
        """Lazy-load entrypoint. Sync, called from a worker thread.

        Returns the resolved Hugging Face ``repo_id`` so callers (notably the
        ``_run_mlx`` closure in ``transcribe``) can reuse the validated value
        without a second ``_get_repo_id()`` call outside the ``try/except``.

        Same name as ``LocalSTTProvider._get_model`` so ``router.py``'s
        ``await asyncio.to_thread(provider._get_model)`` works unchanged
        across platforms (the return value is ignored on the router side).
        """
        with self._load_lock:
            try:
                from mlx_whisper.transcribe import ModelHolder

                repo_id = self._get_repo_id()
                if (
                    ModelHolder.model is not None
                    and ModelHolder.model_path == repo_id
                ):
                    self._last_load_error = None
                    return repo_id  # already warm for this repo

                from mlx_whisper.load_models import load_model

                if _hf_cache_has_snapshot(repo_id):
                    os.environ["HF_HUB_OFFLINE"] = "1"
                else:
                    # Allow first-time download of a *new* model size when the
                    # offline flag was set by a previous load. Without this,
                    # switching whisper_model_size to an uncached value in the
                    # same process would fail with `OfflineModeIsEnabled`.
                    os.environ.pop("HF_HUB_OFFLINE", None)

                log.info("Loading MLX whisper: repo=%s", repo_id)
                load_model(repo_id)
                self._last_load_error = None
                log.info("MLX whisper loaded")
                return repo_id
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self._last_load_error = msg
                log.exception("MLX whisper load failed: %s", msg)
                raise

    async def transcribe(
        self, audio_path: Path, language: str = "uk", **kwargs
    ) -> TranscriptionResult:
        """Transcribe locally via mlx-whisper. ``style`` kwarg is accepted for
        interface parity but ignored.

        ``audio_duration`` (kwarg, seconds) — when provided, drives the
        latency-vs-accuracy decision: short clips get ``beam_size=1``,
        ``condition_on_previous_text=False``, and explicit ``temperature=0.0``
        (so the upstream temperature-fallback loop does not strip
        ``beam_size``); long clips keep ``beam_size=5`` and cross-segment
        context with the default temperature tuple.
        """
        audio_duration = kwargs.get("audio_duration")
        threshold = self._settings.cloud_routing_threshold
        is_short = audio_duration is not None and audio_duration <= threshold

        beam_size = 1 if is_short else 5
        condition_on_previous_text = not is_short
        glossary = self._settings.initial_prompt.strip() or None

        log.info(
            "mlx-whisper: transcribe model=%s file=%s lang=%s "
            "duration=%s beam_size=%d cond_prev=%s glossary=%s",
            self._settings.whisper_model_size,
            audio_path.name,
            language,
            f"{audio_duration:.1f}s" if audio_duration is not None else "?",
            beam_size,
            condition_on_previous_text,
            # Never log glossary content — could leak PII / API keys.
            f"{len(glossary)}chars" if glossary else "none",
        )

        async with self._transcribe_lock:

            def _run_mlx() -> str:
                # _get_model is sync; running it inside _run_mlx keeps the
                # event loop free during a cold first-download. The returned
                # repo_id is the same value that was validated and used to
                # warm ModelHolder — reusing it here avoids a second
                # `_get_repo_id()` call outside the try/except boundary.
                repo_id = self._get_model()
                import mlx_whisper

                kwargs_mlx: dict = {
                    "path_or_hf_repo": repo_id,
                    "language": language,
                    "beam_size": beam_size,
                    "condition_on_previous_text": condition_on_previous_text,
                    "initial_prompt": glossary,
                }
                if is_short:
                    # Explicit temperature=0.0 (scalar) disables the upstream
                    # temperature-fallback loop, which strips beam_size when
                    # t > 0. Short clips prioritise latency over rescue retries.
                    kwargs_mlx["temperature"] = 0.0
                # NB: no_repeat_ngram_size and vad_filter are NOT valid
                # mlx-whisper kwargs — verified against the upstream
                # DecodingOptions dataclass. Do not add them.
                result = mlx_whisper.transcribe(str(audio_path), **kwargs_mlx)
                segments = result.get("segments") or []
                if segments:
                    return " ".join(seg["text"].strip() for seg in segments)
                # Fallback for whole-clip-text responses on some mlx-whisper
                # versions: `text` is provided even without segment splits.
                return (result.get("text") or "").strip()

            text = await asyncio.to_thread(_run_mlx)
            self._loaded = True
            return TranscriptionResult(text=text, tokens_used=None)

    def cleanup(self) -> None:
        """Release MLX model and Metal memory.

        Mirrors LocalSTTProvider.cleanup()'s non-blocking-lock guard (spec 015,
        RED-3 — iteration 2 triage): `cleanup()` is reachable synchronously from
        `PUT /stt/mode`'s `clear_cache()` on the FastAPI event-loop thread, so it
        must never block on `_load_lock` for a multi-minute first-run download —
        nor race `_get_model()`'s own writes to `ModelHolder.model`/
        `ModelHolder.model_path` (process-wide global state owned by the
        `mlx_whisper` package). If the lock is busy, log and return without
        touching `ModelHolder`, `self._loaded`, `gc.collect()`, or the Metal
        cache — the load's own caller (`ensure_local_ready()`'s post-load
        identity recheck) is responsible for cleaning up an orphaned load
        afterwards.
        """
        if not self._load_lock.acquire(blocking=False):
            log.info("cleanup() skipped: a model load is in flight (lock busy)")
            return
        try:
            self._loaded = False
            try:
                from mlx_whisper.transcribe import ModelHolder

                ModelHolder.model = None
                ModelHolder.model_path = None
            except ImportError:
                pass
            gc.collect()
            try:
                import mlx.core as mx

                metal = getattr(mx, "metal", None)
                clear = getattr(metal, "clear_cache", None) if metal else None
                if callable(clear):
                    clear()
                    log.info("MLX Metal cache cleared")
            except ImportError:
                pass
        finally:
            self._load_lock.release()

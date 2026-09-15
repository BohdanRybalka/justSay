"""Embedding provider selection — mirrors the shape of ``app.stt``.

Eligibility is DERIVED from the one Cloud/Local toggle the user can operate,
``stt.mode``, never a toggle of its own:

  - Cloud embeddings when ``stt.mode == CLOUD`` (reuses the Gemini key
    already present for cloud STT).
  - Local embeddings when ``stt.mode == LOCAL`` AND Ollama reports
    ``nomic-embed-text`` pulled; otherwise the feature is disabled with
    ``LOCAL_MISSING_MODEL_REASON``.

There is no third state: with a single switch there is no mixed pairing to
report. ``docs/adr/071-semantic-search-keys-on-the-dictation-mode.md`` records
why, superseding the two-toggle formulation in
``docs/adr/001-sqlite-vec-embedding-provider-selection.md``, which holds for
everything else about this package.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Protocol

from app.core.types import ProviderMode
from app.embeddings.config import EmbeddingSettings
from app.stt.config import STTSettings

log = logging.getLogger(__name__)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingSettings",
    "resolve_embedding_provider",
    "clear_cache",
]


class EmbeddingProvider(Protocol):
    model_name: str

    async def embed(self, text: str) -> list[float]: ...

    def cleanup(self) -> None:
        """Release resources (model memory, connections).

        Called on mode switch and app shutdown. Structural protocol —
        every concrete provider must define this itself (no shared base
        class here).
        """


LOCAL_MISSING_MODEL_REASON = (
    "Local embeddings need Ollama with nomic-embed-text pulled — run "
    "`ollama pull nomic-embed-text`"
)

_cache_lock = threading.Lock()
_local_reprobe_lock = asyncio.Lock()
_cached_provider: EmbeddingProvider | None = None
_cached_reason: str | None = None
_cached_key: ProviderMode | None = None


async def resolve_embedding_provider(
    stt: STTSettings, emb: EmbeddingSettings
) -> tuple[EmbeddingProvider | None, str | None]:
    """Factory with caching, keyed on ``stt.mode`` — same cached-mode pattern
    as ``app.stt.get_provider``. Deliberately ``async`` (unlike the STT
    factory) because the Local-mode branch must probe Ollama's tag list over
    HTTP to check for ``nomic-embed-text`` before deciding eligibility.

    A ``LOCAL`` cache entry is re-probed against Ollama's tag list on *every*
    call, in both directions: a cached negative result caused by a missing
    local model (``LOCAL_MISSING_MODEL_REASON``) re-checks in case the model
    has since appeared, and a cached positive result (a working
    ``LocalEmbeddingProvider``) re-checks in case the model has since
    disappeared (e.g. ``ollama rm nomic-embed-text``) — the stale provider's
    ``cleanup()`` is called before it's dropped from cache. While the model
    remains available across consecutive calls, the same
    ``LocalEmbeddingProvider`` instance is reused rather than reconstructed.
    A ``CLOUD`` result is cached as before — that key never enters this
    re-probe branch.

    Concurrent ``LOCAL`` callers queue behind ``_local_reprobe_lock`` and run
    their entire probe/decide/cleanup-or-reuse/cache-write sequence strictly
    one at a time, so a later call always observes the prior call's
    fully-committed result before making its own cleanup-or-reuse decision.
    This does not change the no-coalescing design above: each queued call
    still independently re-probes Ollama (N concurrent callers still make N
    sequential HTTP round-trips, just serialized rather than racing).
    Cloud resolution is unaffected and keeps using only ``_cache_lock``.
    """
    global _cached_provider, _cached_reason, _cached_key

    key = stt.mode

    if key == ProviderMode.LOCAL:
        from app.embeddings.local import LocalEmbeddingProvider, is_model_available

        async with _local_reprobe_lock:
            with _cache_lock:
                stale_local_provider = _cached_provider if _cached_key == key else None

            if await is_model_available(emb.ollama_host, emb.local_model):
                provider: EmbeddingProvider | None = stale_local_provider or LocalEmbeddingProvider(
                    ollama_host=emb.ollama_host, model=emb.local_model
                )
                reason: str | None = None
            else:
                if stale_local_provider is not None:
                    try:
                        stale_local_provider.cleanup()
                    except Exception:
                        log.warning(
                            "Releasing the stale local embedding provider failed", exc_info=True
                        )
                provider = None
                reason = LOCAL_MISSING_MODEL_REASON

            with _cache_lock:
                _cached_provider = provider
                _cached_reason = reason
                _cached_key = key
        return provider, reason

    with _cache_lock:
        if _cached_key == key:
            return _cached_provider, _cached_reason

    from app.embeddings.cloud import CloudEmbeddingProvider

    provider = CloudEmbeddingProvider(gemini_api_key=stt.gemini_api_key, model=emb.cloud_model)
    reason = None

    with _cache_lock:
        _cached_provider = provider
        _cached_reason = reason
        _cached_key = key
    return provider, reason


def clear_cache() -> None:
    """Invalidate the cached provider. Call on dictation-mode, key or
    Ollama-host change.

    Hooked into ``user_settings.sync_to_runtime``'s existing
    ``changed_stt``/``changed_embeddings`` invalidation, and into ``main.py``'s
    ``lifespan`` shutdown block alongside the STT-cache release.

    Calls the cached provider's ``cleanup()`` before dropping the reference,
    so ``LocalEmbeddingProvider`` gets a chance to unload ``nomic-embed-text``
    from Ollama's memory. A cleanup failure is logged and does not propagate —
    invalidation must succeed even when the unload cannot.
    """
    global _cached_provider, _cached_reason, _cached_key
    with _cache_lock:
        if _cached_provider is not None:
            try:
                _cached_provider.cleanup()
            except Exception:
                log.warning("Releasing the cached embedding provider failed", exc_info=True)
        _cached_provider = None
        _cached_reason = None
        _cached_key = None

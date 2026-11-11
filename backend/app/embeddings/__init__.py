"""Embedding provider selection — mirrors the shape of ``app.stt``.

Eligibility is DERIVED from the one Cloud/Local toggle the user can operate,
``stt.mode``, never a toggle of its own: cloud embeddings when it is ``CLOUD``,
reusing the Gemini key already present for cloud STT; local embeddings when it
is ``LOCAL`` and Ollama reports ``nomic-embed-text`` pulled, otherwise the
feature is disabled with ``LOCAL_MISSING_MODEL_REASON``. There is no third
state (ADR 071, superseding ADR 001 on this point alone).
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
    is_local: bool

    async def embed(self, text: str) -> list[float]: ...

    def cleanup(self) -> None:
        """Release resources (model memory, connections).

        Called on mode switch and app shutdown. Structural protocol: every
        concrete provider defines this itself, with no shared base class.
        """


LOCAL_MISSING_MODEL_REASON = (
    "Local embeddings need Ollama with nomic-embed-text pulled — run "
    "`ollama pull nomic-embed-text`"
)

_RELEASE_THREAD_NAME = "embedding-provider-release"

_cache_lock = threading.Lock()
_local_reprobe_lock = asyncio.Lock()
_cached_provider: EmbeddingProvider | None = None
_cached_reason: str | None = None
_cached_key: ProviderMode | None = None


async def resolve_embedding_provider(
    stt: STTSettings, emb: EmbeddingSettings
) -> tuple[EmbeddingProvider | None, str | None]:
    """Resolve the embedding provider for ``stt.mode``, with a per-mode cache.

    ``async`` because the ``LOCAL`` branch re-probes Ollama's tag list over HTTP on
    every call, dropping or reusing the cached provider; ``CLOUD`` is cached once.
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
                    _release_off_the_loop(stale_local_provider)
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
    """Invalidate the cached provider. Call on mode, key or Ollama-host change.

    The cache is empty when this returns; the dropped provider's ``cleanup()`` runs on
    a worker thread, so a slow or hung Ollama never holds up the caller.
    """
    global _cached_provider, _cached_reason, _cached_key
    with _cache_lock:
        dropped = _cached_provider
        _cached_provider = None
        _cached_reason = None
        _cached_key = None
    if dropped is not None:
        _release_off_the_loop(dropped)


def _release_off_the_loop(provider: EmbeddingProvider) -> None:
    """Run ``provider.cleanup()`` on its own non-daemon thread, so exit still waits for it."""
    threading.Thread(
        target=_release, args=(provider,), name=_RELEASE_THREAD_NAME, daemon=False
    ).start()


def _release(provider: EmbeddingProvider) -> None:
    try:
        provider.cleanup()
    except Exception:
        log.warning("Releasing an embedding provider failed", exc_info=True)

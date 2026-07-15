"""Embedding provider selection — mirrors the shape of ``app.stt`` / ``app.llm``.

Eligibility is DERIVED from the two existing Cloud/Local toggles
(``stt.mode``, ``llm.mode``), never a third user-set toggle:

  - Cloud embeddings only when ``stt.mode == CLOUD and llm.mode == CLOUD``
    (reuses the Gemini key already present for cloud STT).
  - Local embeddings only when ``stt.mode == LOCAL and llm.mode == LOCAL``
    AND Ollama reports ``nomic-embed-text`` pulled.
  - Every other combination (including any mixed cloud/local pairing)
    disables the feature outright — the safe default when ambiguous.

See ``docs/adr/001-sqlite-vec-embedding-provider-selection.md`` for the
full reasoning.
"""

from __future__ import annotations

import threading
from typing import Protocol

from app.core.types import ProviderMode
from app.embeddings.config import EmbeddingSettings
from app.llm.config import LLMSettings
from app.stt.config import STTSettings

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
        class here), same shape as ``app.llm.LLMProvider.cleanup()``.
        """


# Reason strings are part of the public contract: history_router surfaces
# them verbatim as the 503 `detail` for `mode=semantic`, and
# /history/embeddings-status surfaces them as `reason`.
MIXED_MODE_REASON = (
    "Semantic search needs matching Cloud/Local mode on both Speech-to-Text "
    "and AI Processing"
)
LOCAL_MISSING_MODEL_REASON = (
    "Local embeddings need Ollama with nomic-embed-text pulled — run "
    "`ollama pull nomic-embed-text`"
)

_cache_lock = threading.Lock()
_cached_provider: EmbeddingProvider | None = None
_cached_reason: str | None = None
_cached_key: tuple[ProviderMode, ProviderMode] | None = None


async def resolve_embedding_provider(
    stt: STTSettings, llm: LLMSettings, emb: EmbeddingSettings
) -> tuple[EmbeddingProvider | None, str | None]:
    """Factory with caching, keyed on ``(stt.mode, llm.mode)`` — same
    cached-mode pattern as ``app.llm.get_llm_provider``. Deliberately
    ``async`` (unlike the LLM/STT factories) because the Local-mode branch
    must probe Ollama's tag list over HTTP to check for ``nomic-embed-text``
    before deciding eligibility.

    A ``(LOCAL, LOCAL)`` cache entry is re-probed against Ollama's tag list
    on *every* call, in both directions: a cached negative result caused by
    a missing local model (``LOCAL_MISSING_MODEL_REASON``) re-checks in
    case the model has since appeared, and a cached positive result
    (a working ``LocalEmbeddingProvider``) re-checks in case the model has
    since disappeared (e.g. ``ollama rm nomic-embed-text``) — the stale
    provider's ``cleanup()`` is called before it's dropped from cache. While
    the model remains available across consecutive calls, the same
    ``LocalEmbeddingProvider`` instance is reused rather than reconstructed.
    Cloud and mixed-mode results are cached as before — neither key ever
    enters this re-probe branch.
    """
    global _cached_provider, _cached_reason, _cached_key

    key = (stt.mode, llm.mode)
    is_local_key = key == (ProviderMode.LOCAL, ProviderMode.LOCAL)
    with _cache_lock:
        cache_hit = _cached_key == key
        if cache_hit and not is_local_key:
            return _cached_provider, _cached_reason
        stale_local_provider = _cached_provider if (cache_hit and is_local_key) else None

    provider: EmbeddingProvider | None
    reason: str | None

    if stt.mode == ProviderMode.CLOUD and llm.mode == ProviderMode.CLOUD:
        from app.embeddings.cloud import CloudEmbeddingProvider  # lazy: imports google-genai

        provider = CloudEmbeddingProvider(gemini_api_key=stt.gemini_api_key, model=emb.cloud_model)
        reason = None
    elif is_local_key:
        from app.embeddings.local import LocalEmbeddingProvider, is_model_available

        if await is_model_available(llm, emb.local_model):
            provider = stale_local_provider or LocalEmbeddingProvider(
                ollama_host=llm.ollama_host, model=emb.local_model
            )
            reason = None
        else:
            if stale_local_provider is not None:
                try:
                    stale_local_provider.cleanup()
                except Exception:
                    pass
            provider = None
            reason = LOCAL_MISSING_MODEL_REASON
    else:
        provider = None
        reason = MIXED_MODE_REASON

    with _cache_lock:
        _cached_provider = provider
        _cached_reason = reason
        _cached_key = key
    return provider, reason


def clear_cache() -> None:
    """Invalidate the cached provider. Call on stt/llm mode or key change.

    Hooked into ``user_settings.sync_to_runtime``'s existing
    ``changed_stt``/``changed_llm`` invalidation, and into ``main.py``'s
    ``lifespan`` shutdown block alongside ``app.llm.clear_cache()``.

    Mirrors ``app.llm.clear_cache()`` exactly: calls the cached provider's
    ``cleanup()`` (swallowing any exception) before dropping the reference,
    so ``LocalEmbeddingProvider`` gets a chance to unload
    ``nomic-embed-text`` from Ollama's memory.
    """
    global _cached_provider, _cached_reason, _cached_key
    with _cache_lock:
        if _cached_provider is not None:
            try:
                _cached_provider.cleanup()
            except Exception:
                pass
        _cached_provider = None
        _cached_reason = None
        _cached_key = None

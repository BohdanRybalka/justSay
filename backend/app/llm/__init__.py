"""LLM module — text processing with language models."""

import threading

from app.core.types import ProviderMode
from app.llm.base import LLMProvider
from app.llm.config import LLMSettings

__all__ = ["LLMProvider", "LLMSettings", "get_llm_provider", "clear_cache"]

_cache_lock = threading.Lock()
_cached_provider: LLMProvider | None = None
_cached_mode: ProviderMode | None = None


def get_llm_provider(llm_settings: LLMSettings) -> LLMProvider:
    """Factory with caching: returns cached provider if mode hasn't changed."""
    global _cached_provider, _cached_mode

    with _cache_lock:
        if _cached_provider is not None and _cached_mode == llm_settings.mode:
            return _cached_provider

        if llm_settings.mode == ProviderMode.CLOUD:
            from app.llm.cloud import CloudLLMProvider  # lazy: imports groq SDK

            _cached_provider = CloudLLMProvider(llm_settings)
        else:
            from app.llm.local import LocalLLMProvider  # lazy: imports ollama SDK

            _cached_provider = LocalLLMProvider(llm_settings)

        _cached_mode = llm_settings.mode
        return _cached_provider


def clear_cache() -> None:
    """Cleanup and invalidate cached provider. Call on config change / shutdown."""
    global _cached_provider, _cached_mode
    with _cache_lock:
        if _cached_provider is not None:
            try:
                _cached_provider.cleanup()
            except Exception:
                pass
        _cached_provider = None
        _cached_mode = None

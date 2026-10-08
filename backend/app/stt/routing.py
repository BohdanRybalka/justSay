"""Provider routing and cache on top of the STT provider classes.

:func:`get_provider` picks a provider from the mode alone. Local mode defers
to :func:`app.stt.local_factory.get_local_provider_class`; Cloud mode sends
every recording, short or long, to Groq Whisper.
"""

import logging
import threading

from app.core.types import ProviderMode
from app.stt.base import STTProvider
from app.stt.config import STTSettings

log = logging.getLogger(__name__)


_cache_lock = threading.Lock()
_providers: dict[type, STTProvider] = {}


def _get_or_create(cls, stt_settings: STTSettings) -> STTProvider:
    """Thread-safe get-or-create per provider class."""
    with _cache_lock:
        cached = _providers.get(cls)
        if cached is not None:
            return cached
        provider = cls(stt_settings)
        _providers[cls] = provider
        return provider


def _get_groq(stt_settings: STTSettings) -> STTProvider:
    from app.stt.groq_whisper import GroqWhisperSTTProvider
    return _get_or_create(GroqWhisperSTTProvider, stt_settings)


def _get_local(stt_settings: STTSettings) -> STTProvider:
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    return _get_or_create(cls, stt_settings)


def get_provider(mode: ProviderMode, stt_settings: STTSettings) -> STTProvider:
    """The provider that transcribes in ``mode``: this machine's local engine, else Groq."""
    if mode == ProviderMode.LOCAL:
        return _get_local(stt_settings)
    return _get_groq(stt_settings)


def get_local_load_error(stt_settings: STTSettings) -> str | None:
    """Return the most recent local-provider load failure, or ``None``.

    ``None`` also when no local provider is cached — nothing has attempted a
    load, so no error exists. The latch is read without a default (ADR 075).
    """
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    with _cache_lock:
        provider = _providers.get(cls)
    if provider is None:
        return None
    return provider.last_load_error


def peek_local_provider() -> STTProvider | None:
    """Read-only peek at the provider cached for the Local class, or ``None``.

    Never instantiates one, unlike :func:`get_provider`. Comparing identity
    against an earlier peek detects a cache eviction the mode never reports.
    """
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    with _cache_lock:
        return _providers.get(cls)


def is_model_loaded() -> bool:
    """Is the local whisper model currently loaded in memory?

    ``False`` when no local provider is cached. The flag is read without a
    default (ADR 075).
    """
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    with _cache_lock:
        provider = _providers.get(cls)
    if provider is None:
        return False
    return provider.is_loaded


def is_local_provider(provider: STTProvider) -> bool:
    """Is ``provider`` local? Reads the ``is_local`` class attribute the
    provider itself declares -- no I/O, no platform probe, no ``isinstance``
    chain, and no default (ADR 018).
    """
    return provider.is_local


def clear_cache() -> None:
    """Cleanup and invalidate every cached provider. Call on config change / shutdown."""
    with _cache_lock:
        for p in _providers.values():
            try:
                p.cleanup()
            except Exception:
                log.warning(
                    "Releasing the %s provider failed", type(p).__name__, exc_info=True
                )
        _providers.clear()

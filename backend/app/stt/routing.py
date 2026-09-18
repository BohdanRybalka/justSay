"""Provider routing and cache on top of the STT provider classes.

:func:`get_routed_provider` picks a provider from the mode, the engine pin,
the audio duration and the file format; :func:`get_provider` answers the
mode-level question alone, for callers holding no audio. Local mode defers
to :func:`app.stt.local_factory.get_local_provider_class`; Cloud mode routes
short audio to Groq and everything else, including unknown length, to Gemini.
"""

import logging
import threading

from app.core.types import ProviderMode
from app.stt.base import STTProvider
from app.stt.config import STTSettings

log = logging.getLogger(__name__)


GROQ_SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".mp4"}
)


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


def _get_gemini(stt_settings: STTSettings) -> STTProvider:
    from app.stt.cloud import GeminiSTTProvider
    return _get_or_create(GeminiSTTProvider, stt_settings)


def _get_groq(stt_settings: STTSettings) -> STTProvider:
    from app.stt.groq_whisper import GroqWhisperSTTProvider
    return _get_or_create(GroqWhisperSTTProvider, stt_settings)


def _get_local(stt_settings: STTSettings) -> STTProvider:
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    return _get_or_create(cls, stt_settings)


def get_provider(mode: ProviderMode, stt_settings: STTSettings) -> STTProvider:
    """Mode-level provider lookup, no routing heuristics.

    For callers with no audio in hand. Cloud mode always returns Gemini; the
    engine pin and duration routing live in :func:`get_routed_provider`.
    """
    if mode == ProviderMode.LOCAL:
        return _get_local(stt_settings)
    return _get_gemini(stt_settings)


def get_routed_provider(
    stt_settings: STTSettings,
    audio_duration: float | None = None,
    file_extension: str | None = None,
) -> tuple[STTProvider, str | None]:
    """Select a provider from the engine pin, mode, audio duration and format.

    Returns ``(provider, fallback_reason)``; the reason is a sentence for the
    UI when the requested engine had to be overridden, else ``None``.
    """
    if stt_settings.mode == ProviderMode.LOCAL:
        return _get_local(stt_settings), None

    ext = file_extension.lower() if file_extension else None
    engine = stt_settings.engine

    if engine == "gemini":
        return _get_gemini(stt_settings), None

    if engine == "groq":
        if ext is not None and ext not in GROQ_SUPPORTED_FORMATS:
            return _get_gemini(stt_settings), f"Groq doesn't support {ext}"
        return _get_groq(stt_settings), None

    duration_short = (
        audio_duration is not None and audio_duration <= stt_settings.cloud_routing_threshold
    )

    if duration_short:
        if ext is None or ext in GROQ_SUPPORTED_FORMATS:
            return _get_groq(stt_settings), None
        return _get_gemini(stt_settings), None

    return _get_gemini(stt_settings), None


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

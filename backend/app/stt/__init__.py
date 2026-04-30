"""STT module — Speech-to-Text processing.

This file is the **routing and cache** layer on top of the provider classes.
Pipeline code calls :func:`get_routed_provider` with the audio metadata
(mode, duration, style) and gets back the correct provider to use.

Routing rules (see ``docs/hybrid-stt-pipeline.md``):

======================  ========================================================
Mode / conditions        Provider
======================  ========================================================
LOCAL                    :class:`~app.stt.local.LocalSTTProvider`
CLOUD + style=ai_prompt  :class:`~app.stt.cloud.GeminiSTTProvider`
CLOUD + long audio       :class:`~app.stt.cloud.GeminiSTTProvider`
CLOUD + short + normal   :class:`~app.stt.groq_whisper.GroqWhisperSTTProvider`
CLOUD + unknown length   :class:`~app.stt.cloud.GeminiSTTProvider` (safe default)
======================  ========================================================
"""

import threading

from app.core.types import ProviderMode
from app.stt.base import STTProvider
from app.stt.config import STTSettings

__all__ = [
    "STTProvider",
    "STTSettings",
    "get_stt_provider",
    "get_routed_provider",
    "clear_cache",
    "is_model_loaded",
    "GROQ_SUPPORTED_FORMATS",
    "GEMINI_SUPPORTED_FORMATS",
]


# Per-provider accepted extensions (enforced before routing).
GROQ_SUPPORTED_FORMATS: frozenset[str] = frozenset({".wav", ".mp3", ".flac", ".ogg"})
GEMINI_SUPPORTED_FORMATS: frozenset[str] = frozenset({".wav", ".mp3", ".ogg", ".webm", ".flac"})


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
    from app.stt.local import LocalSTTProvider
    return _get_or_create(LocalSTTProvider, stt_settings)


def get_routed_provider(
    stt_settings: STTSettings,
    audio_duration: float | None = None,
    style: str = "normal",
    file_extension: str | None = None,
) -> STTProvider:
    """Select a provider based on mode + audio duration + style + file format.

    Args:
        stt_settings: Current STT configuration.
        audio_duration: Known duration in seconds, or ``None`` when unknown.
        style: ``"normal"`` (plain transcription) or ``"ai_prompt"`` (structured).
        file_extension: e.g. ``".wav"``, ``".webm"``. When a format isn't supported
            by the routed provider, we fall back to the other cloud provider.

    Returns:
        Cached or freshly-created :class:`STTProvider` instance.
    """
    if stt_settings.mode == ProviderMode.LOCAL:
        return _get_local(stt_settings)

    ext = file_extension.lower() if file_extension else None

    # ai_prompt always needs Gemini — it's the only provider that can structure.
    if style == "ai_prompt":
        return _get_gemini(stt_settings)

    # Normal style: route by duration.
    duration_short = (
        audio_duration is not None and audio_duration <= stt_settings.cloud_routing_threshold
    )

    if duration_short:
        if ext is None or ext in GROQ_SUPPORTED_FORMATS:
            return _get_groq(stt_settings)
        # Groq can't handle this container (e.g. .webm) — fall back to Gemini.
        return _get_gemini(stt_settings)

    # Long audio or unknown duration -> Gemini (safe default).
    return _get_gemini(stt_settings)


def get_stt_provider(stt_settings: STTSettings) -> STTProvider:
    """Legacy factory — returns the mode-level provider without routing hints.

    Prefer :func:`get_routed_provider` in pipeline code. Kept for backwards
    compatibility with ``/stt/transcribe``, status endpoints, and tests.
    """
    if stt_settings.mode == ProviderMode.LOCAL:
        return _get_local(stt_settings)
    return _get_gemini(stt_settings)


def is_model_loaded() -> bool:
    """Check if the local whisper model is currently loaded in memory."""
    from app.stt.local import LocalSTTProvider

    with _cache_lock:
        provider = _providers.get(LocalSTTProvider)
    if provider is None:
        return False
    return getattr(provider, "_model", None) is not None


def clear_cache() -> None:
    """Cleanup and invalidate every cached provider. Call on config change / shutdown."""
    with _cache_lock:
        for p in _providers.values():
            try:
                p.cleanup()
            except Exception:  # cleanup must never raise
                pass
        _providers.clear()

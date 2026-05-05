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
# Groq Whisper API: WAV/MP3/FLAC/OGG/M4A/MP4 (no webm/opus container).
# Gemini Native Audio: a superset including webm/opus/aac/etc.
GROQ_SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".mp4"}
)
GEMINI_SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac", ".m4a", ".mp4",
     ".aac", ".opus", ".wma", ".aiff", ".aif"}
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
    from app.stt.local import LocalSTTProvider
    return _get_or_create(LocalSTTProvider, stt_settings)


def get_routed_provider(
    stt_settings: STTSettings,
    audio_duration: float | None = None,
    style: str = "normal",
    file_extension: str | None = None,
) -> tuple[STTProvider, str | None]:
    """Select a provider based on engine pin + mode + audio duration + style + format.

    Args:
        stt_settings: Current STT configuration.
        audio_duration: Known duration in seconds, or ``None`` when unknown.
        style: ``"normal"`` (plain transcription) or ``"ai_prompt"`` (structured).
        file_extension: e.g. ``".wav"``, ``".webm"``. When a format isn't supported
            by the routed provider, we fall back to the other cloud provider.

    Returns:
        ``(provider, fallback_reason)`` — the cached/created :class:`STTProvider`
        plus an optional reason string when the *requested* engine had to be
        overridden (used by the UI to show "fell back to Gemini for ai_prompt").
    """
    if stt_settings.mode == ProviderMode.LOCAL:
        return _get_local(stt_settings), None

    ext = file_extension.lower() if file_extension else None
    engine = getattr(stt_settings, "engine", "auto")

    # --- Pinned engines --------------------------------------------------
    if engine == "gemini":
        return _get_gemini(stt_settings), None

    if engine == "groq":
        # Groq can't structure ai_prompt — Gemini fallback for that style.
        if style == "ai_prompt":
            return _get_gemini(stt_settings), "ai_prompt requires Gemini structuring"
        # Groq can't ingest .webm — Gemini fallback for that container.
        if ext is not None and ext not in GROQ_SUPPORTED_FORMATS:
            return _get_gemini(stt_settings), f"Groq doesn't support {ext}"
        return _get_groq(stt_settings), None

    # --- Auto (default) — duration+style heuristic ----------------------
    if style == "ai_prompt":
        return _get_gemini(stt_settings), None

    duration_short = (
        audio_duration is not None and audio_duration <= stt_settings.cloud_routing_threshold
    )

    if duration_short:
        if ext is None or ext in GROQ_SUPPORTED_FORMATS:
            return _get_groq(stt_settings), None
        # Groq can't handle this container (e.g. .webm) — fall back to Gemini.
        return _get_gemini(stt_settings), None

    # Long audio or unknown duration -> Gemini (safe default).
    return _get_gemini(stt_settings), None


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

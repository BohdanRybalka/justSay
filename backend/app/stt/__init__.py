"""STT module — Speech-to-Text processing.

This file is the **routing and cache** layer on top of the provider classes.
Pipeline code calls :func:`get_routed_provider` with the audio metadata
(mode, duration, style) and gets back the correct provider to use.
Other endpoints (status, /config) call :func:`get_provider`
which returns the mode-level provider without engine/duration heuristics.

Routing rules (see ``docs/plans/005-hybrid-stt-pipeline.md``):

======================  ========================================================
Mode / conditions        Provider
======================  ========================================================
LOCAL                    Platform-selected via
                         :func:`app.stt.local_factory.get_local_provider_class`:
                         :class:`~app.stt.local_mlx.MLXWhisperSTTProvider` on
                         macOS arm64, else :class:`~app.stt.local.LocalSTTProvider`
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
    "get_provider",
    "get_routed_provider",
    "get_local_load_error",
    "peek_local_provider",
    "clear_cache",
    "is_model_loaded",
    "is_local_provider",
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
    # Lazy import inside the function so the autouse conftest fixture can
    # monkeypatch `app.stt.local_factory.get_local_provider_class` without
    # an already-bound module-level reference defeating it.
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    return _get_or_create(cls, stt_settings)


def get_provider(mode: ProviderMode, stt_settings: STTSettings) -> STTProvider:
    """Mode-level provider lookup, no routing heuristics.

    For `/stt/local/load`, `/config`, status endpoints —
    callers that don't have audio duration / style context. Cloud mode always
    returns Gemini (engine pin and duration routing live in
    :func:`get_routed_provider`).
    """
    if mode == ProviderMode.LOCAL:
        return _get_local(stt_settings)
    return _get_gemini(stt_settings)


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


def get_local_load_error(stt_settings: STTSettings) -> str | None:
    """Return the most recent local-provider load failure, or None.

    Returns None when the local provider hasn't been instantiated yet — that's
    the same outcome a fresh process would observe before the first
    transcribe/load call. No error has occurred yet.
    """
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    with _cache_lock:
        provider = _providers.get(cls)
    if provider is None:
        return None
    return getattr(provider, "last_load_error", None)


def peek_local_provider() -> STTProvider | None:
    """Read-only peek at whichever provider is currently cached for the Local
    provider class, or None. Never creates an instance — unlike
    :func:`get_provider`/:func:`_get_local`.

    Used by :func:`app.stt.local_setup.ensure_local_ready` to check whether
    the provider it captured at the top of a prewarm attempt is still the
    one anyone would look up (a cache-identity check), independent of
    whatever ``stt_settings.mode`` currently reports — ``clear_cache()`` can
    evict a provider from the cache without the mode itself ever changing
    (spec 015, RED-1).
    """
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    with _cache_lock:
        return _providers.get(cls)


def is_model_loaded() -> bool:
    """Check if the local whisper model is currently loaded in memory."""
    from app.stt.local_factory import get_local_provider_class
    cls = get_local_provider_class()
    with _cache_lock:
        provider = _providers.get(cls)
    if provider is None:
        return False
    return getattr(provider, "is_loaded", False)


def is_local_provider(provider: STTProvider) -> bool:
    """Is ``provider`` local? Reads the ``is_local`` class attribute the
    provider itself declares (see :class:`app.stt.base.STTProvider` and
    ``docs/adr/018-provider-declared-locality.md``) -- no I/O, no import of
    ``local_factory``, no platform probe, no ``isinstance`` chain.

    ADR 018 supersedes the first implementation of this function, which
    asked the factory (``isinstance(provider, get_local_provider_class())``).
    That performed a full GPU probe (``probe_gpu()``, ~126 ms) to answer
    "is this local?" even for an obviously-Cloud provider, because resolving
    *which* local provider class applies to this platform is not a free
    lookup on Windows. Locality is a property of the object already held,
    not a fact about the host -- deriving it from the platform was the
    wrong instrument. Used by :func:`app.pipeline.service.process_audio` to
    gate the Spec 028 Item 2 readiness barrier onto local routes only.
    """
    return getattr(provider, "is_local", False)


def clear_cache() -> None:
    """Cleanup and invalidate every cached provider. Call on config change / shutdown."""
    with _cache_lock:
        for p in _providers.values():
            try:
                p.cleanup()
            except Exception:  # cleanup must never raise
                pass
        _providers.clear()

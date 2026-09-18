"""STT package — Speech-to-Text processing.

A re-export surface and nothing else. The provider classes live in their own
modules (:mod:`app.stt.cloud`, :mod:`app.stt.local`,
:mod:`app.stt.local_whisper_cpp`, :mod:`app.stt.groq_whisper`) and the routing
rules, the provider cache and the platform dispatch live in
:mod:`app.stt.routing`, which is also the address for this package's internals.
"""

from app.stt.base import STTProvider
from app.stt.config import STTSettings
from app.stt.routing import (
    GROQ_SUPPORTED_FORMATS,
    clear_cache,
    get_local_load_error,
    get_provider,
    get_routed_provider,
    is_local_provider,
    is_model_loaded,
    peek_local_provider,
)

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
]

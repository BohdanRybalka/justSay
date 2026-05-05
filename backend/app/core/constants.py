"""Shared backend constants.

Single source of truth for limits and identifiers used in more than one module.
Implementation-detail constants (e.g. cache TTLs) intentionally stay local to
their owning module.
"""

MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024  # 25 MB — Groq free-tier upper bound
MAX_TEXT_LENGTH: int = 100_000           # ~100 KB body for /llm/process

# All popular containers users may drop into upload endpoints. Per-provider
# whitelists (Groq vs Gemini) live next to the routing logic in app/stt/__init__.py.
ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac",
    ".m4a", ".mp4", ".aac", ".opus", ".wma", ".aiff", ".aif",
})

GROQ_TIMEOUT_SECONDS: float = 10.0

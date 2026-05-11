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

# Extension → MIME map sent to providers that need an explicit Content-Type
# (Gemini accepts `audio/wav`, `audio/mpeg`, etc.). Hardcoding `audio/wav`
# for everything silently produced wrong-but-tolerable Gemini results — see
# `docs/release-notes/v0.8.2.md`. Every key MUST also live in
# ``ALLOWED_AUDIO_EXTENSIONS`` — enforced by a unit test.
MIME_BY_AUDIO_EXTENSION: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",   # Opus payload typically rides inside an OggS container
    ".webm": "audio/webm",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".wma": "audio/x-ms-wma",
}

GROQ_TIMEOUT_SECONDS: float = 10.0

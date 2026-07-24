"""Shared backend constants.

Single source of truth for limits and identifiers used in more than one module.
Implementation-detail constants (e.g. cache TTLs) intentionally stay local to
their owning module.
"""

MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024

ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac",
    ".m4a", ".mp4", ".aac", ".opus", ".wma", ".aiff", ".aif",
})

MIME_BY_AUDIO_EXTENSION: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".wma": "audio/x-ms-wma",
}

GROQ_TIMEOUT_SECONDS: float = 10.0

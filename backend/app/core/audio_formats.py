"""Audio container formats — the extension/MIME table and the magic-bytes detector.

Deliberately free of ``fastapi``: the STT providers need
:func:`mime_for_extension` to set a Content-Type, and a provider must not
acquire a web-framework dependency to look up a string. The HTTP-facing
validator that turns these answers into a 400 lives in
``app.pipeline.upload_validation``.

``ALLOWED_AUDIO_EXTENSIONS`` is derived from the MIME table rather than
written out a second time, so an extension can never be accepted without a
MIME to serve it.
"""

from pathlib import Path

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

ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(MIME_BY_AUDIO_EXTENSION)

DETECTED_MIME_TO_EXTENSIONS: dict[str, frozenset[str]] = {
    "audio/wav": frozenset({".wav"}),
    "audio/mpeg": frozenset({".mp3"}),
    "audio/ogg": frozenset({".ogg", ".oga", ".opus"}),
    "audio/flac": frozenset({".flac"}),
    "audio/mp4": frozenset({".m4a", ".mp4"}),
    "audio/webm": frozenset({".webm"}),
    "audio/aiff": frozenset({".aiff", ".aif"}),
    "audio/x-ms-wma": frozenset({".wma"}),
}

TRUSTED_EXTENSIONS: frozenset[str] = frozenset({".aac"})

MIN_MAGIC_BYTES: int = 16


def detect_audio_mime(content: bytes) -> str | None:
    """Return a MIME type when the first 16 bytes match a known audio container.

    Returns None if the content has no recognised audio magic. Callers must
    still reject sub-16-byte payloads up front.
    """
    if len(content) < MIN_MAGIC_BYTES:
        return None

    if content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return "audio/wav"

    if content[:3] == b"ID3":
        return "audio/mpeg"
    if content[0] == 0xFF and content[1] in (0xFB, 0xF3, 0xE3, 0xF2):
        return "audio/mpeg"

    if content[:4] == b"OggS":
        return "audio/ogg"

    if content[:4] == b"fLaC":
        return "audio/flac"

    if content[4:8] == b"ftyp":
        return "audio/mp4"

    if content[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"

    if content[:4] == b"FORM" and content[8:12] in (b"AIFF", b"AIFC"):
        return "audio/aiff"

    if content[:4] == b"\x30\x26\xb2\x75":
        return "audio/x-ms-wma"

    return None


def mime_for_extension(filename: str | None) -> str:
    """Return the MIME for an audio file based on its extension only.

    Used by providers (e.g. Gemini) that need a Content-Type *after* upload
    validation has already accepted the file. Falls back to ``audio/wav`` for
    safety so a code path that forgets to validate still produces a payload
    Gemini can ingest (suboptimal but not broken).
    """
    ext = Path(filename).suffix.lower() if filename else ""
    return MIME_BY_AUDIO_EXTENSION.get(ext, "audio/wav")

"""Audio upload validation — magic-bytes detector + family check.

The upload endpoint (`/pipeline/process-file`)
used to trust the filename's extension alone. That accepted any payload
renamed to `.wav` — including executables, polyglots, or zero-byte files
that crash the STT provider with a noisy 500.

The validator here:

1. Rejects sub-16-byte payloads (no audio container fits in 16 bytes).
2. Rejects unknown extensions (already enforced upstream, defended again).
3. For containers we can recognise via magic bytes (WAV / MP3 / OGG /
   FLAC / M4A+MP4 / WebM / AIFF / WMA), the detected MIME must belong to
   the extension's family — `.wav` content under a `.mp3` filename is a
   400, even though the .mp3 path is allowed in general.
4. For extensions we genuinely cannot disambiguate via magic bytes
   (currently `.aac`: ADTS frame sync `0xFFF1`/`0xFFF9` collides with MP3
   frame sync), we trust the extension at face value rather than over-
   reject. Listed in ``_TRUSTED_EXTENSIONS`` so reviewers see the gap.
"""

from pathlib import Path

from fastapi import HTTPException

from app.core.constants import ALLOWED_AUDIO_EXTENSIONS, MIME_BY_AUDIO_EXTENSION


# Maps a detected MIME (from magic bytes) → the set of extensions that
# can legitimately appear with that container.
_DETECTED_MIME_TO_EXTS: dict[str, frozenset[str]] = {
    "audio/wav":      frozenset({".wav"}),
    "audio/mpeg":     frozenset({".mp3"}),
    "audio/ogg":      frozenset({".ogg", ".oga", ".opus"}),  # Opus payload inside OggS
    "audio/flac":     frozenset({".flac"}),
    "audio/mp4":      frozenset({".m4a", ".mp4"}),
    "audio/webm":     frozenset({".webm"}),
    "audio/aiff":     frozenset({".aiff", ".aif"}),
    "audio/x-ms-wma": frozenset({".wma"}),
}

# Extensions whose magic bytes we deliberately can't verify uniquely
# (ADTS frame sync conflicts with MP3 frame sync; the heuristic would
# produce false rejects). Listed so the gap is visible.
_TRUSTED_EXTENSIONS: frozenset[str] = frozenset({".aac"})

# Minimum bytes required to even attempt detection.
_MIN_MAGIC_BYTES: int = 16


def detect_audio_mime(content: bytes) -> str | None:
    """Return a MIME type when the first 16 bytes match a known audio container.

    Returns None if the content has no recognised audio magic. Callers must
    still reject sub-16-byte payloads up front via :func:`validate_audio_upload`.
    """
    if len(content) < _MIN_MAGIC_BYTES:
        return None

    # WAV (RIFF....WAVE)
    if content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return "audio/wav"

    # MP3 — ID3v2 tagged or raw frame sync.
    # 0xFFFB / 0xFFF3 / 0xFFE3 / 0xFFF2 — MPEG-1/2 Layer III sync.
    if content[:3] == b"ID3":
        return "audio/mpeg"
    if content[0] == 0xFF and content[1] in (0xFB, 0xF3, 0xE3, 0xF2):
        return "audio/mpeg"

    # OGG container (used for .ogg / .oga / .opus).
    if content[:4] == b"OggS":
        return "audio/ogg"

    # FLAC
    if content[:4] == b"fLaC":
        return "audio/flac"

    # ISO Base Media (MP4 / M4A) — `ftyp` box starts at offset 4.
    if content[4:8] == b"ftyp":
        return "audio/mp4"

    # Matroska / WebM (EBML header).
    if content[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"

    # AIFF / AIFC — FORM....AIFF or AIFC.
    if content[:4] == b"FORM" and content[8:12] in (b"AIFF", b"AIFC"):
        return "audio/aiff"

    # Windows Media Audio (ASF container).
    if content[:4] == b"\x30\x26\xb2\x75":
        return "audio/x-ms-wma"

    return None


def validate_audio_upload(content: bytes, filename: str | None) -> str:
    """Validate an uploaded audio payload and return its canonical MIME.

    Raises HTTPException(400) on:
        - empty / sub-16-byte payload
        - missing or disallowed extension
        - magic-bytes match a different container family than the extension
    """
    if len(content) < _MIN_MAGIC_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Audio file too small or empty (< 16 bytes)",
        )

    ext = Path(filename).suffix.lower() if filename else ""
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    expected_mime = MIME_BY_AUDIO_EXTENSION.get(ext)
    if expected_mime is None:
        # Should be unreachable — every extension in ALLOWED_AUDIO_EXTENSIONS
        # must be in MIME_BY_AUDIO_EXTENSION (covered by test). Defend anyway.
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    if ext in _TRUSTED_EXTENSIONS:
        # ADTS / raw AAC frames cannot be uniquely disambiguated from MP3 in
        # 16 bytes. Trust the extension; the STT provider validates further.
        return expected_mime

    detected = detect_audio_mime(content)
    if detected is None:
        # Magic bytes don't match any known audio container. With a non-trusted
        # extension this means the content is almost certainly not what it
        # claims to be.
        raise HTTPException(
            status_code=400,
            detail=(
                f"File content does not match a recognised audio container "
                f"for extension '{ext}'"
            ),
        )

    allowed_exts_for_mime = _DETECTED_MIME_TO_EXTS.get(detected, frozenset())
    if ext not in allowed_exts_for_mime:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File content (detected as {detected}) does not match the "
                f"declared extension '{ext}'"
            ),
        )

    return expected_mime


def mime_for_extension(filename: str | None) -> str:
    """Return the MIME for an audio file based on its extension only.

    Used by providers (e.g. Gemini) that need a Content-Type *after* upload
    validation has already accepted the file. Falls back to ``audio/wav`` for
    safety so a code path that forgets to validate still produces a payload
    Gemini can ingest (suboptimal but not broken).
    """
    ext = Path(filename).suffix.lower() if filename else ""
    return MIME_BY_AUDIO_EXTENSION.get(ext, "audio/wav")

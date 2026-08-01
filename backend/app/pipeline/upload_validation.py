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
   reject. Listed in ``TRUSTED_EXTENSIONS`` so reviewers see the gap.

The tables and the detector itself live in ``app.core.audio_formats``, which
carries no web-framework dependency so the STT providers can read the MIME
map without acquiring one.
"""

from pathlib import Path

from fastapi import HTTPException

from app.core.audio_formats import (
    ALLOWED_AUDIO_EXTENSIONS,
    DETECTED_MIME_TO_EXTENSIONS,
    MIME_BY_AUDIO_EXTENSION,
    MIN_MAGIC_BYTES,
    TRUSTED_EXTENSIONS,
    detect_audio_mime,
)


def validate_audio_upload(content: bytes, filename: str | None) -> str:
    """Validate an uploaded audio payload and return its canonical MIME.

    Raises HTTPException(400) on:
        - empty / sub-16-byte payload
        - missing or disallowed extension
        - magic-bytes match a different container family than the extension
    """
    if len(content) < MIN_MAGIC_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Audio file too small or empty (< 16 bytes)",
        )

    ext = Path(filename).suffix.lower() if filename else ""
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    expected_mime = MIME_BY_AUDIO_EXTENSION.get(ext)
    if expected_mime is None:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    if ext in TRUSTED_EXTENSIONS:
        return expected_mime

    detected = detect_audio_mime(content)
    if detected is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File content does not match a recognised audio container "
                f"for extension '{ext}'"
            ),
        )

    allowed_exts_for_mime = DETECTED_MIME_TO_EXTENSIONS.get(detected, frozenset())
    if ext not in allowed_exts_for_mime:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File content (detected as {detected}) does not match the "
                f"declared extension '{ext}'"
            ),
        )

    return expected_mime

"""Audio upload validation — magic-bytes detector + family check.

``validate_audio_upload`` refuses a sub-16-byte payload, an unknown extension,
and content whose magic bytes name a different container family than the
filename claims; ``TRUSTED_EXTENSIONS`` lists what magic bytes cannot
disambiguate. ``read_upload_with_limit`` is the 413 beside that 400. The
tables and the detector live in ``app.core.audio_formats``, which carries no
web-framework dependency so the STT providers can read the MIME map.
"""

from pathlib import Path

from fastapi import HTTPException, UploadFile

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

    Raises ``HTTPException(400)`` on an empty or sub-16-byte payload, a missing
    or disallowed extension, or magic bytes from a different container family.
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


async def read_upload_with_limit(file: UploadFile, max_size: int) -> bytes:
    """Stream-read an UploadFile in 64 KB chunks, raising HTTP 413 when exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    "File too large (max "
                    f"{round(max_size / (1024 * 1024), 1):g}MB)"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)

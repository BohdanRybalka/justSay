"""Audio container formats — the extension/MIME table, the magic-bytes detector, the decoder.

Deliberately free of ``fastapi``, so a provider can look up a Content-Type
without acquiring a web-framework dependency.

``ALLOWED_AUDIO_EXTENSIONS`` is derived from the MIME table rather than
written out a second time, so an extension can never be accepted without a
MIME to serve it.
"""

from pathlib import Path

from app.core.scratch import discard_scratch_file

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

UNREADABLE_HERE = "This kind of file can't be read on this computer. Cloud mode can transcribe it"

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

DECODE_BLOCK_FRAMES: int = 65536


class UndecodableAudioError(Exception):
    """soundfile cannot open this file (M4A, MP4, AAC, WMA, WebM and anything damaged)."""


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

    For a provider that needs a Content-Type after upload validation has accepted the file. An
    unknown or absent extension falls back to ``audio/wav`` rather than raising.
    """
    ext = Path(filename).suffix.lower() if filename else ""
    return MIME_BY_AUDIO_EXTENSION.get(ext, "audio/wav")


def decode_to_mono_wav(source: Path, target: Path, rate: int) -> None:
    """Write ``source`` to ``target`` as a 16-bit mono WAV at ``rate``, a block at a time.

    Raises ``UndecodableAudioError`` when soundfile cannot open ``source``; any later failure
    removes ``target`` and propagates.
    """
    import numpy as np
    import soundfile as sf
    import soxr

    try:
        reader = sf.SoundFile(str(source))
    except RuntimeError as exc:
        raise UndecodableAudioError(f"{source.suffix}: {exc}") from exc
    try:
        with reader, sf.SoundFile(
            str(target), "w", samplerate=rate, channels=1, format="WAV", subtype="PCM_16"
        ) as sink:
            stream = soxr.ResampleStream(reader.samplerate, rate, 1, dtype="float32")
            for block in reader.blocks(DECODE_BLOCK_FRAMES, dtype="float32", always_2d=True):
                sink.write(stream.resample_chunk(block.mean(axis=1), last=False))
            sink.write(stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
    except Exception:
        discard_scratch_file(target)
        raise

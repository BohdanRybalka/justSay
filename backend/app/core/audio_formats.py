"""Audio container formats — the extension/MIME table, the magic-bytes detector, the decoder.

Deliberately free of ``fastapi``, so the pipeline can decode audio without
acquiring a web-framework dependency.

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
    """Neither soundfile nor FFmpeg can open this file: it is damaged or not audio at all."""


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


def decode_to_mono_wav(source: Path, target: Path, rate: int) -> None:
    """Write ``source`` to ``target`` as a 16-bit mono WAV at ``rate``, a block at a time.

    soundfile reads what libsndfile knows; FFmpeg (PyAV) the rest, such as M4A, AAC and WebM.
    Raises ``UndecodableAudioError`` when neither can open ``source``; any later failure removes
    ``target`` and propagates.
    """
    import soundfile as sf

    try:
        reader = sf.SoundFile(str(source))
        write = _write_from_soundfile
    except RuntimeError:
        reader = _open_with_ffmpeg(source)
        write = _write_from_ffmpeg

    try:
        with reader, sf.SoundFile(
            str(target), "w", samplerate=rate, channels=1, format="WAV", subtype="PCM_16"
        ) as sink:
            write(reader, sink, rate)
    except Exception:
        discard_scratch_file(target)
        raise


def _write_from_soundfile(reader, sink, rate: int) -> None:
    import numpy as np
    import soxr

    stream = soxr.ResampleStream(reader.samplerate, rate, 1, dtype="float32")
    for block in reader.blocks(DECODE_BLOCK_FRAMES, dtype="float32", always_2d=True):
        sink.write(stream.resample_chunk(block.mean(axis=1), last=False))
    sink.write(stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True))


def _open_with_ffmpeg(source: Path):
    import av

    try:
        container = av.open(str(source))
    except av.FFmpegError as exc:
        raise UndecodableAudioError(f"{source.suffix}: {exc}") from exc
    if not container.streams.audio:
        container.close()
        raise UndecodableAudioError(f"{source.suffix}: no audio stream")
    return container


def _write_from_ffmpeg(container, sink, rate: int) -> None:
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            sink.write(resampled.to_ndarray().reshape(-1))
    for resampled in resampler.resample(None):
        sink.write(resampled.to_ndarray().reshape(-1))


def encode_aac_m4a(path: Path, samples, rate: int) -> None:
    """Write planar stereo float32 ``samples`` (shape 2 x n) to ``path`` as AAC in an M4A."""
    import av
    import numpy as np

    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("aac", rate=rate, layout="stereo")
        for start in range(0, samples.shape[1], 1024):
            block = np.ascontiguousarray(samples[:, start:start + 1024], dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
            frame.rate = rate
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def ffmpeg_selftest() -> tuple[bool, str]:
    """``--selftest-ffmpeg`` backend: an AAC M4A made here decodes to 16 kHz mono. Never raises."""
    import tempfile

    import numpy as np
    import soundfile as sf

    try:
        tone = 0.3 * np.sin(2.0 * np.pi * 220.0 * np.arange(44100) / 44100)
        with tempfile.TemporaryDirectory() as probe_dir:
            source, target = Path(probe_dir) / "probe.m4a", Path(probe_dir) / "probe.wav"
            encode_aac_m4a(source, np.vstack([tone, tone]), 44100)
            decode_to_mono_wav(source, target, 16000)
            info = sf.info(str(target))
    except Exception as e:
        return False, f"decoding an AAC M4A raised: {e}"
    if (info.samplerate, info.channels) != (16000, 1) or abs(info.duration - 1.0) > 0.1:
        return False, f"decoded to {info.samplerate} Hz x {info.channels}, {info.duration:.2f}s"
    return True, "ok"

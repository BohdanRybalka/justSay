import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

WAV_WRITE_CHUNK_FRAMES = 1 << 20


def write_wav_streaming(
    path: Path, audio: np.ndarray, sample_rate: int, channels: int, chunk_frames: int
) -> Path:
    """Write float samples in [-1, 1] as a 16-bit PCM WAV, `chunk_frames` at a time.

    Converts one chunk at a time, so a `np.memmap` source is never
    materialised whole. Returns `path`.
    """
    with open(path, "wb") as raw:
        with wave.open(raw, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for start in range(0, len(audio), chunk_frames):
                chunk = audio[start:start + chunk_frames]
                wf.writeframes((np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return path


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> Path:
    """Write float samples in [-1, 1] as a 16-bit PCM WAV and return `path`.

    For an array already in memory; `write_wav_streaming` is the same write
    for a source too large to convert in one step, and this delegates to it.
    """
    return write_wav_streaming(
        path, audio, sample_rate, channels, WAV_WRITE_CHUNK_FRAMES
    )


class AudioRecorder(ABC):
    """Contract: begin capturing, and report what is being captured.

    `stop()` is deliberately not declared: the two recorders answer it with
    different types, and nothing consumes them through this class.
    """

    @abstractmethod
    async def start(self) -> None:
        """Begin capturing audio."""

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """Whether recording is currently active."""

    @property
    @abstractmethod
    def duration_seconds(self) -> float:
        """Elapsed recording time in seconds. 0 if not recording."""

    @property
    @abstractmethod
    def level_db(self) -> float:
        """Current audio input level in dBFS. -inf if silent or not recording."""

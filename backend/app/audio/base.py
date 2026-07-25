import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> Path:
    """Write float samples in [-1, 1] as a 16-bit PCM WAV and return `path`.

    Shared by every `AudioRecorder`, because two implementations writing the
    same header and the same clip-and-scale by hand is how the dictation path
    and the meeting path would drift into producing subtly different files.
    `backend/tests/test_audio.py::test_microphone_wav_bytes_unchanged` pins the
    dictation output byte-for-byte against the value it had before this
    function existed.
    """
    audio_16bit = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_16bit.tobytes())
    return path


class AudioRecorder(ABC):
    """Contract: start recording → stop recording → get audio file."""

    @abstractmethod
    async def start(self) -> None:
        """Begin capturing audio."""

    @abstractmethod
    async def stop(self) -> Path:
        """Stop capturing and return path to the recorded WAV file."""

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

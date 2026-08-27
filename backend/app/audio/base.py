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

    The file is opened here rather than by path inside `wave.open`, which
    would build a `Wave_write` around the open and, when the open fails --
    a full disk, a `temp_dir` that vanished -- leave a half-constructed
    object whose `__del__` raises `AttributeError: _file` into the
    unraisable hook, printing a second, misleading traceback next to the
    real error. Opening first means a failed open raises and nothing is
    constructed. The bytes written are unchanged either way.
    """
    audio_16bit = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with open(path, "wb") as raw:
        with wave.open(raw, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_16bit.tobytes())
    return path


class AudioRecorder(ABC):
    """Contract: begin capturing, and report what is being captured.

    `stop()` is deliberately not declared here. Both recorders have one and
    they answer with different things — a `Path` for dictation, a
    `MeetingRecording` for a meeting, whose duration and truncation flag
    belong to the capture rather than to the recorder that produced it.
    Declaring one return type would advertise a signature the other subclass
    breaks, and nothing consumes the two through this class: `app.audio.router`
    names both concrete types. The pin is
    `test_meeting_recorder.py::test_the_recorder_abc_declares_only_what_both_recorders_honour`.
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

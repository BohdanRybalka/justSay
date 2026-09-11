import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

WAV_WRITE_CHUNK_FRAMES = 1 << 20


def write_wav_streaming(
    path: Path, audio: np.ndarray, sample_rate: int, channels: int, chunk_frames: int
) -> Path:
    """Write float samples in [-1, 1] as a 16-bit PCM WAV, `chunk_frames` at a time.

    The meeting path hands this a `np.memmap` over a mix file that can be
    hundreds of megabytes, and converting it to int16 in one step would
    materialise both the whole int16 array and the whole `tobytes()` copy of
    it. The conversion is elementwise, so the bytes written do not depend on
    where the chunk boundaries fall — `write_wav` delegates here and
    `test_audio.py::test_microphone_wav_bytes_unchanged` still pins the
    dictation output byte-for-byte.

    The file is opened here rather than by path inside `wave.open`, which
    would build a `Wave_write` around the open and, when the open fails --
    a full disk, a `temp_dir` that vanished -- leave a half-constructed
    object whose `__del__` raises `AttributeError: _file` into the
    unraisable hook, printing a second, misleading traceback next to the
    real error. Opening first means a failed open raises and nothing is
    constructed.
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

    Shared by every `AudioRecorder`, because two implementations writing the
    same header and the same clip-and-scale by hand is how the dictation path
    and the meeting path would drift into producing subtly different files.
    `backend/tests/test_audio.py::test_microphone_wav_bytes_unchanged` pins the
    dictation output byte-for-byte against the value it had before this
    function existed.

    One call, for an array already in memory; `write_wav_streaming` is the
    same write for a source too large to convert in one step, and this
    delegates to it so the two cannot drift.
    """
    return write_wav_streaming(
        path, audio, sample_rate, channels, WAV_WRITE_CHUNK_FRAMES
    )


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

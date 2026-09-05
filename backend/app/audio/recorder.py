"""Microphone recorder using sounddevice."""

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import sounddevice as sd

from app.audio.analysis import rms_dbfs
from app.audio.base import AudioRecorder, write_wav
from app.audio.config import AudioSettings

log = logging.getLogger(__name__)


class MicrophoneRecorder(AudioRecorder):
    """Records audio from the default microphone input."""

    def __init__(self, settings: AudioSettings):
        self._settings = settings
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._start_time: float = 0.0
        self._final_duration: float = 0.0
        self._current_level: float = float("-inf")

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        """Called by sounddevice from a separate thread for each audio block."""
        with self._lock:
            if not self._recording:
                return
            self._frames.append(indata.copy())
            self._current_level = rms_dbfs(indata)

    async def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._current_level = float("-inf")
            self._recording = True

        try:
            self._settings.temp_dir.mkdir(parents=True, exist_ok=True)
            self._stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception:
            self.cleanup()
            raise

        self._start_time = time.monotonic()

    async def stop(self) -> Path:
        with self._lock:
            if not self._recording or self._stream is None:
                raise RuntimeError("Not recording")
            self._final_duration = time.monotonic() - self._start_time
            self._recording = False

        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

        with self._lock:
            frames = self._frames
            self._frames = []

        if not frames:
            raise RuntimeError("No audio data captured")

        filename = f"rec_{uuid.uuid4().hex[:12]}.wav"
        output_path = self._settings.temp_dir / filename

        return await asyncio.to_thread(self._concatenate_and_write, frames, output_path)

    def _concatenate_and_write(self, frames: list[np.ndarray], output_path: Path) -> Path:
        """The dictation counterpart of the meeting recorder's off-loop write.

        Smaller -- a dictation clip is seconds, not a 45-minute call -- but the
        same shape, reached from the same `async def`, so a long recording
        stalls every other endpoint for the length of the write.
        """
        return write_wav(
            output_path,
            np.concatenate(frames, axis=0),
            self._settings.sample_rate,
            self._settings.channels,
        )

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def duration_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def level_db(self) -> float:
        with self._lock:
            return self._current_level

    @property
    def last_duration_seconds(self) -> float:
        """Duration of the most recently completed recording, or 0.0 if never stopped."""
        return self._final_duration

    def cleanup(self) -> None:
        """Release the audio stream if one is open. Safe to call any time,
        including when never started. Discards buffered frames without writing
        a WAV — call on app shutdown, or to roll a failed start() back to a
        stopped state, but never as a substitute for stop()."""
        with self._lock:
            stream = self._stream
            self._stream = None
            self._recording = False
            self._frames = []
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.warning(
                    "Stopping and closing the dictation microphone stream failed",
                    exc_info=True,
                )

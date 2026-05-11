"""Microphone recorder using sounddevice."""

import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from app.audio.base import AudioRecorder
from app.audio.config import AudioSettings


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
            rms = np.sqrt(np.mean(indata**2))
            self._current_level = 20 * np.log10(max(rms, 1e-10))

    async def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._current_level = float("-inf")
            self._recording = True

        self._settings.temp_dir.mkdir(parents=True, exist_ok=True)

        self._stream = sd.InputStream(
            samplerate=self._settings.sample_rate,
            channels=self._settings.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._start_time = time.monotonic()

    async def stop(self) -> Path:
        with self._lock:
            if not self._recording or self._stream is None:
                raise RuntimeError("Not recording")
            self._final_duration = min(
                time.monotonic() - self._start_time,
                float(self._settings.max_duration_seconds),
            )
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

        audio_data = np.concatenate(frames, axis=0)
        filename = f"rec_{uuid.uuid4().hex[:12]}.wav"
        output_path = self._settings.temp_dir / filename

        audio_16bit = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(self._settings.channels)
            wf.setsampwidth(2)
            wf.setframerate(self._settings.sample_rate)
            wf.writeframes(audio_16bit.tobytes())

        return output_path

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def duration_seconds(self) -> float:
        if not self._recording:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self._settings.max_duration_seconds:
            return self._settings.max_duration_seconds
        return elapsed

    @property
    def level_db(self) -> float:
        with self._lock:
            return self._current_level

    @property
    def last_duration_seconds(self) -> float:
        """Duration of the most recently completed recording, or 0.0 if never stopped."""
        return self._final_duration

    @property
    def max_duration_exceeded(self) -> bool:
        if not self._recording:
            return False
        return (time.monotonic() - self._start_time) >= self._settings.max_duration_seconds

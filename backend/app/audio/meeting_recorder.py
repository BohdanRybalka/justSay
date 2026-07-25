"""Meeting recorder — microphone plus system audio, mixed into one WAV.

Implements the same `AudioRecorder` contract as `MicrophoneRecorder`, so the
file it produces enters the pipeline through the same door and nothing
downstream has to know a meeting was recorded. `MicrophoneRecorder` itself is
deliberately untouched by this module: the dictation path must not move.

Phase 1 ships with no UI — see
docs/adr/039-meeting-recording-ships-dark-until-macos-lands.md.
"""

from __future__ import annotations

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
from app.audio.system_source import (
    SystemAudioSource,
    SystemAudioUnavailableError,
    create_system_audio_source,
)
from app.audio.timeline import CapturedBlock, mix_and_normalize, place_on_timeline, to_mono

log = logging.getLogger(__name__)


class MeetingRecorder(AudioRecorder):
    """Captures the microphone and the system render endpoint at once."""

    def __init__(self, settings: AudioSettings):
        self._settings = settings
        self._lock = threading.Lock()
        self._microphone_blocks: list[CapturedBlock] = []
        self._system_blocks: list[CapturedBlock] = []
        self._raw_bytes = 0
        self._truncated = False
        self._stream: sd.InputStream | None = None
        self._system_source: SystemAudioSource | None = None
        self._recording = False
        self._start_time: float = 0.0
        self._stop_time: float = 0.0
        self._final_duration: float = 0.0
        self._current_level: float = float("-inf")
        self._system_level: float = float("-inf")
        self._endpoint_name: str | None = None

    def _store(self, blocks: list[CapturedBlock], arrival: float, mono: np.ndarray) -> bool:
        """Append under the lock unless the raw-store cap is already reached.

        Returns whether the block was kept. Both stores share one budget, and
        once it is exhausted capture keeps running but stops accumulating, so
        `stop()` still returns a valid WAV of everything up to that point.
        """
        with self._lock:
            if not self._recording:
                return False
            if self._raw_bytes >= self._settings.meeting_max_raw_bytes:
                if not self._truncated:
                    self._truncated = True
                    log.warning(
                        "Meeting recording hit the %d-byte raw buffer cap — "
                        "further audio is dropped",
                        self._settings.meeting_max_raw_bytes,
                    )
                return False
            blocks.append(CapturedBlock(arrival=arrival, samples=mono))
            self._raw_bytes += mono.nbytes
            return True

    def _microphone_callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        arrival = time.monotonic()
        mono = to_mono(indata)
        if self._store(self._microphone_blocks, arrival, mono):
            with self._lock:
                self._current_level = rms_dbfs(mono)

    def _system_callback(self, arrival: float, mono: np.ndarray) -> None:
        if self._store(self._system_blocks, arrival, mono):
            with self._lock:
                self._system_level = rms_dbfs(mono)

    async def start(self) -> None:
        with self._lock:
            if self._recording:
                return

        source = create_system_audio_source(self._settings)
        if source is None:
            raise SystemAudioUnavailableError(
                "System audio capture is not available on this platform — "
                "meeting recording requires Windows or macOS"
            )

        with self._lock:
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0
            self._truncated = False
            self._current_level = float("-inf")
            self._system_level = float("-inf")
            self._endpoint_name = source.endpoint_name
            self._recording = True

        self._system_source = source
        self._start_time = time.monotonic()

        try:
            self._settings.temp_dir.mkdir(parents=True, exist_ok=True)
            source.start(self._system_callback)
            self._stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="float32",
                blocksize=self._settings.meeting_block_frames,
                callback=self._microphone_callback,
            )
            self._stream.start()
        except Exception:
            self._release()
            with self._lock:
                self._recording = False
            raise

    async def stop(self) -> Path:
        with self._lock:
            if not self._recording:
                raise RuntimeError("Not recording")
            self._stop_time = time.monotonic()
            self._final_duration = self._stop_time - self._start_time
            self._recording = False

        system_rate = (
            self._system_source.native_sample_rate
            if self._system_source is not None
            else self._settings.sample_rate
        )
        self._release()

        with self._lock:
            microphone_blocks = self._microphone_blocks
            system_blocks = self._system_blocks
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0

        if not microphone_blocks and not system_blocks:
            raise RuntimeError("No audio data captured")

        return self._write_wav(
            self._assemble(microphone_blocks, system_blocks, system_rate)
        )

    def _assemble(
        self,
        microphone_blocks: list[CapturedBlock],
        system_blocks: list[CapturedBlock],
        system_rate: int,
    ) -> np.ndarray:
        target_rate = self._settings.sample_rate
        common = {
            "target_rate": target_rate,
            "recording_start": self._start_time,
            "recording_stop": self._stop_time,
            "gap_tolerance_blocks": self._settings.meeting_gap_tolerance_blocks,
            "rate_tolerance": self._settings.meeting_rate_tolerance,
        }
        microphone = place_on_timeline(
            microphone_blocks, nominal_rate=target_rate, **common
        )
        system = place_on_timeline(system_blocks, nominal_rate=system_rate, **common)
        return mix_and_normalize(microphone, system)

    def _write_wav(self, audio_data: np.ndarray) -> Path:
        filename = f"meeting_{uuid.uuid4().hex[:12]}.wav"
        output_path = self._settings.temp_dir / filename

        return write_wav(output_path, audio_data, self._settings.sample_rate, channels=1)

    def _release(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.warning("Closing the meeting microphone stream failed", exc_info=True)

        source = self._system_source
        self._system_source = None
        if source is not None:
            try:
                source.stop()
            except Exception:
                log.warning("Closing the system audio source failed", exc_info=True)

        with self._lock:
            self._endpoint_name = None

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
        """The microphone level, identical in meaning to the dictation path's."""
        with self._lock:
            return self._current_level

    @property
    def system_level_db(self) -> float:
        """The system half's level, so a silent far side is visible while it happens.

        A meeting recording that captured only the microphone is indistinguishable
        from a working one until someone plays the file back; this is what makes
        the difference visible at the machine.
        """
        with self._lock:
            return self._system_level

    @property
    def system_endpoint(self) -> str | None:
        """The output being captured, or None when nothing is being captured."""
        with self._lock:
            return self._endpoint_name

    @property
    def last_duration_seconds(self) -> float:
        """Duration of the most recently completed recording, or 0.0 if never stopped."""
        return self._final_duration

    @property
    def truncated(self) -> bool:
        """Whether the raw-store cap was reached and audio was dropped."""
        return self._truncated

    def cleanup(self) -> None:
        """Release both capture streams without writing a WAV.

        For app shutdown, not as a substitute for stop(). Safe to call any
        time, including when never started.
        """
        with self._lock:
            self._recording = False
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0
        self._release()

"""Meeting recorder — microphone plus system audio, mixed into one WAV.

Implements the same `AudioRecorder` contract as `MicrophoneRecorder`, so the
file it produces enters the pipeline through the same door and nothing
downstream has to know a meeting was recorded. `MicrophoneRecorder` itself is
deliberately untouched by this module: the dictation path must not move.

System audio arrives through `app.audio.system_source`, which is the only
place that knows which platform it is running on — see
docs/adr/037-system-audio-capture-is-a-per-platform-source.md.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import sounddevice as sd

from app.audio.analysis import rms_dbfs, to_mono
from app.audio.base import AudioRecorder, write_wav
from app.audio.config import AudioSettings
from app.audio.system_source import (
    SystemAudioSource,
    SystemAudioUnavailableError,
    create_system_audio_source,
)
from app.audio.timeline import CapturedBlock, mix_and_normalize, place_on_timeline

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
        """Claim the recording slot, then open both captures off the event loop.

        The check and the flag share one lock hold, the way
        `MicrophoneRecorder.start()` does it: releasing between them let two
        concurrent starts both open devices, the second overwriting the first's
        stream and source with nothing left holding a reference to close them.
        """
        with self._lock:
            if self._recording:
                return
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0
            self._truncated = False
            self._current_level = float("-inf")
            self._system_level = float("-inf")
            self._endpoint_name = None
            self._recording = True

        self._start_time = time.monotonic()

        try:
            await asyncio.to_thread(self._open_capture)
        except BaseException:
            self._release()
            with self._lock:
                self._recording = False
            raise

    def _open_capture(self) -> None:
        """Device open, endpoint enumeration and the helper handshake, off the loop.

        Windows enumerates render endpoints through COM and macOS waits up to
        five seconds for the tap helper's header. Run inline this blocked every
        other endpoint for the whole start -- including `/health` and the
        meeting status the widget polls twice a second, which is the moment the
        user is waiting to see the recording begin. Same reasoning as
        `_assemble_and_write`, which JS-81 moved off the loop for `stop()`.
        """
        source = create_system_audio_source(self._settings)
        if source is None:
            raise SystemAudioUnavailableError(
                "System audio capture is not available on this platform — "
                "meeting recording requires Windows or macOS"
            )

        self._system_source = source
        with self._lock:
            self._endpoint_name = source.endpoint_name

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

    async def stop(self) -> Path:
        with self._lock:
            if not self._recording:
                raise RuntimeError("Not recording")
            self._stop_time = time.monotonic()
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

        return await asyncio.to_thread(
            self._assemble_and_write, microphone_blocks, system_blocks, system_rate
        )

    def _assemble_and_write(
        self,
        microphone_blocks: list[CapturedBlock],
        system_blocks: list[CapturedBlock],
        system_rate: int,
    ) -> Path:
        """Two resamples, a mix and a synchronous wave write, off the event loop.

        A 45-minute call is tens of millions of samples and ~86 MB to disk.
        Run inline this blocked every other endpoint for the whole write --
        including `/health` and the meeting status the widget polls twice a
        second, which is the moment the user is waiting on their transcript.
        """
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

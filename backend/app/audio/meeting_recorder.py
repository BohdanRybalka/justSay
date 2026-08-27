"""Meeting recorder — microphone plus system audio, mixed into one WAV.

Implements the same `AudioRecorder` contract as `MicrophoneRecorder`, so the
file it produces enters the pipeline through the same door and nothing
downstream has to know a meeting was recorded. `MicrophoneRecorder` itself is
deliberately untouched by this module: the dictation path must not move.

System audio arrives through `app.audio.system_source`, which is the only
place that knows which platform it is running on — see
docs/adr/037-system-audio-capture-is-a-per-platform-source.md.

Every call that creates, starts, stops or destroys a device handle runs on
the single worker of `self._devices`, never on the event loop — see
docs/adr/048-one-thread-owns-every-meeting-device-handle.md.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
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


class MeetingState(str, Enum):
    """The four states a meeting recorder can be in.

    `STARTING` exists because opening the two devices takes seconds on both
    platforms, and during that window the devices are spoken for while no
    audio is arriving yet. Collapsing that window into a single boolean is
    what let a stop or a cleanup reach handles the opening thread had not
    created yet.

    `STOPPING` is the mirror window: the capture has ended but both handles
    are still open while the release runs off the loop, which on macOS
    budgets up to 1.5 s. Reporting the recorder as idle for that window let a
    second start claim it and wipe the finished meeting's buffers.
    """

    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"


class MeetingCaptureAbortedError(RuntimeError):
    """No meeting file can be produced, and it is the caller's situation.

    Covers a start that was cancelled by a competing stop, cleanup or task
    cancellation, a stop with nothing recording, and a stop that captured no
    audio. The router maps it to 409; subclassing `RuntimeError` keeps every
    caller that only distinguishes "it failed" working unchanged.
    """


def _release_devices(
    stream: sd.InputStream | None, source: SystemAudioSource | None
) -> None:
    """Close whichever of the two handles exists, never raising.

    Must run on the thread that created them: PortAudio's WASAPI host API
    initialises COM on the calling thread, so an instance opened on one
    thread has to be terminated on that same thread.
    """
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.warning("Closing the meeting microphone stream failed", exc_info=True)

    if source is not None:
        try:
            source.stop()
        except Exception:
            log.warning("Closing the system audio source failed", exc_info=True)


class MeetingRecorder(AudioRecorder):
    """Captures the microphone and the system render endpoint at once."""

    def __init__(self, settings: AudioSettings):
        self._settings = settings
        self._lock = threading.Lock()
        self._devices = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="meeting-devices"
        )
        self._microphone_blocks: list[CapturedBlock] = []
        self._system_blocks: list[CapturedBlock] = []
        self._raw_bytes = 0
        self._truncated = False
        self._stream: sd.InputStream | None = None
        self._system_source: SystemAudioSource | None = None
        self._state = MeetingState.IDLE
        self._generation = 0
        self._start_time: float = 0.0
        self._current_level: float = float("-inf")
        self._system_level: float = float("-inf")
        self._endpoint_name: str | None = None

    def _store(self, blocks: list[CapturedBlock], arrival: float, mono: np.ndarray) -> bool:
        """Append under the lock unless the raw-store cap is already reached.

        Returns whether the block was kept. Both stores share one budget, and
        once it is exhausted capture keeps running but stops accumulating, so
        `stop()` still returns a valid WAV of everything up to that point.

        A block arriving before the capture is published is dropped: it
        belongs to a start that may still be abandoned.
        """
        with self._lock:
            if self._state is not MeetingState.RECORDING:
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
        """Claim the recorder in one lock hold, then open both devices off the loop.

        The claim has to be atomic with the check, because the `await` below
        is a yield point: two callers could otherwise both pass the check and
        each open a pair of devices. The same hold stamps a generation on the
        claim, which is what tells the open whether the `STARTING` it finds
        when it finishes is still its own.
        """
        with self._lock:
            if self._state is not MeetingState.IDLE:
                return
            self._state = MeetingState.STARTING
            self._generation += 1
            generation = self._generation
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0
            self._truncated = False
            self._current_level = float("-inf")
            self._system_level = float("-inf")
            self._endpoint_name = None

        loop = asyncio.get_running_loop()
        try:
            published = await loop.run_in_executor(
                self._devices, self._open_capture, generation
            )
        except BaseException:
            self._abandon_start()
            raise

        if not published:
            raise MeetingCaptureAbortedError(
                "The meeting recording was cancelled while it was opening its devices"
            )

    def _open_capture(self, generation: int) -> bool:
        """Open both devices on the owner thread and decide whether to publish them.

        Runs entirely on `self._devices`. Returns whether the handles were
        published; when they were not, they have already been released here,
        on the thread that created them, so nothing else ever sees them.
        """
        source = create_system_audio_source(self._settings)
        if source is None:
            raise SystemAudioUnavailableError(
                "System audio capture is not available on this platform — "
                "meeting recording requires Windows or macOS"
            )

        stream: sd.InputStream | None = None
        try:
            self._settings.temp_dir.mkdir(parents=True, exist_ok=True)
            source.start(self._system_callback)
            stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="float32",
                blocksize=self._settings.meeting_block_frames,
                callback=self._microphone_callback,
            )
            stream.start()
        except Exception:
            _release_devices(stream, source)
            raise

        return self._publish_capture(source, stream, generation)

    def _publish_capture(
        self, source: SystemAudioSource, stream: sd.InputStream, generation: int
    ) -> bool:
        """Hand the open handles to the recorder, unless the start was abandoned.

        `_start_time` is sampled here rather than before the open, so a slow
        device open contributes no leading silence to the written WAV.

        Both halves of the condition are load-bearing and neither implies the
        other: the state check catches "nobody is starting", the generation
        check catches "somebody else is starting" — an abandoned start whose
        slot a later start has already re-claimed, whose publish would leave
        a stale endpoint and a ticking duration on a recorder with no devices.
        """
        with self._lock:
            publishing = (
                self._state is MeetingState.STARTING
                and self._generation == generation
            )
            if publishing:
                self._system_source = source
                self._stream = stream
                self._endpoint_name = source.endpoint_name
                self._start_time = time.monotonic()
                self._state = MeetingState.RECORDING

        if not publishing:
            _release_devices(stream, source)
        return publishing

    def _abandon_start(self) -> None:
        """Drop back to idle and release anything the open may already have published.

        The generation is deliberately not bumped. The state check is what
        covers abandonment, and a release guarded by the generation would
        skip the live handles of a start that published and was then
        cancelled — a leak of both devices.
        """
        with self._lock:
            self._state = MeetingState.IDLE
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0
            self._endpoint_name = None
            stream = self._stream
            source = self._system_source
            self._stream = None
            self._system_source = None
        self._devices.submit(_release_devices, stream, source)

    async def stop(self) -> Path:
        """End the capture and take everything the WAV needs in one lock hold.

        No mutable recorder state is read after the first `await`: the
        release is a yield point long enough for a second start to claim the
        recorder, and a harvest on the far side of it would find that start's
        empty buffers and its `_start_time` instead of this meeting's.
        """
        system_rate = self._settings.sample_rate
        stream: sd.InputStream | None = None
        source: SystemAudioSource | None = None
        microphone_blocks: list[CapturedBlock] = []
        system_blocks: list[CapturedBlock] = []
        recording_start = 0.0
        recording_stop = 0.0
        with self._lock:
            state = self._state
            if state is MeetingState.RECORDING:
                recording_start = self._start_time
                recording_stop = time.monotonic()
                self._state = MeetingState.STOPPING
                self._endpoint_name = None
                if self._system_source is not None:
                    system_rate = self._system_source.native_sample_rate
                stream = self._stream
                source = self._system_source
                self._stream = None
                self._system_source = None
                microphone_blocks = self._microphone_blocks
                system_blocks = self._system_blocks
                self._microphone_blocks = []
                self._system_blocks = []
                self._raw_bytes = 0

        if state is MeetingState.STARTING:
            self._abandon_start()
            raise MeetingCaptureAbortedError(
                "The meeting recording was still opening its devices — the start "
                "was cancelled and nothing was captured"
            )
        if state is not MeetingState.RECORDING:
            raise MeetingCaptureAbortedError("Not recording")

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._devices, _release_devices, stream, source)
        finally:
            with self._lock:
                if self._state is MeetingState.STOPPING:
                    self._state = MeetingState.IDLE

        if not microphone_blocks and not system_blocks:
            raise MeetingCaptureAbortedError("No audio data captured")

        return await asyncio.to_thread(
            self._assemble_and_write,
            microphone_blocks,
            system_blocks,
            system_rate,
            recording_start,
            recording_stop,
        )

    def _assemble_and_write(
        self,
        microphone_blocks: list[CapturedBlock],
        system_blocks: list[CapturedBlock],
        system_rate: int,
        recording_start: float,
        recording_stop: float,
    ) -> Path:
        """Two resamples, a mix and a synchronous wave write, off the event loop.

        A 45-minute call is tens of millions of samples and ~86 MB to disk.
        Run inline this blocked every other endpoint for the whole write --
        including `/health` and the meeting status the widget polls twice a
        second, which is the moment the user is waiting on their transcript.
        """
        return self._write_wav(
            self._assemble(
                microphone_blocks,
                system_blocks,
                system_rate,
                recording_start,
                recording_stop,
            )
        )

    def _assemble(
        self,
        microphone_blocks: list[CapturedBlock],
        system_blocks: list[CapturedBlock],
        system_rate: int,
        recording_start: float,
        recording_stop: float,
    ) -> np.ndarray:
        target_rate = self._settings.sample_rate
        common = {
            "target_rate": target_rate,
            "recording_start": recording_start,
            "recording_stop": recording_stop,
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

    @property
    def is_recording(self) -> bool:
        """Whether audio is arriving right now."""
        with self._lock:
            return self._state is MeetingState.RECORDING

    @property
    def is_busy(self) -> bool:
        """Whether the microphone and the render endpoint are spoken for.

        Differs from `is_recording` for the whole length of the device open,
        and that window is exactly what the mutual-exclusion guards have to
        cover: the devices are claimed before any audio arrives.
        """
        with self._lock:
            return self._state is not MeetingState.IDLE

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            if self._state is not MeetingState.RECORDING:
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
        time, including when never started. Returns without waiting for the
        release to finish, the shape `LocalWhisperCppSTTProvider.cleanup()`
        already uses for a teardown reachable from the event-loop thread: the
        macOS release alone budgets up to 1.5 s of terminate, kill and reader
        join. A recorder that never opened anything submits nothing, so a
        session with no meeting in it spawns no worker at shutdown.
        """
        with self._lock:
            idle = self._state is MeetingState.IDLE
            stream = self._stream
            source = self._system_source
            self._stream = None
            self._system_source = None
            self._state = MeetingState.IDLE
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0
            self._endpoint_name = None

        if idle and stream is None and source is None:
            return
        self._devices.submit(_release_devices, stream, source)

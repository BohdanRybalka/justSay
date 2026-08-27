"""Meeting recorder — microphone plus system audio, mixed into one WAV.

Implements the same `AudioRecorder` contract as `MicrophoneRecorder`, so the
file it produces enters the pipeline through the same door and nothing
downstream has to know a meeting was recorded. `MicrophoneRecorder` itself is
deliberately untouched by this module: the dictation path must not move.

System audio arrives through `app.audio.system_source`, which is the only
place that knows which platform it is running on — see
docs/adr/037-system-audio-capture-is-a-per-platform-source.md.

Every call that creates, starts, stops or destroys a device handle runs on
the single worker of `self._devices`, and so does every lifecycle transition
that describes one: the event loop submits commands and awaits results, and
reads the state only in order to report it — see
docs/adr/048-one-thread-owns-every-meeting-device-handle.md.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import NamedTuple, TypeVar

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

_T = TypeVar("_T")


def _drop_outcome(pending: asyncio.Future[object]) -> None:
    """Retrieve a finished future's exception so it is not reported unhandled.

    A future nobody reads the exception of reaches the event loop's
    exception handler as `Future exception was never retrieved`, with a
    traceback, at ERROR level. Retrieving it here discards it instead.
    """
    if not pending.cancelled():
        pending.exception()


class MeetingState(str, Enum):
    """The four states a meeting recorder can be in.

    `STARTING` exists because opening the two devices takes seconds on both
    platforms, and during that window the devices are spoken for while no
    audio is arriving yet. Collapsing that window into a single boolean is
    what let a stop or a cleanup reach handles the opening thread had not
    created yet.

    `STOPPING` is the mirror window: the capture has ended but both handles
    are still open while the release runs, which on macOS budgets up to
    1.5 s. Reporting the recorder as idle for that window let a second start
    claim it and wipe the finished meeting's buffers.
    """

    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"


class MeetingCaptureAbortedError(RuntimeError):
    """No meeting file can be produced, and it is the caller's situation.

    Covers a start that found the recorder already spoken for, a stop with
    nothing recording, and a stop that captured no audio. The router maps it
    to 409; subclassing `RuntimeError` keeps every caller that only
    distinguishes "it failed" working unchanged.
    """


class _CapturedMeeting(NamedTuple):
    """Everything the WAV needs, taken out of the recorder in one lock hold."""

    microphone_blocks: list[CapturedBlock]
    system_blocks: list[CapturedBlock]
    system_rate: int
    recording_start: float
    recording_stop: float


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
        self._devices_in_flight = 0
        self._tokens = itertools.count(1)
        self._session_token: int | None = None
        self._microphone_blocks: list[CapturedBlock] = []
        self._system_blocks: list[CapturedBlock] = []
        self._raw_bytes = 0
        self._truncated = False
        self._stream: sd.InputStream | None = None
        self._system_source: SystemAudioSource | None = None
        self._state = MeetingState.IDLE
        self._start_time: float = 0.0
        self._current_level: float = float("-inf")
        self._system_level: float = float("-inf")
        self._endpoint_name: str | None = None

    def _transition(self, state: MeetingState) -> None:
        """Move the lifecycle to `state` — the only site that assigns it.

        Called only from callables running on `self._devices`, so the thread
        that owns the two handles is the thread that writes the state
        describing them.
        """
        with self._lock:
            self._state = state

    def _submit_on_devices(self, fn: Callable[..., _T], *args: object) -> Future[_T]:
        """Queue `fn` on the owner thread and count it as in flight at once.

        The increment happens here, on the calling thread, before the submit:
        a command that has been queued but not yet picked up leaves the state
        untouched, and `is_busy` has to cover that window too or a dictation
        start opens a second stream on a microphone this recorder still holds.
        """
        with self._lock:
            self._devices_in_flight += 1
        try:
            return self._devices.submit(self._counted, fn, *args)
        except BaseException:
            with self._lock:
                self._devices_in_flight -= 1
            raise

    async def _run_on_devices(self, fn: Callable[..., _T], *args: object) -> _T:
        """Queue `fn` on the owner thread and await its result on the loop.

        The await is shielded: cancelling it detaches the awaiter from the
        answer, and the command still runs on the owner thread.
        """
        pending = asyncio.wrap_future(self._submit_on_devices(fn, *args))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            pending.add_done_callback(_drop_outcome)
            raise

    def _counted(self, fn: Callable[..., _T], *args: object) -> _T:
        try:
            return fn(*args)
        finally:
            with self._lock:
                self._devices_in_flight -= 1

    def _store(self, blocks: list[CapturedBlock], arrival: float, mono: np.ndarray) -> bool:
        """Append under the lock unless the raw-store cap is already reached.

        Returns whether the block was kept. Both stores share one budget, and
        once it is exhausted capture keeps running but stops accumulating, so
        `stop()` still returns a valid WAV of everything up to that point.

        A block arriving outside `RECORDING` is dropped: it belongs to a
        capture that is not published yet, or to one that has already ended.
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
        """Send the owner thread a start command and wait for its answer.

        Reads no recorder state at all: whether this start may proceed is
        answered by `_begin_capture`, on the only thread that can change the
        answer. A failure — a device error or a cancelled await — sends an
        abandonment behind it, carrying the token that says which capture it
        is entitled to tear down.
        """
        token = next(self._tokens)
        try:
            await self._run_on_devices(self._begin_capture, token)
        except BaseException:
            self._submit_on_devices(self._abandon_capture, token)
            raise

    def _begin_capture(self, token: int) -> None:
        """Open both devices and publish them, in one callable on the owner thread.

        There is no publish check: nothing can happen between the open and
        the publish, because they are one callable on the only thread that
        writes the state. `_start_time` is sampled at the publish rather than
        before the open, so a slow device open contributes no leading silence
        to the written WAV.
        """
        with self._lock:
            if self._state is not MeetingState.IDLE:
                raise MeetingCaptureAbortedError(
                    "The meeting recorder is busy — a previous recording is still "
                    "starting, recording or releasing its devices"
                )
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0
            self._truncated = False
            self._current_level = float("-inf")
            self._system_level = float("-inf")
            self._endpoint_name = None
        self._transition(MeetingState.STARTING)

        source: SystemAudioSource | None = None
        stream: sd.InputStream | None = None
        try:
            source = create_system_audio_source(self._settings)
            if source is None:
                raise SystemAudioUnavailableError(
                    "System audio capture is not available on this platform — "
                    "meeting recording requires Windows or macOS"
                )
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
        except BaseException:
            _release_devices(stream, source)
            self._transition(MeetingState.IDLE)
            raise

        with self._lock:
            self._system_source = source
            self._stream = stream
            self._endpoint_name = source.endpoint_name
            self._start_time = time.monotonic()
            self._session_token = token
        self._transition(MeetingState.RECORDING)

    async def stop(self) -> Path:
        """End the capture on the owner thread, then write the WAV off it.

        The assemble stays on `asyncio.to_thread` rather than moving to the
        owner thread: a 45-minute call is ~86 MB of resampling, mixing and
        wave writing, and on the owner thread that would block every device
        command queued behind it for the whole write.
        """
        captured = await self._run_on_devices(self._end_capture)
        return await asyncio.to_thread(
            self._assemble_and_write,
            captured.microphone_blocks,
            captured.system_blocks,
            captured.system_rate,
            captured.recording_start,
            captured.recording_stop,
        )

    def _end_capture(self) -> _CapturedMeeting:
        """Harvest, release and return to idle, in one callable on the owner thread.

        The return to `IDLE` is in a `finally` because the loop-side task
        awaiting this may be cancelled — FastAPI cancels the endpoint task
        when a client disconnects — and the recorder must not be left
        reporting itself busy with both handles already closed.
        """
        with self._lock:
            if self._state is not MeetingState.RECORDING:
                raise MeetingCaptureAbortedError("Not recording")
        self._transition(MeetingState.STOPPING)

        try:
            with self._lock:
                system_rate = self._settings.sample_rate
                if self._system_source is not None:
                    system_rate = self._system_source.native_sample_rate
                captured = _CapturedMeeting(
                    microphone_blocks=self._microphone_blocks,
                    system_blocks=self._system_blocks,
                    system_rate=system_rate,
                    recording_start=self._start_time,
                    recording_stop=time.monotonic(),
                )
                stream = self._stream
                source = self._system_source
                self._stream = None
                self._system_source = None
                self._microphone_blocks = []
                self._system_blocks = []
                self._raw_bytes = 0
                self._endpoint_name = None
                self._session_token = None
            _release_devices(stream, source)
        finally:
            self._transition(MeetingState.IDLE)

        if not captured.microphone_blocks and not captured.system_blocks:
            raise MeetingCaptureAbortedError("No audio data captured")
        return captured

    def _abandon_capture(self, token: int) -> None:
        """Tear down the capture `token` names, and only that one.

        Runs behind the start it abandons, in submission order. A start that
        failed before publishing owns no session, and a start whose slot a
        later start has since taken does not match — either way this returns
        without touching a recording that is not its own.
        """
        with self._lock:
            if self._session_token != token:
                return
        self._discard_capture()

    def _discard_capture(self) -> None:
        """Release both handles and return to idle, writing no WAV."""
        with self._lock:
            stream = self._stream
            source = self._system_source
            self._stream = None
            self._system_source = None
            self._microphone_blocks = []
            self._system_blocks = []
            self._raw_bytes = 0
            self._endpoint_name = None
            self._session_token = None
        _release_devices(stream, source)
        self._transition(MeetingState.IDLE)

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

        Counts the device commands in flight as well as the lifecycle state,
        because a command that has been submitted but not yet picked up
        leaves the state untouched. Its only failure direction is refusing
        something it could have allowed.
        """
        with self._lock:
            return self._state is not MeetingState.IDLE or self._devices_in_flight > 0

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
        join. A recorder with nothing in flight and nothing open submits
        nothing, so a session with no meeting in it spawns no worker at
        shutdown.
        """
        with self._lock:
            idle = self._state is MeetingState.IDLE and self._devices_in_flight == 0
            nothing_open = self._stream is None and self._system_source is None

        if idle and nothing_open:
            return
        self._submit_on_devices(self._discard_capture)

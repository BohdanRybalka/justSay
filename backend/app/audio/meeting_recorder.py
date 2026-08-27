"""Meeting recorder — microphone plus system audio, mixed into one WAV.

Writes the same 16 kHz mono 16-bit WAV `MicrophoneRecorder` does, so the file
enters the pipeline through the same door and nothing downstream has to know a
meeting was recorded. The `AudioRecorder` contract it shares covers starting
and reporting, not stopping: `stop()` here answers with a `MeetingRecording`
rather than the dictation path's bare `Path`, which is why the base class
declares no `stop()` at all — see `app.audio.base`. `MicrophoneRecorder`
itself is deliberately untouched by this module: the dictation path must not
move.

System audio arrives through `app.audio.system_source`, which is the only
place that knows which platform it is running on — see
docs/adr/037-system-audio-capture-is-a-per-platform-source.md.

Every call that creates, starts, stops or destroys a device handle runs on
the single worker of `self._devices`, and so does every lifecycle transition
that describes one: the event loop submits commands and awaits results, and
reads the state only in order to report it. The written file is the
recorder's obligation too, so `self._writer` carries it on a second worker
that the owner thread hands the capture to before it returns — see
docs/adr/048-one-thread-owns-every-meeting-device-handle.md.
"""

from __future__ import annotations

import asyncio
import functools
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
    platforms, and during that window the devices are spoken for while the
    second of them is not open yet. Collapsing that window into a single
    boolean is what let a stop or a cleanup reach handles the opening thread
    had not created yet. It is not a window in which nothing is captured:
    the system source is started first and its audio is kept from that
    moment, which is why `_start_time` rather than the state is what says
    whether a recording is under way.

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

    Covers a start that found the recorder already spoken for and a stop with
    nothing recording. The router maps it to 409; subclassing `RuntimeError`
    keeps every caller that only distinguishes "it failed" working unchanged.
    """


class MeetingCaptureEmptyError(MeetingCaptureAbortedError):
    """A meeting ran, both capture paths delivered nothing, and there is no file.

    Separate from its base class because the two outcomes are different
    outcomes rather than different wordings: the router answers 410 for this
    one and 409 for a stop that found nothing recording, and the widget picks
    its message from the status rather than from the prose.
    """


class MeetingWriteFailedError(RuntimeError):
    """The capture ended, both devices were released, and no file was written.

    Deliberately outside the `MeetingCaptureAbortedError` hierarchy: those two
    say a meeting never ran or captured nothing, and this one says a meeting
    ran and its audio was lost on the way to disk — a full disk, a `temp_dir`
    that vanished, a resample that failed. The router answers 507 for it, and
    that status is the whole point of the class: by the time the write is
    submitted `_end_capture` has already released both handles and returned
    the recorder to `IDLE`, so the widget must take its indicator down. A bare
    500 is indistinguishable from an unreachable backend, which may still be
    recording, and left the indicator lit after the meeting had ended.
    """


MEETING_BUSY_DETAIL = (
    "The meeting recorder is busy — a previous recording is still "
    "starting, recording or releasing its devices"
)


class _CapturedMeeting(NamedTuple):
    """Everything the WAV needs, taken out of the recorder in one lock hold."""

    microphone_blocks: list[CapturedBlock]
    system_blocks: list[CapturedBlock]
    system_rate: int
    recording_start: float
    recording_stop: float
    truncated: bool


class MeetingRecording(NamedTuple):
    """A written meeting file and the facts describing that same capture.

    The duration and the truncation flag travel with the path because they
    are properties of the capture, not of the recorder: the recorder stops
    being busy the moment the harvest returns, so a second meeting can start
    and finish while this file is still being written, and anything stored on
    the recorder would by then describe that other meeting instead.
    """

    path: Path
    duration_seconds: float
    truncated: bool


def _release_devices(
    stream: sd.InputStream | None, source: SystemAudioSource | None
) -> None:
    """Close whichever of the two handles exists, never raising.

    Must run on the thread that created them: PortAudio's WASAPI host API
    initialises COM on the calling thread, so an instance opened on one
    thread has to be terminated on that same thread.

    `stream.stop()` and `stream.close()` get a `try` each because the caller
    has already dropped its reference and `sounddevice._StreamBase` has no
    finalizer: a `stop()` that raises — the unplugged-headset case — would
    otherwise skip the `close()` and hold that PortAudio stream for the life
    of the process. Their failures are logged separately so the log says
    which of the two calls failed.
    """
    if stream is not None:
        try:
            stream.stop()
        except Exception:
            log.warning("Stopping the meeting microphone stream failed", exc_info=True)
        try:
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
        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="meeting-writer"
        )
        self._tokens = itertools.count(1)
        self._session_token: int | None = None
        self._microphone_blocks: list[CapturedBlock] = []
        self._system_blocks: list[CapturedBlock] = []
        self._raw_bytes = 0
        self._truncated = False
        self._stream: sd.InputStream | None = None
        self._system_source: SystemAudioSource | None = None
        self._state = MeetingState.IDLE
        self._start_time: float | None = None
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

        A submit onto the owner thread retired by `cleanup()` raises
        `MeetingCaptureAbortedError` rather than the executor's bare
        `RuntimeError`, so a start or a stop that lands in the shutdown
        window is answered with the 409 both endpoints already produce
        instead of a 500.
        """
        with self._lock:
            self._devices_in_flight += 1
        try:
            return self._devices.submit(self._counted, fn, *args)
        except RuntimeError as e:
            with self._lock:
                self._devices_in_flight -= 1
            raise MeetingCaptureAbortedError("The meeting recorder has shut down") from e
        except BaseException:
            with self._lock:
                self._devices_in_flight -= 1
            raise

    async def _detachable_result(self, pending: Future[_T]) -> _T:
        """Await work already submitted to a worker, detaching on cancellation.

        The await is shielded: cancelling it detaches the awaiter from the
        answer, and the submitted work still runs on its worker. One
        implementation, because both the device command and the file write
        are obligations the caller may stop waiting for but may not withdraw.
        """
        wrapped = asyncio.wrap_future(pending)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            wrapped.add_done_callback(_drop_outcome)
            raise

    async def _run_on_devices(self, fn: Callable[..., _T], *args: object) -> _T:
        """Queue `fn` on the owner thread and await its result on the loop."""
        return await self._detachable_result(self._submit_on_devices(fn, *args))

    def _counted(self, fn: Callable[..., _T], *args: object) -> _T:
        try:
            return fn(*args)
        finally:
            with self._lock:
                self._devices_in_flight -= 1

    def _store(
        self, token: int, blocks: list[CapturedBlock], arrival: float, mono: np.ndarray
    ) -> bool:
        """Append under the lock unless the raw-store cap is already reached.

        Returns whether the block was kept. Both stores share one budget, and
        once it is exhausted capture keeps running but stops accumulating, so
        `stop()` still returns a valid WAV of everything up to that point.

        A block is kept only for the session that registered the callback,
        and only while that session is capturing.
        """
        with self._lock:
            if self._session_token != token:
                return False
            if self._state not in (MeetingState.STARTING, MeetingState.RECORDING):
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
        self,
        token: int,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Keep the block, then publish its level if the session still exists.

        The keep and the level write are two separate lock holds, and a stop
        can complete in the gap between them, so the token is checked again
        in the second one. Without that check a callback preempted mid-way
        republishes an ended meeting's level after `_end_capture` cleared it.
        """
        arrival = time.monotonic()
        mono = to_mono(indata)
        if self._store(token, self._microphone_blocks, arrival, mono):
            with self._lock:
                if self._session_token == token:
                    self._current_level = rms_dbfs(mono)

    def _system_callback(self, token: int, arrival: float, mono: np.ndarray) -> None:
        """The far side's half of `_microphone_callback`, with the same re-check.

        The level write is a second lock hold here too, so a stop landing in
        the gap must not be followed by the ended meeting's far-side level.
        """
        if self._store(token, self._system_blocks, arrival, mono):
            with self._lock:
                if self._session_token == token:
                    self._system_level = rms_dbfs(mono)

    async def start(self) -> None:
        """Send the owner thread a start command and wait for its answer.

        Reads no recorder state at all: whether this start may proceed is
        answered by `_begin_capture`, on the only thread that can change the
        answer. A failure — a device error or a cancelled await — sends an
        abandonment behind it, carrying the token that says which capture it
        is entitled to tear down.

        The failure the caller sees is always the original one, cancellation
        included: a `cleanup()` that retired the owner thread first makes the
        abandonment unsendable, and that is logged rather than raised.
        """
        token = next(self._tokens)
        try:
            await self._run_on_devices(self._begin_capture, token)
        except BaseException:
            try:
                self._submit_on_devices(self._abandon_capture, token)
            except RuntimeError:
                log.warning(
                    "The meeting recorder was retired before the abandoned start "
                    "could be torn down"
                )
            raise

    def _begin_capture(self, token: int) -> None:
        """Open both devices and publish them, in one callable on the owner thread.

        There is no publish check: nothing can happen between the open and
        the publish, because they are one callable on the only thread that
        writes the state. The token is claimed in the first lock hold, so the
        two callbacks are admitted from the moment their devices are live.

        `capture_start` is read immediately before the system source starts,
        which is the earliest instant a far-side block can arrive, so a block
        delivered while `start()` is still returning lands at a non-negative
        offset instead of being trimmed by `place_on_timeline`. It is
        published the moment that call returns, before the microphone opens,
        so the microphone open contributes no leading silence and the
        recorder reports itself recording for the rest of that open rather
        than only once both devices are up.
        """
        with self._lock:
            if self._state is not MeetingState.IDLE:
                raise MeetingCaptureAbortedError(MEETING_BUSY_DETAIL)
            self._session_token = token
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
            capture_start = time.monotonic()
            source.start(functools.partial(self._system_callback, token))
            with self._lock:
                self._start_time = capture_start
                self._endpoint_name = source.endpoint_name
            stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="float32",
                blocksize=self._settings.meeting_block_frames,
                callback=functools.partial(self._microphone_callback, token),
            )
            stream.start()
        except BaseException:
            _release_devices(stream, source)
            with self._lock:
                self._microphone_blocks = []
                self._system_blocks = []
                self._raw_bytes = 0
                self._session_token = None
                self._current_level = float("-inf")
                self._system_level = float("-inf")
                self._start_time = None
                self._endpoint_name = None
            self._transition(MeetingState.IDLE)
            raise

        with self._lock:
            self._system_source = source
            self._stream = stream
        self._transition(MeetingState.RECORDING)

    async def stop(self) -> MeetingRecording:
        """End the capture on the owner thread and await the file it writes.

        Answers with a `MeetingRecording` rather than the bare path the
        dictation recorder returns: the duration and the truncation flag
        belong to the capture that produced the file and cannot be read off
        the recorder afterwards, which by then may be serving a later meeting.

        Both awaits are detachable and neither of them owns the work behind
        it: once `_end_capture` has returned, the write is already submitted,
        so a disconnect from that point on costs the answer and never the
        recording, and the file's path reaches the log without it.

        A cancellation that lands on the first await is different: the harvest
        may not have run yet, and when it does it can still find both block
        lists empty and produce no file at all.
        """
        try:
            writing = await self._run_on_devices(self._end_capture)
            return await self._detachable_result(writing)
        except asyncio.CancelledError:
            log.warning(
                "The meeting stop request was abandoned — if the capture produced "
                "a file, its path is in the log"
            )
            raise

    def _end_capture(self) -> Future[MeetingRecording]:
        """Harvest, release, submit the write and return its future.

        The return to `IDLE` is in a `finally` because the loop-side task
        awaiting this may be cancelled — FastAPI cancels the endpoint task
        when a client disconnects — and the recorder must not be left
        reporting itself busy with both handles already closed. Submitting
        the write here, as the last act on the owner thread, is what puts the
        recording out of that cancellation's reach: the `_CapturedMeeting`
        never crosses back to the loop.

        The truncation flag is harvested in the same lock hold as the blocks
        and travels inside the `_CapturedMeeting`, because it describes this
        capture while the live copy belongs to whichever meeting starts next.

        Both level meters are cleared with the rest of the capture's identity,
        the same way the abandoned-start path clears them, so a status read
        between a stop and the next start cannot report the ended meeting's
        levels next to `system_endpoint: null`.
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
                    truncated=self._truncated,
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
                self._start_time = None
                self._current_level = float("-inf")
                self._system_level = float("-inf")
            _release_devices(stream, source)
        finally:
            self._transition(MeetingState.IDLE)

        if not captured.microphone_blocks and not captured.system_blocks:
            raise MeetingCaptureEmptyError("No audio data captured")
        return self._writer.submit(self._write_captured_meeting, captured)

    def _write_captured_meeting(self, captured: _CapturedMeeting) -> MeetingRecording:
        """Write the harvested capture and describe the file it produced.

        The log line is how a meeting recording is found: nothing reads the
        stop response's filename, so the record of where the audio went has
        to be made by the party that owns the write rather than by the one
        that asked for it, which may already be gone.

        The duration and the truncation flag come back with the path so the
        answer describes this capture whatever the recorder is doing by the
        time the write lands.
        """
        try:
            output_path = self._assemble_and_write(
                captured.microphone_blocks,
                captured.system_blocks,
                captured.system_rate,
                captured.recording_start,
                captured.recording_stop,
            )
        except Exception as e:
            log.error("Writing the meeting recording failed", exc_info=True)
            raise MeetingWriteFailedError(str(e) or type(e).__name__) from e
        log.info("Meeting recording written to %s", output_path)
        return MeetingRecording(
            path=output_path,
            duration_seconds=captured.recording_stop - captured.recording_start,
            truncated=captured.truncated,
        )

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
        """Release both handles and return to idle, writing no WAV.

        Leaves the recorder exactly as a start that failed on its devices
        does: the same fields, including both level meters, so a status read
        after an abandoned start cannot report the discarded session's
        levels next to `system_endpoint: null`.
        """
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
            self._start_time = None
            self._current_level = float("-inf")
            self._system_level = float("-inf")
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
        """Whether audio is arriving right now.

        True from the instant the system source starts, which is inside
        `STARTING` and seconds before the microphone stream is up: far-side
        audio is kept from that moment, so this is what the widget's
        indicator must follow. `_start_time` carries both this answer and
        `duration_seconds`, so a status that reports a live meeting always
        reports the elapsed time of that same meeting.
        """
        with self._lock:
            return self._start_time is not None

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
            if self._start_time is None:
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
        """The output being captured, or None when nothing is being captured.

        Published in the same lock hold as the clock `is_recording` reads and
        cleared in the same one, so a status that reports a live meeting can
        always name what that meeting is capturing.
        """
        with self._lock:
            return self._endpoint_name

    def cleanup(self) -> None:
        """Send the owner thread one last command, then retire it.

        For app shutdown, not as a substitute for stop(). Returns without
        waiting for the release to finish, the shape
        `LocalWhisperCppSTTProvider.cleanup()` already uses for a teardown
        reachable from the event-loop thread: the macOS release alone budgets
        up to 1.5 s of terminate, kill and reader join.

        Terminal for recording — a `start()` or `stop()` submitted after it
        raises `MeetingCaptureAbortedError` — but idempotent in itself: a
        second call finds the owner thread already retired and returns.
        """
        try:
            self._submit_on_devices(self._shutdown_capture)
        except MeetingCaptureAbortedError:
            return
        self._devices.shutdown(wait=False)

    def _shutdown_capture(self) -> None:
        """Discard the capture, then retire the writer, on the owner thread.

        The writer is retired here rather than from the event loop because a
        `_end_capture` still sitting in the device queue has not submitted
        its write yet; behind the discard, the queue's own order guarantees
        it already has.
        """
        self._discard_capture()
        self._writer.shutdown(wait=False)

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
import queue
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import NamedTuple, TypeVar

import numpy as np
import sounddevice as sd

from app.audio.analysis import rms_dbfs, to_mono
from app.audio.base import AudioRecorder, write_wav_streaming
from app.audio.config import AudioSettings
from app.audio.meeting_spool import (
    MeetingSpool,
    assembly_reserve_bytes,
    close_memmap,
    free_bytes,
)
from app.audio.system_source import (
    SystemAudioSource,
    SystemAudioUnavailableError,
    create_system_audio_source,
)
from app.audio.timeline import normalize_in_place, place_on_timeline

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


class CaptureIncident(str, Enum):
    """Something ended or degraded a capture, named so the widget can say it.

    Exactly the tokens the code can produce, and no others: each one has a
    site that observes the event, and `src/contracts.ts` carries the same set
    pinned by `backend/tests/test_cross_language_contracts.py`. The first
    incident of a capture wins — a later one does not overwrite it — because
    what the user needs is the reason their recording went wrong, and the
    first thing to go wrong is usually the cause of the rest.

    A meeting with an incident is still a meeting: capture keeps running,
    the file is still written, and the token says what is missing from it.
    """

    MICROPHONE_STALLED = "microphone_stalled"
    SYSTEM_AUDIO_ENDED = "system_audio_ended"
    STORAGE_FAILED = "storage_failed"
    STORAGE_LOW = "storage_low"
    STORAGE_BACKLOG = "storage_backlog"


def microphone_has_stalled(last_arrival: float | None, now: float, tolerance: float) -> bool:
    """Whether the microphone has gone quiet for longer than `tolerance`.

    Derived from the last arrival rather than watched by a background task,
    so nothing new touches a device handle off the owner thread — ADR 048's
    invariant is untouched. `last_arrival` is `None` only when no capture is
    running, which is not a stall.
    """
    return last_arrival is not None and now - last_arrival > tolerance


MICROPHONE_SOURCE = "microphone"

SYSTEM_SOURCE = "system"

_SPILL_SENTINEL = object()


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
    """Everything the WAV needs, taken out of the recorder in one lock hold.

    The two spools are files on disk rather than lists of blocks, so what
    crosses to the writer thread is a pair of handles whatever the length of
    the meeting. Discarding them is the writer's obligation from here on.
    """

    microphone_spool: MeetingSpool
    system_spool: MeetingSpool
    system_rate: int
    recording_start: float
    recording_stop: float
    incident: CaptureIncident | None


class MeetingRecording(NamedTuple):
    """A written meeting file and the facts describing that same capture.

    The duration and the incident travel with the path because they
    are properties of the capture, not of the recorder: the recorder stops
    being busy the moment the harvest returns, so a second meeting can start
    and finish while this file is still being written, and anything stored on
    the recorder would by then describe that other meeting instead.
    """

    path: Path
    duration_seconds: float
    incident: CaptureIncident | None


class MeetingStatusSnapshot(NamedTuple):
    """The six reported facts as they stood in one lock hold.

    Exists because reading the five properties one after another does not
    describe one moment: each takes its own lock hold, and the owner thread
    is free to finish `_end_capture` between any two of them, which produced
    `{is_recording: true, duration_seconds: 0.0, system_endpoint: null}` —
    the combination the harvest's single write hold exists to make
    impossible. The properties stay for the callers that want one fact; a
    status that reports several of them at once takes this instead.
    """

    is_recording: bool
    duration_seconds: float
    level_db: float
    system_endpoint: str | None
    system_level_db: float
    capture_incident: CaptureIncident | None


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
        self._spill = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="meeting-spill"
        )
        self._tokens = itertools.count(1)
        self._session_token: int | None = None
        self._spill_queue: queue.Queue | None = None
        self._spill_drain: Future[None] | None = None
        self._microphone_spool: MeetingSpool | None = None
        self._system_spool: MeetingSpool | None = None
        self._last_microphone_arrival: float | None = None
        self._incident: CaptureIncident | None = None
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

    def _store(self, token: int, source: str, arrival: float, mono: np.ndarray) -> bool:
        """Hand the block to the spill thread and return, doing no I/O here.

        Returns whether the block was queued. The device callbacks reach this
        on a thread PortAudio calls with a deadline, so what happens here is a
        bounded `put_nowait` and nothing else: a write to a file a slow disk
        has stalled would be answered with substituted zeros and an underflow
        flag, which is the failure this queue exists to avoid.

        A full queue is an incident rather than an exception — the block is
        lost and the user is told — because raising at the device would take
        the stream down instead of the seconds it could not keep up with.

        A block is kept only for the session that registered the callback,
        and only while that session is capturing.

        The put happens under the same lock hold as the state check, and not
        after it: `_finish_spill` drops the queue under that lock before it
        enqueues the sentinel, so a callback that passed the check cannot be
        preempted and then land a block behind the sentinel, into a queue no
        worker is draining any more. `put_nowait` never blocks, so the hold
        stays as short as the check it joins, and the incident is recorded
        after the lock is released because `_note_incident` takes it too.
        """
        full = False
        with self._lock:
            if self._session_token != token:
                return False
            if self._state not in (MeetingState.STARTING, MeetingState.RECORDING):
                return False
            work = self._spill_queue
            if work is None:
                return False
            try:
                work.put_nowait((source, arrival, mono))
            except queue.Full:
                full = True
        if full:
            self._note_incident(
                CaptureIncident.STORAGE_BACKLOG,
                "the spill thread could not keep up, so some audio was dropped",
            )
            return False
        return True

    def _note_incident(self, incident: CaptureIncident, detail: str) -> bool:
        """Record the first incident of this capture and log it once.

        Returns whether this call was the one that recorded it, so a site
        reached repeatedly — a status read deriving a stall, a callback
        raising the same flag every block — logs one line per capture rather
        than one per call.
        """
        with self._lock:
            first = self._incident is None
            if first:
                self._incident = incident
        if not first:
            return False
        if incident is CaptureIncident.STORAGE_FAILED:
            log.error("Meeting capture incident %s: %s", incident.value, detail)
        else:
            log.warning("Meeting capture incident %s: %s", incident.value, detail)
        return True

    def _drain_spill(
        self, work: queue.Queue, spools: dict[str, MeetingSpool], started_at: float
    ) -> None:
        """Write every queued block to its spool until the sentinel arrives.

        Stops accumulating — rather than stopping — on a storage failure or a
        free-space floor, because the audio already on disk is a recording the
        user can still have and the callbacks must keep finding a drained
        queue whatever happened to the disk.

        The free-space check runs every `meeting_free_space_check_blocks`
        appends rather than every append: `disk_usage` is a syscall, and at
        the default that is one every 10.9 s of 48 kHz audio.
        """
        appended = 0
        accumulating = True
        while True:
            item = work.get()
            if item is _SPILL_SENTINEL:
                return
            if not accumulating:
                continue
            source, arrival, mono = item
            try:
                if appended % self._settings.meeting_free_space_check_blocks == 0:
                    elapsed = time.monotonic() - started_at
                    reserve = assembly_reserve_bytes(elapsed, self._settings.sample_rate)
                    if free_bytes(self._settings.temp_dir) < reserve:
                        accumulating = False
                        self._note_incident(
                            CaptureIncident.STORAGE_LOW,
                            f"assembling this recording needs {reserve} free bytes and "
                            f"the disk holding the recording has fewer, so capture "
                            f"stopped accumulating",
                        )
                        continue
                spools[source].append(arrival, mono)
            except OSError as e:
                accumulating = False
                self._note_incident(CaptureIncident.STORAGE_FAILED, str(e))
                continue
            appended += 1

    def _finish_spill(self) -> None:
        """Drain everything already queued, then let the spill worker return.

        Called on the owner thread once the state forbids further puts, so
        the sentinel is the last item the queue can receive and every block
        ahead of it is written before the spools are read.
        """
        with self._lock:
            work = self._spill_queue
            drain = self._spill_drain
            self._spill_queue = None
            self._spill_drain = None
        if work is not None:
            work.put(_SPILL_SENTINEL)
        if drain is not None:
            try:
                drain.result()
            except Exception:
                log.warning("The meeting spill thread ended badly", exc_info=True)

    def _system_failed(self, token: int, reason: str) -> None:
        """The system source says it has stopped delivering audio.

        The meeting keeps recording the microphone: half a call is what the
        user has, and taking the capture down would replace it with nothing.
        """
        with self._lock:
            if self._session_token != token:
                return
        self._note_incident(CaptureIncident.SYSTEM_AUDIO_ENDED, reason)

    def _microphone_callback(
        self,
        token: int,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Keep the block, then publish its level if the session is still live.

        The keep and the level write are two separate lock holds, and a stop
        can complete in the gap between them, so the session is re-checked in
        the second one. Without that check a callback preempted mid-way
        republishes an ended meeting's level after `_end_capture` cleared it.

        The level is published whether or not the block was kept. `_store`
        answers `False` for a block a full queue or a stopped session refused,
        and gating the meter on that answer froze both readings at their last
        accepted value, which is the opposite of what `system_level_db` exists
        for. The re-check repeats `_store`'s own session token rather than its
        return value, so a block belonging to a session that has already ended
        still publishes nothing.

        `status` is read rather than ignored: a non-zero `sd.CallbackFlags`
        is PortAudio saying it substituted zeros for audio it could not
        deliver, and a microphone that has stopped producing looks exactly
        like a quiet room without it.
        """
        arrival = time.monotonic()
        mono = to_mono(indata)
        if status:
            self._note_incident(
                CaptureIncident.MICROPHONE_STALLED,
                f"PortAudio reported {status} on the meeting microphone stream",
            )
        self._store(token, MICROPHONE_SOURCE, arrival, mono)
        with self._lock:
            if self._session_token == token:
                self._current_level = rms_dbfs(mono)
                self._last_microphone_arrival = arrival

    def _system_callback(self, token: int, arrival: float, mono: np.ndarray) -> None:
        """The far side's half of `_microphone_callback`, with the same re-check.

        The level write is a second lock hold here too, so a stop landing in
        the gap must not be followed by the ended meeting's far-side level,
        and it is reached past a refused store for the same reason.
        """
        self._store(token, SYSTEM_SOURCE, arrival, mono)
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

    def _forget_capture(self) -> None:
        """Return every field describing a capture to its no-meeting value.

        Called with `_lock` held, by all three sites that end a capture: a
        start that failed on its devices, an abandoned start and a harvest.
        They cleared the same eight fields in three separate copies, and a
        copy that ended up clearing seven of them is how a recorder that
        held no handle went on answering `is_recording: true` with a ticking
        duration and a named endpoint.

        `_incident` is deliberately not here. It belongs to the capture
        being harvested and travels inside the `_CapturedMeeting`; the next
        start clears it.

        The two spools are taken out rather than discarded: whoever ends the
        capture owns the files from here on, which is the writer for a
        harvest and the caller itself for an abandoned start.
        """
        self._microphone_spool = None
        self._system_spool = None
        self._last_microphone_arrival = None
        self._session_token = None
        self._start_time = None
        self._endpoint_name = None
        self._current_level = float("-inf")
        self._system_level = float("-inf")

    def _begin_capture(self, token: int) -> None:
        """Open both devices and publish them, in one callable on the owner thread.

        There is no publish check: nothing can happen between the open and
        the publish, because they are one callable on the only thread that
        writes the state. The token is claimed in the first lock hold, so the
        two callbacks are admitted from the moment their devices are live.

        `_start_time` is read immediately before the system source starts,
        which is the earliest instant a far-side block can arrive, so a block
        delivered while `start()` is still returning lands at a non-negative
        offset instead of being trimmed by `place_on_timeline`. It is
        published in that same lock hold, i.e. before that call rather than
        after it: `WindowsLoopbackSource.start` hands PyAudio a
        `stream_callback` and PyAudio starts the stream inside `open()`, so
        far-side blocks are already being kept while `start()` has not
        returned. Publishing afterwards left that window answering
        `is_recording: false` for audio the recording already contains. Both
        sources resolve `endpoint_name` in their constructor, so the name is
        publishable at the same instant and no window can report a live
        meeting it cannot name.
        """
        with self._lock:
            if self._state is not MeetingState.IDLE:
                raise MeetingCaptureAbortedError(MEETING_BUSY_DETAIL)
            self._session_token = token
            self._incident = None
            self._current_level = float("-inf")
            self._system_level = float("-inf")
            self._endpoint_name = None
        self._transition(MeetingState.STARTING)

        source: SystemAudioSource | None = None
        stream: sd.InputStream | None = None
        spools: dict[str, MeetingSpool] = {}
        try:
            source = create_system_audio_source(self._settings)
            if source is None:
                raise SystemAudioUnavailableError(
                    "System audio capture is not available on this platform — "
                    "meeting recording requires Windows or macOS"
                )
            self._settings.temp_dir.mkdir(parents=True, exist_ok=True)
            meeting_id = uuid.uuid4().hex[:12]
            spools = {
                name: MeetingSpool(self._settings.temp_dir, meeting_id, name)
                for name in (MICROPHONE_SOURCE, SYSTEM_SOURCE)
            }
            started_at = time.monotonic()
            work: queue.Queue = queue.Queue(
                maxsize=self._settings.meeting_spill_queue_blocks
            )
            drain = self._spill.submit(self._drain_spill, work, spools, started_at)
            with self._lock:
                self._start_time = started_at
                self._endpoint_name = source.endpoint_name
                self._last_microphone_arrival = started_at
                self._microphone_spool = spools[MICROPHONE_SOURCE]
                self._system_spool = spools[SYSTEM_SOURCE]
                self._spill_queue = work
                self._spill_drain = drain
            source.start(
                functools.partial(self._system_callback, token),
                functools.partial(self._system_failed, token),
            )
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
                self._forget_capture()
            self._finish_spill()
            for spool in spools.values():
                spool.discard()
            self._transition(MeetingState.IDLE)
            raise

        with self._lock:
            self._system_source = source
            self._stream = stream
        self._transition(MeetingState.RECORDING)

    async def stop(self) -> MeetingRecording:
        """End the capture on the owner thread and await the file it writes.

        Answers with a `MeetingRecording` rather than the bare path the
        dictation recorder returns: the duration and the incident belong to
        the capture that produced the file and cannot be read off the
        recorder afterwards, which by then may be serving a later meeting.

        Both awaits are detachable and neither of them owns the work behind
        it: once `_end_capture` has returned, the write is already submitted,
        so a disconnect from that point on costs the answer and never the
        recording, and the file's path reaches the log without it.

        A cancellation that lands on the first await is different: the harvest
        may not have run yet, and when it does it can still find both spools
        empty and produce no file at all.
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

        Forgetting the capture, releasing it and returning to `IDLE` are all
        three in the `finally`, and they hold the three halves of one
        invariant: an idle recorder holds no device handle and describes no
        meeting. The forgetting is there because a raise inside the harvest
        hold used to leave the fields that name the capture — `_start_time`,
        `_endpoint_name`, `_session_token` — set on a recorder that was
        already `IDLE` and already empty-handed, which answered
        `is_recording: true` with a duration ticking up and a named endpoint
        for a call nothing was capturing, refused every stop with 409, and
        healed only on the next successful start. The transition is there
        because the loop-side task awaiting this may be cancelled — FastAPI
        cancels the endpoint task when a client disconnects — and the
        recorder must not be left reporting itself busy with both handles
        already closed. The release is there because the mirror state is just
        as reachable: the handles are taken out of the fields in the first
        statements of the harvest hold, so anything that raises later in it —
        `native_sample_rate`, which is a property on a live device object,
        or the tuple build itself — would otherwise reach an `IDLE` recorder
        holding an open microphone stream and an open loopback source that no
        reference path survives to, and the next start would open a second
        pair on top of them. `_release_devices` never raises, so it cannot
        cost the transition behind it. Submitting the write here, as the last
        act on the owner thread, is what puts the recording out of the
        cancellation's reach: the `_CapturedMeeting` never crosses back to
        the loop.

        The incident is harvested in the same lock hold as the spools and
        travels inside the `_CapturedMeeting`, because it describes this
        capture while the live copy belongs to whichever meeting starts next.

        The spill thread is drained before the harvest reads the spools, and
        only after the transition to `STOPPING` has made `_store` refuse every
        further block, so the sentinel is the last thing the queue can carry
        and nothing captured is left unwritten.

        Both level meters are cleared with the rest of the capture's identity,
        by the same `_forget_capture` the abandoned-start path calls, so a
        status read between a stop and the next start cannot report the ended
        meeting's levels next to `system_endpoint: null`.
        """
        with self._lock:
            if self._state is not MeetingState.RECORDING:
                raise MeetingCaptureAbortedError("Not recording")
        self._transition(MeetingState.STOPPING)

        stream: sd.InputStream | None = None
        source: SystemAudioSource | None = None
        self._finish_spill()
        try:
            with self._lock:
                stream = self._stream
                source = self._system_source
                self._stream = None
                self._system_source = None
                microphone_spool = self._microphone_spool
                system_spool = self._system_spool
                system_rate = self._settings.sample_rate
                if source is not None:
                    system_rate = source.native_sample_rate
                captured = _CapturedMeeting(
                    microphone_spool=microphone_spool,
                    system_spool=system_spool,
                    system_rate=system_rate,
                    recording_start=self._start_time,
                    recording_stop=time.monotonic(),
                    incident=self._incident,
                )
        finally:
            with self._lock:
                self._forget_capture()
            _release_devices(stream, source)
            self._transition(MeetingState.IDLE)

        captured.microphone_spool.close()
        captured.system_spool.close()
        if not captured.microphone_spool.frames and not captured.system_spool.frames:
            captured.microphone_spool.discard()
            captured.system_spool.discard()
            raise MeetingCaptureEmptyError("No audio data captured")
        return self._writer.submit(self._write_captured_meeting, captured)

    def _write_captured_meeting(self, captured: _CapturedMeeting) -> MeetingRecording:
        """Write the harvested capture and describe the file it produced.

        The log line is how a meeting recording is found: nothing reads the
        stop response's filename, so the record of where the audio went has
        to be made by the party that owns the write rather than by the one
        that asked for it, which may already be gone.

        The duration and the incident come back with the path so the answer
        describes this capture whatever the recorder is doing by the time the
        write lands.
        """
        try:
            output_path = self._assemble_and_write(captured)
        except Exception as e:
            log.error("Writing the meeting recording failed", exc_info=True)
            raise MeetingWriteFailedError(str(e) or type(e).__name__) from e
        log.info("Meeting recording written to %s", output_path)
        return MeetingRecording(
            path=output_path,
            duration_seconds=captured.recording_stop - captured.recording_start,
            incident=captured.incident,
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
            spools = [self._microphone_spool, self._system_spool]
            self._stream = None
            self._system_source = None
            self._forget_capture()
        self._finish_spill()
        for spool in spools:
            if spool is not None:
                spool.discard()
        _release_devices(stream, source)
        self._transition(MeetingState.IDLE)

    def _assemble_and_write(self, captured: _CapturedMeeting) -> Path:
        """Stream both spools into one mix file and read the WAV out of it.

        Off the event loop, because run inline this blocked every other
        endpoint for the whole write -- including `/health` and the meeting
        status the widget polls, which is the moment the user is waiting on
        their transcript.

        The spools and the mix file go in the `finally` — they are transient
        by construction, and a write that failed must not leave the raw
        capture behind next to no recording. Every mapping `_mix_into` opens
        is closed by that call's own `ExitStack` before it returns or
        propagates, because an open mapping cannot be unlinked on Windows and
        a raising call's frame -- and so its memmaps -- stays alive in the
        traceback for as long as the exception does.
        """
        output_path = self._settings.temp_dir / f"meeting_{uuid.uuid4().hex[:12]}.wav"
        mix_path = self._settings.temp_dir / f"meeting_mix_{uuid.uuid4().hex[:12]}.f32"
        try:
            return self._mix_into(captured, mix_path, output_path)
        finally:
            try:
                mix_path.unlink(missing_ok=True)
            except OSError:
                log.warning("The meeting mix file %s could not be removed", mix_path)
            captured.microphone_spool.discard()
            captured.system_spool.discard()

    def _mix_into(
        self, captured: _CapturedMeeting, mix_path: Path, output_path: Path
    ) -> Path:
        """Sum both spools into one mix file and read the WAV out of it.

        The mix is a `np.memmap` over a temporary file rather than an array:
        a meeting has no length limit, and the recording only has to fit on
        the disk it was already being captured to. A fresh memmap is
        zero-filled, which is the silence a source that produced nothing
        contributes, so both sources are simply summed into it.

        All three mappings — the mix and one over each spool — are registered
        with an `ExitStack` as they are opened, so a write that fails part-way
        leaves none of them open. Nothing reads any of them after the stack
        unwinds; the caller's `finally` only removes the files they mapped.
        """
        target_rate = self._settings.sample_rate
        chunk_frames = self._settings.meeting_assembly_chunk_frames
        span = captured.recording_stop - captured.recording_start
        total_samples = max(int(round(span * target_rate)), 0)

        with ExitStack() as mappings:
            mix = np.memmap(
                mix_path, dtype=np.float32, mode="w+", shape=(max(total_samples, 1),)
            )
            mappings.callback(close_memmap, mix)
            timeline = mix[:total_samples]
            common = {
                "target_rate": target_rate,
                "recording_start": captured.recording_start,
                "gap_tolerance_blocks": self._settings.meeting_gap_tolerance_blocks,
                "rate_tolerance": self._settings.meeting_rate_tolerance,
                "chunk_frames": chunk_frames,
                "out": timeline,
            }
            for spool, nominal_rate in (
                (captured.microphone_spool, target_rate),
                (captured.system_spool, captured.system_rate),
            ):
                samples = spool.samples()
                mappings.callback(close_memmap, samples)
                place_on_timeline(
                    spool.index(), samples, nominal_rate=nominal_rate, **common
                )
            normalize_in_place(timeline, chunk_frames)
            return write_wav_streaming(
                output_path, timeline, target_rate, 1, chunk_frames
            )

    @property
    def is_recording(self) -> bool:
        """Whether audio is arriving right now.

        True from before the system source is asked to start, which is inside
        `STARTING` and seconds before the microphone stream is up: far-side
        audio is kept from the moment that call is made, and on Windows from
        inside it, so this is what the widget's indicator must follow.
        `_start_time` carries both this answer and `duration_seconds`, and
        `status_snapshot` is what reads the two together.
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
        cleared in the same one. That makes the two consistent at any single
        instant; a report that names both has to read them at one instant
        too, which is `status_snapshot`.
        """
        with self._lock:
            return self._endpoint_name

    def status_snapshot(self) -> MeetingStatusSnapshot:
        """Read every reported fact in one lock hold.

        The writes are already atomic — the harvest publishes and clears the
        clock, the endpoint name and both meters together — and this is the
        matching read, so what the caller reports describes one moment of
        the recorder rather than up to six.

        A stalled microphone is derived here rather than watched by a
        background task: the last arrival is written by the callback that
        already takes this lock, so the staleness is a comparison against the
        clock and nothing new touches a device handle.

        The incident is reported only while a capture is running, because
        `_incident` deliberately outlives the harvest that reads it and a
        status answering with a finished meeting's incident would degrade the
        widget for a call that had already ended.
        """
        stalled = False
        with self._lock:
            start_time = self._start_time
            if start_time is not None and microphone_has_stalled(
                self._last_microphone_arrival,
                time.monotonic(),
                self._settings.meeting_stall_tolerance_seconds,
            ):
                stalled = True
            snapshot = MeetingStatusSnapshot(
                is_recording=start_time is not None,
                duration_seconds=(
                    0.0 if start_time is None else time.monotonic() - start_time
                ),
                level_db=self._current_level,
                system_endpoint=self._endpoint_name,
                system_level_db=self._system_level,
                capture_incident=None if start_time is None else self._incident,
            )
        if not stalled:
            return snapshot
        self._note_incident(
            CaptureIncident.MICROPHONE_STALLED,
            f"no microphone block has arrived for more than "
            f"{self._settings.meeting_stall_tolerance_seconds} s",
        )
        with self._lock:
            return snapshot._replace(capture_incident=self._incident)

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
        self._spill.shutdown(wait=False)

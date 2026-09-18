"""Meeting recorder — microphone plus system audio, mixed into one WAV.

Writes the same 16 kHz mono 16-bit WAV `MicrophoneRecorder` does, so the file
enters the pipeline through the same door; `stop()` answers with a
`MeetingRecording` rather than a bare `Path`, which is why `AudioRecorder`
declares none. System audio comes from `app.audio.system_source` (ADR 037).
Every device handle and lifecycle transition runs on `self._devices`' single
worker, and the file write on `self._writer`'s (ADR 048).
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
from typing import ClassVar, NamedTuple, TypeVar

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
    SystemAudioUnsupportedError,
    create_system_audio_source,
)
from app.audio.timeline import normalize_in_place, place_on_timeline
from app.core.errors import NotReadyError, ResourceUnavailableError

log = logging.getLogger(__name__)

_T = TypeVar("_T")


def _drop_outcome(pending: asyncio.Future[object]) -> None:
    """Retrieve a finished future's exception so it is not reported unhandled.

    An unread exception reaches the event loop's handler as `Future exception
    was never retrieved`, at ERROR level with a traceback.
    """
    if not pending.cancelled():
        pending.exception()


class MeetingState(str, Enum):
    """The four states a meeting recorder can be in.

    `STARTING` and `STOPPING` are the windows in which the two device handles
    are spoken for while one of them is not yet open, or not yet released.
    """

    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"


class CaptureIncident(str, Enum):
    """Something ended or degraded a capture, named so the widget can say it.

    The first incident of a capture wins. A meeting with an incident is still a
    meeting: capture keeps running and the file is still written.
    """

    MICROPHONE_STALLED = "microphone_stalled"
    SYSTEM_AUDIO_ENDED = "system_audio_ended"
    STORAGE_FAILED = "storage_failed"
    STORAGE_LOW = "storage_low"
    STORAGE_BACKLOG = "storage_backlog"


def microphone_has_stalled(last_arrival: float | None, now: float, tolerance: float) -> bool:
    """Whether the microphone has gone quiet for longer than `tolerance`.

    Derived rather than watched, so nothing touches a device handle off the
    owner thread (ADR 048). `None` means nothing is capturing, which is no stall.
    """
    return last_arrival is not None and now - last_arrival > tolerance


MICROPHONE_SOURCE = "microphone"

SYSTEM_SOURCE = "system"

_SPILL_SENTINEL = object()


class MeetingCaptureAbortedError(NotReadyError):
    """No meeting file can be produced, and it is the caller's situation.

    Covers a start that found the recorder busy and a stop with nothing
    recording. A `NotReadyError`, so the 409 is inherited rather than declared.
    """

    code: ClassVar[str] = "meeting_capture_aborted"


class MeetingCaptureEmptyError(MeetingCaptureAbortedError):
    """A meeting ran, both capture paths delivered nothing, and there is no file.

    Answers 410 where its base answers 409, and the widget picks its message
    from the status rather than from the prose.
    """

    status_code: ClassVar[int] = 410
    code: ClassVar[str] = "meeting_capture_empty"


class MeetingWriteFailedError(ResourceUnavailableError):
    """The capture ended, both devices were released, and no file was written.

    Answers 507, not its base's 503: by then the recorder is `IDLE` and holds
    nothing, so the widget must take its indicator down.
    """

    status_code: ClassVar[int] = 507
    code: ClassVar[str] = "meeting_write_failed"


MEETING_BUSY_DETAIL = (
    "The meeting recorder is busy — a previous recording is still "
    "starting, recording or releasing its devices"
)


class _CapturedMeeting(NamedTuple):
    """Everything the WAV needs, taken out of the recorder in one lock hold.

    The two spools are files on disk; discarding them is the writer's
    obligation from here on.
    """

    microphone_spool: MeetingSpool
    system_spool: MeetingSpool
    system_rate: int
    recording_start: float
    recording_stop: float
    incident: CaptureIncident | None


class MeetingRecording(NamedTuple):
    """A written meeting file and the facts describing that same capture.

    The duration and the incident travel with the path because they are
    properties of the capture: a later meeting may already own the recorder.
    """

    path: Path
    duration_seconds: float
    incident: CaptureIncident | None


class MeetingStatusSnapshot(NamedTuple):
    """The six reported facts as they stood in one lock hold.

    Reading the properties one after another does not describe one moment; a
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
    initialises COM on the calling thread. Each close is guarded separately.
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

        Called only from callables running on `self._devices`, so the thread that
        owns the two handles is the thread that writes the state describing them.
        """
        with self._lock:
            self._state = state

    def _submit_on_devices(self, fn: Callable[..., _T], *args: object) -> Future[_T]:
        """Queue `fn` on the owner thread and count it as in flight at once.

        The increment precedes the submit, so `is_busy` covers a queued command
        too. A submit onto a retired thread raises `MeetingCaptureAbortedError`.
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

        The await is shielded: cancelling it detaches the awaiter from the answer
        and the submitted work still runs on its worker.
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

        Kept only for the session that registered the callback, and only while it
        is capturing. Returns whether it was queued; a full queue is an incident.
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

        Returns whether this call was the one that recorded it, so a site reached
        repeatedly logs one line per capture rather than one per call.
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

        Stops accumulating — rather than returning — on a storage failure or a
        free-space floor, so the callbacks keep finding a drained queue.
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

        Called on the owner thread once the state forbids further puts, so the
        sentinel is the last item the queue can receive.
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

        The keep and the level write are separate lock holds, so the session is
        re-checked in the second. A non-zero `status` is a stall, not a quiet room.
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

        The level write is a second lock hold, so a stop landing in the gap must
        not be followed by the ended meeting's far-side level.
        """
        self._store(token, SYSTEM_SOURCE, arrival, mono)
        with self._lock:
            if self._session_token == token:
                self._system_level = rms_dbfs(mono)

    async def start(self) -> None:
        """Send the owner thread a start command and wait for its answer.

        Reads no recorder state: `_begin_capture` decides. A failure sends an
        abandonment behind it and re-raises the original, cancellation included.
        """
        token = next(self._tokens)
        try:
            await self._run_on_devices(self._begin_capture, token)
        except BaseException:
            try:
                self._submit_on_devices(self._abandon_capture, token)
            except MeetingCaptureAbortedError:
                log.warning(
                    "The meeting recorder was retired before the abandoned start "
                    "could be torn down"
                )
            raise

    def _forget_capture(self) -> None:
        """Return every field describing a capture to its no-meeting value.

        Called with `_lock` held by every site that ends a capture. `_incident` is
        excluded; whoever ended the capture owns the two spool files from here on.
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

        `_start_time` is read and published before the system source starts, which
        is the earliest instant a far-side block can arrive.
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
                raise SystemAudioUnsupportedError(
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

        Both awaits are detachable: once `_end_capture` has returned the write is
        submitted, so a disconnect costs the answer and never the recording.
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

        The `finally` holds one invariant: an idle recorder holds no device handle
        and describes no meeting. The spill thread is drained before the harvest.
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

        The log line is how a meeting recording is found: nothing reads the stop
        response's filename, and the party that asked may already be gone.
        """
        try:
            output_path = self._assemble_and_write(captured)
        except Exception as e:
            log.error("Writing the meeting recording failed", exc_info=True)
            raise MeetingWriteFailedError(str(e) or "the write failed without saying why") from e
        log.info("Meeting recording written to %s", output_path)
        return MeetingRecording(
            path=output_path,
            duration_seconds=captured.recording_stop - captured.recording_start,
            incident=captured.incident,
        )

    def _abandon_capture(self, token: int) -> None:
        """Tear down the capture `token` names, and only that one.

        Runs behind the start it abandons, in submission order; a token that no
        longer matches returns without touching the live recording.
        """
        with self._lock:
            if self._session_token != token:
                return
        self._discard_capture()

    def _discard_capture(self) -> None:
        """Release both handles and return to idle, writing no WAV.

        Leaves the recorder exactly as a start that failed on its devices does,
        both level meters included.
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

        The spools and the mix file go in the `finally`: a write that failed must
        not leave the raw capture behind next to no recording.
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

        The mix is a `np.memmap`, so a meeting has no length limit beyond the disk.
        All three mappings are closed by an `ExitStack` before this returns.
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

        True from before the system source is asked to start, inside `STARTING`:
        far-side audio is kept from that moment, so the indicator must follow it.
        """
        with self._lock:
            return self._start_time is not None

    @property
    def is_busy(self) -> bool:
        """Whether the microphone and the render endpoint are spoken for.

        Counts the device commands in flight as well as the state, so its only
        failure direction is refusing something it could have allowed.
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

        A recording that captured only the microphone is otherwise indistinguishable
        from a working one until someone plays the file back.
        """
        with self._lock:
            return self._system_level

    @property
    def system_endpoint(self) -> str | None:
        """The output being captured, or None when nothing is being captured.

        Published and cleared in the same lock hold as the clock `is_recording`
        reads; a report naming both reads them at one instant via `status_snapshot`.
        """
        with self._lock:
            return self._endpoint_name

    def status_snapshot(self) -> MeetingStatusSnapshot:
        """Read every reported fact in one lock hold.

        A stalled microphone is derived here rather than watched. The incident is
        reported only while a capture is running.
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

        For app shutdown, not a substitute for `stop()`, and returns without
        waiting for the release. Terminal for recording, but idempotent.
        """
        try:
            self._submit_on_devices(self._shutdown_capture)
        except MeetingCaptureAbortedError:
            return
        self._devices.shutdown(wait=False)

    def _shutdown_capture(self) -> None:
        """Discard the capture, then retire the writer, on the owner thread.

        Retired here rather than from the event loop: behind the discard, the
        device queue's own order guarantees an `_end_capture` has submitted its write.
        """
        self._discard_capture()
        self._writer.shutdown(wait=False)
        self._spill.shutdown(wait=False)

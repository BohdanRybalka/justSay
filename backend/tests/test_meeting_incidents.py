"""Spec 153: a meeting has no length limit, and what ends one early is named.

Sibling of `test_meeting_recorder.py` rather than a section of it: that module
is the single biggest concentration of assertions in the suite already, and
everything here is about the two things spec 153 added -- capture that is
bounded by free disk space instead of by a byte ceiling, and the
`CaptureIncident` token that says what went wrong.

Every device is stubbed, exactly as in `test_meeting_recorder.py`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import tracemalloc
import wave
from unittest.mock import patch

import numpy as np
import pytest
import sounddevice as sd

from app.audio.config import AudioSettings
from app.audio.dependencies import get_meeting_recorder
from app.audio.meeting_recorder import (
    MICROPHONE_SOURCE,
    CaptureIncident,
    MeetingRecorder,
    MeetingWriteFailedError,
    microphone_has_stalled,
)
from app.audio.meeting_spool import MeetingSpool
from app.main import app

BLOCK_FRAMES = 1024

SYSTEM_RATE = 48000

OLD_RAW_BYTE_CEILING = 700_000_000

NO_CALLBACK_FLAGS = sd.CallbackFlags()


class _FakeSystemAudioSource:
    """The same stub `test_meeting_recorder.py` drives, with a failure channel."""

    def __init__(self, rate: int = SYSTEM_RATE):
        self._rate = rate
        self.on_block = None
        self.on_failure = None
        self.stopped = False

    @property
    def native_sample_rate(self) -> int:
        return self._rate

    @property
    def endpoint_name(self) -> str:
        return "Headset [Loopback]"

    def start(self, on_block, on_failure=None) -> None:
        self.on_block = on_block
        self.on_failure = on_failure

    def stop(self) -> None:
        self.stopped = True

    def deliver(self, arrival: float, frames: int = BLOCK_FRAMES, fill: float = 0.2) -> None:
        self.on_block(arrival, np.full(frames, fill, dtype=np.float32))

    def fail(self, reason: str = "the helper exited with code 3") -> None:
        self.on_failure(reason)


@pytest.fixture
def settings(tmp_path):
    return AudioSettings(sample_rate=16000, channels=1, temp_dir=tmp_path / "tmp")


@pytest.fixture
def source():
    fake = _FakeSystemAudioSource()
    with patch("app.audio.meeting_recorder.create_system_audio_source", return_value=fake):
        yield fake


@pytest.fixture
def microphone_stream():
    with patch("app.audio.meeting_recorder.sd.InputStream"):
        yield


def _feed(recorder: MeetingRecorder, count: int, frames: int = BLOCK_FRAMES, fill=0.3):
    """Deliver `count` microphone blocks as the live session's own callback.

    Waits while the spill queue is full, the way a real device does by
    arriving in real time: a loop that puts as fast as Python can run
    outruns the drain thread on a long feed and reports `storage_backlog`,
    which would be this helper's own speed rather than anything the test
    under it drove.
    """
    token = recorder._session_token
    started = recorder._start_time
    block = np.full((frames, 1), fill, dtype=np.float32)
    for index in range(count):
        work = recorder._spill_queue
        while work is not None and work.full():
            time.sleep(0.001)
        recorder._microphone_callback(token, block, frames, None, NO_CALLBACK_FLAGS)
        recorder._last_microphone_arrival = started + index * frames / 16000


def _wav_frames(path) -> int:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes()


def test_no_audio_setting_bounds_a_meeting_by_size_or_by_time():
    """AC: the ceiling is gone, and no successor field carries it.

    Enumerated rather than asserted against one deleted name: the defect this
    spec closes was a meeting ending itself part-way through, and any field
    that bounds the capture by bytes or by seconds would bring it back under a
    different spelling.
    """
    bounding = {
        name
        for name in AudioSettings.model_fields
        if name.startswith("meeting_")
        and ("max" in name or "limit" in name or "ceiling" in name or "cap" in name)
    }

    assert bounding == set(), (
        f"{sorted(bounding)} bounds a meeting recording again -- spec 153 removed "
        f"the 45-minute ceiling and the user reversed it explicitly"
    )
    assert "meeting_max_raw_bytes" not in AudioSettings.model_fields


@pytest.mark.asyncio
async def test_a_capture_past_the_old_ceiling_is_kept_whole_and_reports_nothing(
    settings, source, microphone_stream
):
    """AC: a meeting fed more than the old 700,000,000-byte ceiling keeps all of it.

    Fed through the microphone at the target rate, so the bytes -- not a
    resample -- are what this test is about. The old `_store` stopped
    accumulating at exactly this figure and flipped a flag nothing read; the
    spool has to hold every frame and report no incident at all.
    """
    recorder = MeetingRecorder(settings)
    frames_per_block = 16384
    blocks = OLD_RAW_BYTE_CEILING // (frames_per_block * 4) + 1

    try:
        await recorder.start()
        _feed(recorder, blocks, frames=frames_per_block, fill=0.1)
        recorder._finish_spill()

        kept = recorder._microphone_spool.frames
        assert kept * 4 > OLD_RAW_BYTE_CEILING, (
            f"only {kept * 4} bytes reached the spool, which is under the ceiling "
            f"this test exists to drive past"
        )
        assert kept == blocks * frames_per_block, (
            f"{blocks * frames_per_block - kept} frames were dropped on the way to disk"
        )
        assert recorder._incident is None, (
            f"a capture that lost nothing reported {recorder._incident}"
        )
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_a_meeting_longer_than_the_old_ceiling_writes_a_wav_of_its_full_span(
    settings, source, microphone_stream
):
    """AC: the written file covers the whole fed duration.

    The arrival span rather than the byte count is what sizes the WAV, so this
    feeds a 46-minute span -- past the 45.6 minutes the old cap allowed -- and
    reads the frame count back out of the file.
    """
    recorder = MeetingRecorder(settings)
    span_seconds = 46 * 60

    try:
        await recorder.start()
        started = recorder._start_time
        recorder._start_time = started - span_seconds
        _feed(recorder, 4)
        recording = await recorder.stop()
        await asyncio.to_thread(recorder._writer.shutdown, True)

        frames = _wav_frames(recording.path)
        assert frames >= span_seconds * settings.sample_rate, (
            f"the WAV holds {frames / settings.sample_rate / 60:.1f} minutes of a "
            f"{span_seconds / 60:.0f}-minute meeting"
        )
        assert recording.incident is None
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_what_capture_retains_does_not_grow_with_the_length_of_the_meeting(
    settings, source, microphone_stream
):
    """AC: four times the blocks does not cost four times the retained memory.

    Measured with `tracemalloc` around capture only. The old store held every
    block for the length of the call, so this ratio was the block count's;
    what remains resident now is the queue and the index, and neither is
    proportional to the meeting.
    """
    async def retained(blocks: int) -> int:
        recorder = MeetingRecorder(settings)
        try:
            await recorder.start()
            tracemalloc.start()
            before = tracemalloc.get_traced_memory()[0]
            _feed(recorder, blocks)
            recorder._finish_spill()
            after = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            return after - before
        finally:
            recorder.cleanup()

    baseline = await retained(200)
    longer = await retained(800)
    accumulating = 800 * BLOCK_FRAMES * 4

    assert longer <= max(baseline, 64 * 1024) * 1.2, (
        f"capturing four times as long retained {longer} bytes against the "
        f"baseline's {baseline} -- the capture is accumulating in memory again"
    )
    assert longer < accumulating / 4, (
        f"{longer} retained bytes is within a quarter of the {accumulating} an "
        f"accumulating store would hold, so this threshold proves nothing"
    )


@pytest.mark.asyncio
async def test_a_spill_that_cannot_be_written_is_reported_once_and_keeps_what_it_had(
    settings, source, microphone_stream, caplog
):
    """AC: an `OSError` from the spool stops accumulation and names itself.

    The audio already on disk is a recording the user can still have, so the
    capture keeps running and the WAV is still written -- with the incident
    saying which part of it is missing.
    """
    recorder = MeetingRecorder(settings)
    real_append = MeetingSpool.append
    calls = {"count": 0}

    def failing_append(self, arrival, samples):
        calls["count"] += 1
        if calls["count"] > 2:
            raise OSError("No space left on device")
        return real_append(self, arrival, samples)

    try:
        with caplog.at_level(logging.WARNING, logger="app.audio.meeting_recorder"):
            with patch.object(MeetingSpool, "append", failing_append):
                await recorder.start()
                _feed(recorder, 20)
                await asyncio.sleep(0.05)
                recorder._finish_spill()

        assert recorder._incident is CaptureIncident.STORAGE_FAILED
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, (
            f"{len(errors)} ERROR lines for one failing capture -- the user's log "
            f"is filled with one line per block"
        )
        assert "No space left on device" in errors[0].getMessage()
        assert recorder._microphone_spool.frames == 2 * BLOCK_FRAMES
    finally:
        recorder.cleanup()


@pytest.mark.anyio
async def test_a_storage_failure_answers_200_with_the_incident_on_the_wire(
    client, settings, source, microphone_stream
):
    """AC: `/audio/meeting/stop` answers 200 and names the incident.

    Not 507: a meeting that lost part of its audio still produced a file, and
    the widget must offer it rather than describe the call as lost.
    """
    recorder = MeetingRecorder(settings)
    app.dependency_overrides[get_meeting_recorder] = lambda: recorder
    real_append = MeetingSpool.append
    calls = {"count": 0}

    def failing_append(self, arrival, samples):
        calls["count"] += 1
        if calls["count"] > 2:
            raise OSError("No space left on device")
        return real_append(self, arrival, samples)

    try:
        with patch.object(MeetingSpool, "append", failing_append):
            await recorder.start()
            started = recorder._start_time
            _feed(recorder, 20)
            recorder._start_time = started - 5.0
            await asyncio.sleep(0.05)

        resp = await client.post("/audio/meeting/stop")

        assert resp.status_code == 200
        assert resp.json()["capture_incident"] == "storage_failed"
    finally:
        recorder.cleanup()
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_capture_stops_accumulating_while_the_disk_cannot_hold_the_assembly(
    settings, source, microphone_stream
):
    """AC: the floor is the computed assembly reserve, not `ENOSPC`.

    Filling the disk during capture and discovering it only when the mix file
    cannot be allocated would lose a recording that was already safely on
    disk, so capture stops while there is still room to assemble what it has.
    """
    recorder = MeetingRecorder(settings)

    try:
        with patch("app.audio.meeting_recorder.free_bytes", return_value=0):
            await recorder.start()
            _feed(recorder, 20)
            await asyncio.sleep(0.05)
            recorder._finish_spill()

        assert recorder._incident is CaptureIncident.STORAGE_LOW
        assert recorder._microphone_spool.frames == 0, (
            "capture went on filling the disk it had already been told it "
            "could not assemble from"
        )
    finally:
        recorder.cleanup()


@pytest.mark.parametrize(
    ("last_arrival", "now", "expected"),
    [
        (None, 100.0, False),
        (100.0, 100.5, False),
        (100.0, 103.0, False),
        (100.0, 103.1, True),
    ],
)
def test_when_a_microphone_counts_as_stalled(last_arrival, now, expected):
    """The whole staleness decision, with no recorder and no clock."""
    assert microphone_has_stalled(last_arrival, now, 3.0) is expected


@pytest.mark.asyncio
async def test_a_microphone_that_stops_delivering_is_reported_once_however_often_read(
    tmp_path, source, microphone_stream, caplog
):
    """AC: the status names the stall, and the log carries exactly one line.

    The staleness is derived inside `status_snapshot`, which the widget calls
    every two seconds for the length of the call -- a log line per read would
    be hundreds of them for one unplugged microphone.
    """
    stalling = AudioSettings(
        sample_rate=16000,
        channels=1,
        temp_dir=tmp_path / "tmp",
        meeting_stall_tolerance_seconds=0.05,
    )
    recorder = MeetingRecorder(stalling)

    try:
        with caplog.at_level(logging.WARNING, logger="app.audio.meeting_recorder"):
            await recorder.start()
            _feed(recorder, 1)
            await asyncio.sleep(0.2)
            snapshots = [recorder.status_snapshot() for _ in range(5)]

        assert all(s.capture_incident is CaptureIncident.MICROPHONE_STALLED for s in snapshots)
        assert all(s.is_recording for s in snapshots)
        stall_lines = [
            record
            for record in caplog.records
            if "microphone_stalled" in record.getMessage()
        ]
        assert len(stall_lines) == 1, (
            f"{len(stall_lines)} lines for one stall across five status reads"
        )
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_a_status_read_before_the_tolerance_expires_reports_nothing(
    settings, source, microphone_stream
):
    """The other half: an ordinary meeting must not be degraded by this."""
    recorder = MeetingRecorder(settings)

    try:
        await recorder.start()
        _feed(recorder, 1)

        assert recorder.status_snapshot().capture_incident is None
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_a_system_source_that_fails_is_named_and_the_microphone_keeps_recording(
    settings, source, microphone_stream
):
    """AC: the far side ending is an incident, not the end of the meeting.

    Half a call is what the user has, and taking the capture down would
    replace it with nothing.
    """
    recorder = MeetingRecorder(settings)

    try:
        await recorder.start()
        source.deliver(recorder._start_time)
        source.fail("the macOS system-audio helper exited with code 3")
        _feed(recorder, 4)
        recorder._finish_spill()

        snapshot = recorder.status_snapshot()
        assert snapshot.capture_incident is CaptureIncident.SYSTEM_AUDIO_ENDED
        assert snapshot.is_recording is True
        assert recorder._microphone_spool.frames == 4 * BLOCK_FRAMES, (
            "the microphone stopped being recorded when the far side ended"
        )
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_the_first_incident_of_a_capture_is_the_one_reported(
    settings, source, microphone_stream
):
    """A later incident is usually a consequence of the first one.

    What the user needs is the reason their recording went wrong, so the
    report does not drift to whatever failed most recently.
    """
    recorder = MeetingRecorder(settings)

    try:
        await recorder.start()
        source.fail("the helper exited with code 3")
        recorder._note_incident(CaptureIncident.STORAGE_FAILED, "a later failure")

        assert recorder._incident is CaptureIncident.SYSTEM_AUDIO_ENDED
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_an_incident_does_not_outlive_the_meeting_it_describes(
    settings, source, microphone_stream
):
    """A status read between two meetings must not degrade the next one."""
    recorder = MeetingRecorder(settings)

    try:
        await recorder.start()
        started = recorder._start_time
        source.deliver(started)
        source.fail("the helper exited with code 3")
        recorder._start_time = started - 2.0
        await recorder.stop()
        await asyncio.to_thread(recorder._writer.shutdown, True)

        assert recorder.status_snapshot().capture_incident is None
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_a_meeting_leaves_no_spill_files_behind(
    settings, source, microphone_stream
):
    """The spools are transient by construction.

    Raw meeting audio sits on disk for the length of the call where it used to
    sit in RAM, so the `finally` that removes it is part of the bargain ADR
    058 struck rather than tidiness.
    """
    recorder = MeetingRecorder(settings)

    try:
        await recorder.start()
        started = recorder._start_time
        _feed(recorder, 4)
        source.deliver(started)
        recorder._start_time = started - 2.0
        await recorder.stop()
        await asyncio.to_thread(recorder._writer.shutdown, True)

        left = sorted(p.name for p in settings.temp_dir.glob("meeting_spill_*"))
        mixes = sorted(p.name for p in settings.temp_dir.glob("meeting_mix_*"))
        assert left == [], f"the capture left its raw audio on disk: {left}"
        assert mixes == [], f"the assembly left its mix file on disk: {mixes}"
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_an_abandoned_start_leaves_no_spill_files_behind(
    settings, microphone_stream
):
    """A start that failed on its second device owns the spools it opened."""
    recorder = MeetingRecorder(settings)
    fake = _FakeSystemAudioSource()

    try:
        with patch("app.audio.meeting_recorder.create_system_audio_source", return_value=fake):
            with patch(
                "app.audio.meeting_recorder.sd.InputStream",
                side_effect=RuntimeError("the microphone went away"),
            ):
                with pytest.raises(RuntimeError):
                    await recorder.start()

        await asyncio.sleep(0.05)
        left = sorted(p.name for p in settings.temp_dir.glob("meeting_spill_*"))
        assert left == [], f"the abandoned start left its spools behind: {left}"
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_the_far_side_arrives_at_its_own_rate_and_still_lands_on_the_timeline(
    settings, source, microphone_stream
):
    """The 48 kHz spool is resampled out of the file, not out of a block list.

    This is the one path that exercises `resample_chunks` against a real
    spool, so a streaming resample that produced nothing would be silent here
    rather than in `test_timeline.py`.
    """
    recorder = MeetingRecorder(settings)
    blocks = 200

    try:
        await recorder.start()
        started = recorder._start_time
        for index in range(blocks):
            source.deliver(started + index * BLOCK_FRAMES / SYSTEM_RATE, fill=0.4)
        await asyncio.sleep(0.05)
        recording = await recorder.stop()
        await asyncio.to_thread(recorder._writer.shutdown, True)

        with wave.open(str(recording.path), "rb") as wf:
            signal = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        audible = np.flatnonzero(np.abs(signal) > 1000)

        assert audible.size > 0, "the far side spoke and the file is silent"
        assert float(audible[0]) / settings.sample_rate < 0.1
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_a_microphone_underflow_flag_is_read_rather_than_discarded(
    settings, source, microphone_stream
):
    """The flag PortAudio raises when it substitutes zeros for missing audio.

    It was accepted and ignored, so a microphone that had stopped producing
    looked exactly like a quiet room.
    """
    recorder = MeetingRecorder(settings)

    try:
        await recorder.start()
        block = np.zeros((BLOCK_FRAMES, 1), dtype=np.float32)
        recorder._microphone_callback(
            recorder._session_token, block, BLOCK_FRAMES, None, sd.CallbackFlags(2)
        )

        assert recorder._incident is CaptureIncident.MICROPHONE_STALLED
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_a_failure_from_a_finished_session_cannot_degrade_the_running_one(
    settings, source, microphone_stream
):
    """The macOS reader outlives a bounded join while holding its callbacks."""
    recorder = MeetingRecorder(settings)

    try:
        await recorder.start()
        started = recorder._start_time
        stale_failure = source.on_failure
        source.deliver(started)
        recorder._start_time = started - 2.0
        await recorder.stop()
        await asyncio.to_thread(recorder._writer.shutdown, True)

        await recorder.start()
        stale_failure("the helper exited with code 3")

        assert recorder._incident is None, (
            "a finished meeting's system source degraded the meeting that "
            "started after it"
        )
    finally:
        recorder.cleanup()


def test_the_spill_thread_is_named_so_a_stack_dump_says_which_thread_it_is(settings):
    """`meeting-devices` and `meeting-writer` already carry names; so does this."""
    recorder = MeetingRecorder(settings)
    try:
        assert recorder._spill._thread_name_prefix == "meeting-spill"
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_a_write_that_fails_leaves_no_spill_or_mix_file_behind(
    settings, source, microphone_stream
):
    """The recording a disk-full write could not produce costs no disk either.

    The failure is raised from `write_wav_streaming`, which is the last thing
    assembly does and the only point at which all three mappings -- the mix
    and one over each spool -- are open at once. On Windows a file cannot be
    unlinked while a mapping over it is open, and a raising call keeps its
    frame alive in the traceback, so before `_mix_into` closed its mappings
    explicitly the `finally` that removes them swallowed `WinError 32` and
    left hundreds of megabytes of `.pcm` and `.f32` in the scratch directory
    -- during the disk-full incident, which is the worst possible moment for
    it. On POSIX the unlink succeeds either way, so this pins the behaviour
    everywhere and the regression on Windows.
    """
    recorder = MeetingRecorder(settings)
    try:
        await recorder.start()
        _feed(recorder, 8)
        for index in range(8):
            source.deliver(recorder._start_time + index * BLOCK_FRAMES / SYSTEM_RATE)
        recorder._start_time = recorder._start_time - 1.0

        def refuse_to_write(*args, **kwargs):
            raise OSError("[Errno 28] No space left on device")

        with patch(
            "app.audio.meeting_recorder.write_wav_streaming", refuse_to_write
        ), pytest.raises(MeetingWriteFailedError):
            await recorder.stop()
        await asyncio.to_thread(recorder._writer.shutdown, True)

        left_behind = sorted(path.name for path in settings.temp_dir.iterdir())
        assert left_behind == [], (
            f"a failed write left {left_behind} in the scratch directory"
        )
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_the_spill_thread_is_retired_with_the_rest_of_the_recorder(
    settings, source, microphone_stream
):
    """`cleanup()` is terminal, and a live spill thread would outlive it.

    Asks this recorder's own executor which threads it started rather than
    scanning the process for the `meeting-spill` name: every recorder names
    its worker identically, and the suite builds recorders that are never
    cleaned up, so a name scan asserts the hygiene of every test that ran
    before this one instead of what `cleanup()` did here.
    """
    recorder = MeetingRecorder(settings)
    await recorder.start()
    _feed(recorder, 2)
    started = recorder._start_time
    recorder._start_time = started - 2.0
    await recorder.stop()
    await asyncio.to_thread(recorder._writer.shutdown, True)
    spill_threads = set(recorder._spill._threads)
    assert spill_threads, "the capture never started a spill thread to retire"

    recorder.cleanup()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not any(thread.is_alive() for thread in spill_threads):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("this recorder's meeting-spill thread outlived cleanup()")


@pytest.mark.asyncio
async def test_a_block_that_passed_the_state_check_cannot_land_behind_the_sentinel(
    settings, source, microphone_stream
):
    """A callback preempted on its way to the queue still reaches the spool.

    `_store` used to release the lock before putting, so a block whose state
    check passed while the meeting was still recording could be suspended,
    let `_finish_spill` enqueue the sentinel and the drain worker return, and
    only then put -- into a queue nobody reads any more. The block was lost
    with no incident and no log line, which is the silent loss this spec
    exists to close.

    The preemption is made deterministic by delaying `put_nowait` itself
    rather than by hoping for a scheduler: the storing thread reaches the put
    and stalls there, and the owner thread finishes the spill meanwhile. With
    the put inside the same lock hold as the check, the owner thread waits for
    it instead of racing past it.
    """
    recorder = MeetingRecorder(settings)
    try:
        await recorder.start()
        work = recorder._spill_queue
        reached_the_put = threading.Event()
        queue_a_block = work.put_nowait

        def stall_on_the_way_to_the_queue(item):
            reached_the_put.set()
            time.sleep(0.3)
            queue_a_block(item)

        work.put_nowait = stall_on_the_way_to_the_queue
        block = np.full(BLOCK_FRAMES, 0.4, dtype=np.float32)
        storing = threading.Thread(
            target=recorder._store,
            args=(
                recorder._session_token,
                MICROPHONE_SOURCE,
                recorder._start_time,
                block,
            ),
            name="preempted-callback",
        )
        storing.start()
        assert reached_the_put.wait(timeout=5.0), "the block never reached the queue"
        recorder._finish_spill()
        storing.join(timeout=5.0)

        assert recorder._microphone_spool.frames == BLOCK_FRAMES, (
            f"{BLOCK_FRAMES - recorder._microphone_spool.frames} frames were "
            f"dropped between the state check and the queue, and the capture "
            f"reports {recorder._incident}"
        )
    finally:
        recorder.cleanup()

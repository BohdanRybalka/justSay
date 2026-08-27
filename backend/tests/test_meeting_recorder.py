"""Spec 066: MeetingRecorder, the platform factory, and the meeting endpoints.

Every device is stubbed. `pyaudiowpatch` is a Windows-only wheel that cannot
be installed on the ubuntu CI runner at all, so the Windows source's
block-handling logic is exercised against a fake module injected into
`sys.modules` — only real-device behaviour is left to the `[win]` checklist.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import importlib
import logging
import shutil
import sys
import threading
import time
import types
import typing
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.audio import get_active_recorder, get_meeting_recorder
from app.audio.base import write_wav
from app.audio.config import AudioSettings
from app.audio.meeting_recorder import (
    MEETING_BUSY_DETAIL,
    MeetingCaptureAbortedError,
    MeetingCaptureEmptyError,
    MeetingRecorder,
    MeetingRecording,
    MeetingState,
    MeetingWriteFailedError,
    _CapturedMeeting,
    _release_devices,
)
from app.audio.system_source import (
    SystemAudioSource,
    SystemAudioUnavailableError,
    create_system_audio_source,
)
from app.main import app

BLOCK_FRAMES = 1024


@pytest.fixture
def audio_settings(tmp_path):
    return AudioSettings(sample_rate=16000, channels=1, temp_dir=tmp_path / "tmp")


class _FakeSystemAudioSource(SystemAudioSource):
    """A system source that delivers exactly the blocks a test hands it."""

    def __init__(self, rate: int = 48000, endpoint_name: str = "Headset [Loopback]"):
        self._rate = rate
        self._endpoint_name = endpoint_name
        self.on_block = None
        self.started = False
        self.stopped = False

    @property
    def native_sample_rate(self) -> int:
        return self._rate

    @property
    def endpoint_name(self) -> str:
        return self._endpoint_name

    def start(self, on_block) -> None:
        self.on_block = on_block
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def deliver(self, arrival: float, frames: int = BLOCK_FRAMES, fill: float = 0.2) -> None:
        self.on_block(arrival, np.full(frames, fill, dtype=np.float32))


@pytest.fixture
def fake_system_source():
    source = _FakeSystemAudioSource()
    with patch("app.audio.meeting_recorder.create_system_audio_source", return_value=source):
        yield source


@pytest.fixture
def fake_microphone_stream():
    with patch("app.audio.meeting_recorder.sd.InputStream") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


SYSTEM_RATE = 48000


def _deliver_over_a_real_span(
    recorder: MeetingRecorder, source: _FakeSystemAudioSource, blocks: int
) -> int:
    """Deliver `blocks` at spaced arrivals and let the recording clock advance.

    `place_on_timeline` sizes the WAV from `recording_stop - recording_start`,
    so a capture that starts and stops inside one clock tick writes an empty
    file whatever arrived. Returns the frame count those blocks are worth at
    the target rate, which is the floor the written file has to reach.
    """
    started = recorder._start_time
    for index in range(blocks):
        source.deliver(started + index * BLOCK_FRAMES / SYSTEM_RATE)
    return blocks * BLOCK_FRAMES * 16000 // SYSTEM_RATE


def _feed_microphone(recorder: MeetingRecorder, count: int, fill: float = 0.3) -> None:
    """Deliver `count` microphone blocks as the live session's own callback.

    The token is what `_store` admits on, so a block fed without the one the
    running session claimed is dropped exactly as a stale callback's would be.
    """
    token = recorder._session_token
    for _ in range(count):
        block = np.full((BLOCK_FRAMES, 1), fill, dtype=np.float32)
        recorder._microphone_callback(token, block, BLOCK_FRAMES, None, MagicMock())


class _BlockingSystemAudioSource(_FakeSystemAudioSource):
    """A fake source that parks the device open until the test releases it.

    `opening_started` fires the moment the open reaches the block and the
    open resumes only once the test sets `release_open`, which makes "a stop
    arrives mid-open" a deterministic sequence rather than a sleep race.
    Every call records the thread that made it, so the one-owner-thread rule
    is checkable rather than reviewable.

    `block_on="stop"` parks inside the release instead, on its own pair of
    events — `closing_started` and `release_close` — so that a name never
    claims to be about the open when it is about the release. A source in
    that mode opens normally, so `await recorder.start()` completes.
    """

    def __init__(
        self,
        block_on: str = "construction",
        rate: int = 48000,
        endpoint_name: str = "Headset [Loopback]",
    ):
        self.opening_started = threading.Event()
        self.release_open = threading.Event()
        self.closing_started = threading.Event()
        self.release_close = threading.Event()
        self.calls: list[str] = []
        self.idents: list[int] = []
        self._block_on = block_on
        super().__init__(rate=rate, endpoint_name=endpoint_name)

    def construct(self, settings) -> _FakeSystemAudioSource:
        """Stand in for `create_system_audio_source` — the Windows COM
        enumeration and the macOS helper handshake both happen here."""
        self._note("construct")
        if self._block_on == "construction":
            self._park()
        return self

    def _note(self, name: str) -> None:
        self.calls.append(name)
        self.idents.append(threading.get_ident())

    def _park(self) -> None:
        self.opening_started.set()
        assert self.release_open.wait(timeout=5.0), "the test never released the open"

    def start(self, on_block) -> None:
        self._note("start")
        if self._block_on == "start":
            self._park()
        super().start(on_block)

    def stop(self) -> None:
        self._note("stop")
        if self._block_on == "stop":
            self.closing_started.set()
            assert self.release_close.wait(timeout=5.0), (
                "the test never released the close"
            )
        super().stop()


class _BlockingMicrophoneStream:
    """An `sd.InputStream` stand-in that parks the device open until released.

    `_BlockingSystemAudioSource` is the same shape for the system half. This
    one exists because the microphone open is the window the far side is
    already live in, and what happens to its audio during that window is the
    thing under test.
    """

    def __init__(self, opening_started, release_open, **kwargs):
        self.kwargs = kwargs
        self.closes = 0
        opening_started.set()
        assert release_open.wait(timeout=5.0), (
            "the test never released the microphone open"
        )

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closes += 1


class _RecordingMicrophoneStream:
    """An `sd.InputStream` stand-in that records which thread called it."""

    def __init__(self, calls: list[str], idents: list[int], **kwargs):
        self.kwargs = kwargs
        self.stops = 0
        self.closes = 0
        self._calls = calls
        self._idents = idents
        self._note("construct")

    def _note(self, name: str) -> None:
        self._calls.append(f"stream.{name}")
        self._idents.append(threading.get_ident())

    def start(self) -> None:
        self._note("start")

    def stop(self) -> None:
        self.stops += 1
        self._note("stop")

    def close(self) -> None:
        self.closes += 1
        self._note("close")


def _wait_for_devices(recorder: MeetingRecorder, timeout: float = 5.0) -> None:
    """Block until everything queued on the recorder's device thread has run.

    The executor has exactly one worker, so a barrier submitted last
    completes only after everything queued before it. Call it only once the
    open has been released — a barrier submitted while the open is still
    parked would park with it.

    After `cleanup()` the executor is retired and refuses the barrier; the
    join then does the same job, and it is finite because every parked
    callable in this module is bounded at five seconds.
    """
    try:
        recorder._devices.submit(lambda: None).result(timeout=timeout)
    except RuntimeError:
        recorder._devices.shutdown(wait=True)


def _wait_for_writes(recorder: MeetingRecorder, timeout: float = 5.0) -> None:
    """Block until every meeting file the recorder submitted has been written.

    The write is submitted by the device thread, so this is meaningful only
    after `_wait_for_devices`.
    """
    try:
        recorder._writer.submit(lambda: None).result(timeout=timeout)
    except RuntimeError:
        recorder._writer.shutdown(wait=True)


def _wav_signal(path: Path) -> tuple[int, np.ndarray]:
    """The written file as a normalised float signal, plus its rate."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return rate, samples.astype(np.float32) / 32768.0


def _hold_the_owner_thread(recorder: MeetingRecorder) -> threading.Event:
    """Occupy the device thread until the returned event is set.

    Submitted straight to the executor rather than through
    `_submit_on_devices`, so the barrier itself never counts as a device
    command in flight — that is the value under test in the row that uses it.

    The wait is bounded: a failing assertion before the test opens the gate
    would otherwise leave the worker parked forever, and the executor's
    atexit join then hangs the whole interpreter rather than reporting the
    failure.
    """
    gate = threading.Event()
    recorder._devices.submit(gate.wait, 5.0)
    return gate


def _hold_the_owner_thread_until_queued(recorder: MeetingRecorder, expected: int) -> None:
    """Occupy the device thread until `expected` counted commands are queued.

    Releasing on the counter rather than on a timer keeps the sequence
    deterministic: the hold ends exactly when the command under test has been
    submitted and not a moment before, which is the window
    `_devices_in_flight` exists to cover.
    """

    def hold() -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with recorder._lock:
                if recorder._devices_in_flight >= expected:
                    return
            time.sleep(0.005)

    recorder._devices.submit(hold)


@pytest.fixture
def blocking_system_source(request):
    source = _BlockingSystemAudioSource(block_on=getattr(request, "param", "construction"))
    with patch("app.audio.meeting_recorder.create_system_audio_source", source.construct):
        yield source


@pytest.fixture
def recording_microphone_stream():
    calls: list[str] = []
    idents: list[int] = []
    streams: list[_RecordingMicrophoneStream] = []

    def _factory(**kwargs):
        stream = _RecordingMicrophoneStream(calls, idents, **kwargs)
        streams.append(stream)
        return stream

    with patch("app.audio.meeting_recorder.sd.InputStream", _factory):
        yield types.SimpleNamespace(calls=calls, idents=idents, streams=streams)


def test_memory_cap_covers_a_45_minute_meeting():
    """AC: `meeting_max_raw_bytes` divided by the raw-store growth rate at a
    48 kHz system device is at least 45 minutes.

    The growth rate is recomputed from the live AudioSettings defaults rather
    than written down as a second number, so changing either default without
    changing the other fails here instead of silently shrinking the ceiling.
    """
    settings = AudioSettings()
    bytes_per_float32 = 4
    system_device_rate = 48000

    growth_per_second = (
        system_device_rate * bytes_per_float32 + settings.sample_rate * bytes_per_float32
    )
    capped_seconds = settings.meeting_max_raw_bytes / growth_per_second

    assert capped_seconds >= 45 * 60, (
        f"the raw buffer cap covers only {capped_seconds / 60:.1f} minutes at a "
        f"48 kHz system device — a 45-minute meeting would be truncated"
    )


@pytest.mark.asyncio
async def test_recorder_stops_accepting_blocks_at_the_cap_and_still_writes_a_wav(
    tmp_path, fake_system_source, fake_microphone_stream
):
    """The cap truncates rather than crashing or growing without bound."""
    settings = AudioSettings(
        sample_rate=16000,
        channels=1,
        temp_dir=tmp_path / "tmp",
        meeting_max_raw_bytes=BLOCK_FRAMES * 4 * 3,
    )
    recorder = MeetingRecorder(settings)

    await recorder.start()
    _feed_microphone(recorder, 50)
    recording = await recorder.stop()

    assert recording.truncated is True
    assert recording.path.exists()


@pytest.mark.asyncio
async def test_stop_writes_exactly_one_file_inside_temp_dir(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """AC: one file, inside AudioSettings.temp_dir, and nowhere else — the
    same containment MicrophoneRecorder already has."""
    recorder = MeetingRecorder(audio_settings)

    await recorder.start()
    _feed_microphone(recorder, 8)
    for i in range(8):
        fake_system_source.deliver(recorder._start_time + i * BLOCK_FRAMES / 48000)
    audio_path = (await recorder.stop()).path

    written = list(audio_settings.temp_dir.iterdir())
    assert written == [audio_path]
    assert audio_path.parent == audio_settings.temp_dir
    assert audio_path.suffix == ".wav"


@pytest.mark.asyncio
async def test_stop_writes_16khz_mono_16bit_like_the_dictation_path(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """The pipeline must receive the same kind of file it receives today."""
    recorder = MeetingRecorder(audio_settings)

    await recorder.start()
    _feed_microphone(recorder, 8)
    audio_path = (await recorder.stop()).path

    with wave.open(str(audio_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000


@pytest.mark.asyncio
async def test_both_sources_are_audible_in_the_written_wav(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """Neither half may be dropped by the mix."""
    recorder = MeetingRecorder(audio_settings)

    await recorder.start()
    start = recorder._start_time
    _feed_microphone(recorder, 16, fill=0.3)
    for i in range(48):
        fake_system_source.deliver(start + i * BLOCK_FRAMES / 48000, fill=0.2)
    time.sleep(0.02)
    audio_path = (await recorder.stop()).path

    with wave.open(str(audio_path), "rb") as wf:
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    assert np.max(np.abs(samples)) > int(0.4 * 32767), (
        "the mixed peak is below either source's own level — one of them was lost"
    )


@pytest.mark.asyncio
async def test_stop_without_start_raises(audio_settings):
    recorder = MeetingRecorder(audio_settings)

    with pytest.raises(RuntimeError, match="Not recording"):
        await recorder.stop()


@pytest.mark.asyncio
async def test_start_raises_when_no_system_source_exists(
    audio_settings, fake_microphone_stream
):
    """AC: on a platform with no system-audio source, no stream is opened."""
    recorder = MeetingRecorder(audio_settings)

    with patch("app.audio.meeting_recorder.create_system_audio_source", return_value=None):
        with pytest.raises(SystemAudioUnavailableError):
            await recorder.start()

    assert recorder.is_recording is False
    fake_microphone_stream.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_releases_both_streams(
    audio_settings, fake_system_source, fake_microphone_stream
):
    recorder = MeetingRecorder(audio_settings)

    await recorder.start()
    _feed_microphone(recorder, 3)
    recorder.cleanup()
    _wait_for_devices(recorder)

    assert recorder.is_recording is False
    assert fake_system_source.stopped is True
    fake_microphone_stream.return_value.stop.assert_called_once()
    fake_microphone_stream.return_value.close.assert_called_once()


def test_cleanup_before_any_meeting_leaves_the_recorder_idle(audio_settings):
    """`cleanup()` is unconditional, so it must be harmless on a recorder
    that never opened anything."""
    recorder = MeetingRecorder(audio_settings)

    recorder.cleanup()
    _wait_for_devices(recorder)

    assert recorder.is_recording is False


@pytest.mark.asyncio
async def test_a_failing_microphone_stream_releases_the_system_source(
    audio_settings, fake_system_source
):
    """A half-open recording would hold the loopback device forever."""
    recorder = MeetingRecorder(audio_settings)

    with patch("app.audio.meeting_recorder.sd.InputStream", side_effect=OSError("no mic")):
        with pytest.raises(OSError):
            await recorder.start()

    assert recorder.is_recording is False
    assert fake_system_source.stopped is True


@pytest.mark.asyncio
async def test_an_unwritable_scratch_directory_releases_the_system_source_too(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """The scratch directory is created inside the same try that releases.

    Created before it, a failing mkdir left the loopback endpoint open with
    nothing left holding a reference able to close it.
    """
    recorder = MeetingRecorder(audio_settings)

    with patch.object(Path, "mkdir", side_effect=OSError("read-only")):
        with pytest.raises(OSError):
            await recorder.start()

    assert recorder.is_recording is False
    assert fake_system_source.stopped is True


def test_factory_returns_none_on_a_platform_with_no_implementation(monkeypatch):
    """AC: no system-audio source on the ubuntu CI runner."""
    monkeypatch.setattr(sys, "platform", "linux")

    assert create_system_audio_source(AudioSettings()) is None


def test_factory_reports_why_the_windows_device_lookup_failed(monkeypatch):
    """A Windows machine with no usable loopback device says so (JS-78).

    None is reserved for a platform with no capture path at all. This case
    asserted the opposite until JS-78: the reason was swallowed and the caller
    told a Windows user that meeting recording requires Windows.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules, "app.audio.windows_loopback", _module_raising_on_construction()
    )

    with pytest.raises(SystemAudioUnavailableError, match="no loopback endpoint"):
        create_system_audio_source(AudioSettings())


def _module_raising_on_construction() -> types.ModuleType:
    module = types.ModuleType("app.audio.windows_loopback")

    class _Exploding:
        def __init__(self, settings):
            raise SystemAudioUnavailableError("no loopback endpoint")

    module.WindowsLoopbackSource = _Exploding
    return module


@pytest.fixture
def fake_pyaudiowpatch(monkeypatch):
    """A stand-in for the Windows-only wheel, injected before import.

    Mirrors the surface `windows_loopback` actually uses: the paFloat32 and
    paContinue constants, `PyAudio()` with the loopback lookup helpers, and
    `open()` returning a stream object.
    """
    module = types.ModuleType("pyaudiowpatch")
    module.paFloat32 = 1
    module.paContinue = 0

    class _Stream:
        def __init__(self):
            self.started = False
            self.closed = False

        def start_stream(self):
            self.started = True

        def stop_stream(self):
            self.started = False

        def close(self):
            self.closed = True

    class _PyAudio:
        instances: list = []

        def __init__(self):
            self.opened_kwargs = None
            self.stream = _Stream()
            self.terminated = False
            _PyAudio.instances.append(self)

        def get_default_wasapi_loopback(self):
            return {
                "index": 7,
                "name": "Speakers (loopback)",
                "defaultSampleRate": 48000.0,
                "maxInputChannels": 2,
            }

        def get_loopback_device_info_generator(self):
            yield self.get_default_wasapi_loopback()

        def open(self, **kwargs):
            self.opened_kwargs = kwargs
            return self.stream

        def terminate(self):
            self.terminated = True

    module.PyAudio = _PyAudio
    _PyAudio.instances = []
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", module)
    monkeypatch.delitem(sys.modules, "app.audio.windows_loopback", raising=False)
    return module


@pytest.fixture
def render_endpoints(fake_pyaudiowpatch, monkeypatch):
    """Stand in for the COM lookup, which cannot run off Windows.

    Returns the mutable role→name map the source will see, so a test can point
    the two roles at different endpoints.

    `importlib.import_module`, not `from app.audio import windows_loopback`:
    `fake_pyaudiowpatch` drops the module from `sys.modules` but cannot drop
    the attribute the package still holds, so the `from ... import` form hands
    back the stale module object and the patch lands on nothing.
    """
    module = importlib.import_module("app.audio.windows_loopback")

    names = {"communications": "Speakers", "console": "Speakers"}
    monkeypatch.setattr(module, "render_endpoint_names", lambda: dict(names))
    return names


def test_windows_source_opens_the_loopback_endpoint_at_its_native_format(
    fake_pyaudiowpatch, render_endpoints
):
    from app.audio.windows_loopback import WindowsLoopbackSource

    settings = AudioSettings()
    source = WindowsLoopbackSource(settings)
    source.start(lambda arrival, mono: None)

    assert source.native_sample_rate == 48000
    kwargs = fake_pyaudiowpatch.PyAudio.instances[-1].opened_kwargs
    assert kwargs["input"] is True
    assert kwargs["input_device_index"] == 7
    assert kwargs["rate"] == 48000
    assert kwargs["channels"] == 2
    assert kwargs["frames_per_buffer"] == settings.meeting_block_frames
    assert kwargs["format"] == fake_pyaudiowpatch.paFloat32


def test_windows_source_downmixes_interleaved_stereo_to_mono(
    fake_pyaudiowpatch, render_endpoints
):
    """The only work the realtime callback does — and it must average the
    channels, not read one and discard the other."""
    from app.audio.windows_loopback import WindowsLoopbackSource

    source = WindowsLoopbackSource(AudioSettings())
    received: list[np.ndarray] = []
    source.start(lambda arrival, mono: received.append(mono))

    interleaved = np.array([1.0, 0.0, 0.5, 0.5, -1.0, 1.0], dtype=np.float32)
    source._stream_callback(interleaved.tobytes(), 3, None, 0)

    assert len(received) == 1
    assert received[0].tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_windows_source_delivers_nothing_after_stop(fake_pyaudiowpatch, render_endpoints):
    from app.audio.windows_loopback import WindowsLoopbackSource

    source = WindowsLoopbackSource(AudioSettings())
    received: list[np.ndarray] = []
    source.start(lambda arrival, mono: received.append(mono))
    source.stop()

    source._stream_callback(np.zeros(4, dtype=np.float32).tobytes(), 2, None, 0)

    assert received == []
    assert fake_pyaudiowpatch.PyAudio.instances[-1].terminated is True


def test_windows_source_captures_the_communications_endpoint(
    fake_pyaudiowpatch, render_endpoints
):
    """AC: a headset set as the Default Communication Device is what a Teams
    call plays through, and it is what gets captured."""
    from app.audio.windows_loopback import WindowsLoopbackSource

    render_endpoints["communications"] = "Headset"
    render_endpoints["console"] = "Speakers"
    fake_pyaudiowpatch.PyAudio.get_loopback_device_info_generator = lambda self: iter(
        [
            {
                "index": 7,
                "name": "Speakers [Loopback]",
                "defaultSampleRate": 48000.0,
                "maxInputChannels": 2,
            },
            {
                "index": 11,
                "name": "Headset [Loopback]",
                "defaultSampleRate": 44100.0,
                "maxInputChannels": 2,
            },
        ]
    )

    source = WindowsLoopbackSource(AudioSettings())

    assert source.endpoint_name == "Headset [Loopback]"
    assert source.native_sample_rate == 44100


def test_windows_source_honours_the_configured_role_preference(
    fake_pyaudiowpatch, render_endpoints
):
    """The escape hatch for a conferencing app that renders to the console
    endpoint is a configuration key, not a code change."""
    from app.audio.windows_loopback import WindowsLoopbackSource

    render_endpoints["communications"] = "Headset"
    render_endpoints["console"] = "Speakers"
    fake_pyaudiowpatch.PyAudio.get_loopback_device_info_generator = lambda self: iter(
        [
            {
                "index": 7,
                "name": "Speakers [Loopback]",
                "defaultSampleRate": 48000.0,
                "maxInputChannels": 2,
            },
            {
                "index": 11,
                "name": "Headset [Loopback]",
                "defaultSampleRate": 44100.0,
                "maxInputChannels": 2,
            },
        ]
    )

    source = WindowsLoopbackSource(AudioSettings(meeting_system_endpoint_role="console"))

    assert source.endpoint_name == "Speakers [Loopback]"


def test_windows_source_raises_and_releases_when_no_loopback_exists(
    fake_pyaudiowpatch, render_endpoints
):
    """A machine with no loopback endpoint must not leak a PyAudio instance."""
    from app.audio.windows_loopback import WindowsLoopbackSource

    fake_pyaudiowpatch.PyAudio.get_loopback_device_info_generator = lambda self: iter(())

    with pytest.raises(SystemAudioUnavailableError):
        WindowsLoopbackSource(AudioSettings())

    assert fake_pyaudiowpatch.PyAudio.instances[-1].terminated is True


def test_windows_source_releases_pyaudio_when_the_com_lookup_fails(
    fake_pyaudiowpatch, monkeypatch
):
    """A COM failure must not leave a PortAudio instance holding the device."""
    windows_loopback = importlib.import_module("app.audio.windows_loopback")

    def _explode():
        raise SystemAudioUnavailableError("CoInitializeEx failed with 0x80004005")

    monkeypatch.setattr(windows_loopback, "render_endpoint_names", _explode)

    with pytest.raises(SystemAudioUnavailableError):
        windows_loopback.WindowsLoopbackSource(AudioSettings())

    assert fake_pyaudiowpatch.PyAudio.instances[-1].terminated is True


@pytest.fixture(autouse=True)
def _acknowledged_meeting_consent():
    """The disclosure is a first-run gate, not the subject of most of these
    tests — the ones that are unset it explicitly."""
    from app.preferences import user_settings

    user_settings.update_user_settings({"meeting_consent_acknowledged": True})


class _FakeRecorder:
    def __init__(self, is_recording: bool = False, is_busy: bool | None = None):
        self.is_recording = is_recording
        self.is_busy = is_recording if is_busy is None else is_busy
        self.duration_seconds = 0.0
        self.level_db = float("-inf")
        self.system_level_db = float("-inf")
        self.system_endpoint = None
        self.started = False

    async def start(self):
        self.started = True
        self.is_recording = True
        self.is_busy = True

    async def stop(self):
        self.is_recording = False
        self.is_busy = False
        raise AssertionError("this stub is not expected to stop")


@pytest.mark.anyio
async def test_meeting_start_returns_501_where_no_system_source_exists(client):
    """AC: the endpoint answers 501 naming the platform limitation."""

    class _Unavailable(_FakeRecorder):
        async def start(self):
            raise SystemAudioUnavailableError(
                "System audio capture is not available on this platform"
            )

    app.dependency_overrides[get_meeting_recorder] = lambda: _Unavailable()

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 501
    assert "platform" in resp.json()["detail"]


@pytest.mark.anyio
async def test_meeting_start_is_409_while_dictation_is_recording(client):
    """AC: the microphone is never opened twice."""
    meeting = _FakeRecorder()
    app.dependency_overrides[get_meeting_recorder] = lambda: meeting
    app.dependency_overrides[get_active_recorder] = lambda: _FakeRecorder(is_recording=True)

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 409
    assert meeting.started is False


@pytest.mark.anyio
async def test_audio_start_is_409_while_a_meeting_is_recording(client):
    """AC: the mirror-image guard."""
    from app.audio import get_active_meeting_recorder, get_recorder

    dictation = _FakeRecorder()
    app.dependency_overrides[get_recorder] = lambda: dictation
    app.dependency_overrides[get_active_meeting_recorder] = lambda: _FakeRecorder(
        is_recording=True
    )

    resp = await client.post("/audio/start")

    assert resp.status_code == 409
    assert dictation.started is False


@pytest.mark.anyio
async def test_meeting_start_is_409_when_already_recording(client):
    app.dependency_overrides[get_meeting_recorder] = lambda: _FakeRecorder(is_recording=True)

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 409


@pytest.mark.anyio
async def test_meeting_stop_without_start_is_409(client):
    app.dependency_overrides[get_meeting_recorder] = lambda: _FakeRecorder()

    resp = await client.post("/audio/meeting/stop")

    assert resp.status_code == 409


@pytest.mark.anyio
async def test_meeting_status_reports_idle(client):
    app.dependency_overrides[get_meeting_recorder] = lambda: _FakeRecorder()

    resp = await client.get("/audio/meeting/status")

    assert resp.status_code == 200
    assert resp.json()["is_recording"] is False


@pytest.mark.anyio
async def test_meeting_stop_reports_the_filename_and_truncation(client, tmp_path):
    class _Stopping(_FakeRecorder):
        async def stop(self):
            self.is_recording = False
            path = tmp_path / "meeting_abc123.wav"
            path.write_bytes(b"")
            return MeetingRecording(path=path, duration_seconds=12.5, truncated=True)

    app.dependency_overrides[get_meeting_recorder] = lambda: _Stopping(is_recording=True)

    resp = await client.post("/audio/meeting/stop")

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "meeting_abc123.wav"
    assert body["truncated"] is True


@pytest.mark.anyio
async def test_instant_prompt_stop_response_has_no_meeting_fields(client, tmp_path):
    """The Instant Prompt response model must stay exactly as it was — the
    frontend parses it, and spec 066 ships no frontend change."""
    from app.audio import get_recorder

    class _Stopping(_FakeRecorder):
        def __init__(self):
            super().__init__(is_recording=True)

        async def stop(self):
            self.is_recording = False
            path = tmp_path / "rec_abc123.wav"
            path.write_bytes(b"")
            return path

    app.dependency_overrides[get_recorder] = lambda: _Stopping()

    resp = await client.post("/audio/stop")

    assert resp.status_code == 200
    assert set(resp.json()) == {"filename", "duration_seconds"}


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/audio/status", "/audio/start"])
async def test_the_dictation_status_shape_did_not_move(client, path):
    """AC: RecordingStatus has exactly the fields it has today."""
    from app.audio import get_recorder

    app.dependency_overrides[get_recorder] = lambda: _FakeRecorder()

    resp = await client.request("POST" if path == "/audio/start" else "GET", path)

    assert resp.status_code == 200
    assert set(resp.json()) == {"is_recording", "duration_seconds", "level_db"}


@pytest.mark.anyio
async def test_the_meeting_status_reports_the_endpoint_and_the_system_level(client):
    """AC: the meeting responses grew by exactly two fields."""
    recorder = _FakeRecorder()
    recorder.system_endpoint = "Headset [Loopback]"
    recorder.system_level_db = -21.5
    app.dependency_overrides[get_meeting_recorder] = lambda: recorder

    resp = await client.get("/audio/meeting/status")

    assert resp.status_code == 200
    assert set(resp.json()) == {
        "is_recording",
        "duration_seconds",
        "level_db",
        "system_endpoint",
        "system_level_db",
    }
    assert resp.json()["system_endpoint"] == "Headset [Loopback]"
    assert resp.json()["system_level_db"] == -21.5


@pytest.mark.anyio
async def test_meeting_start_is_403_until_the_disclosure_is_acknowledged(client):
    """AC: the 403 is what makes the dialog impossible to drive around with
    curl, and no capture source is created."""
    from app.preferences import user_settings

    user_settings.update_user_settings({"meeting_consent_acknowledged": False})
    recorder = _FakeRecorder()
    app.dependency_overrides[get_meeting_recorder] = lambda: recorder

    with patch("app.audio.meeting_recorder.create_system_audio_source") as factory:
        resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 403
    assert recorder.started is False
    factory.assert_not_called()


@pytest.mark.anyio
async def test_meeting_start_is_never_403_once_acknowledged(client):
    app.dependency_overrides[get_meeting_recorder] = lambda: _FakeRecorder()

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 200


@pytest.mark.anyio
async def test_meeting_stop_and_status_are_not_behind_the_consent_gate(client):
    """A recording already running must always be stoppable and visible."""
    from app.preferences import user_settings

    user_settings.update_user_settings({"meeting_consent_acknowledged": False})
    app.dependency_overrides[get_meeting_recorder] = lambda: _FakeRecorder()

    assert (await client.get("/audio/meeting/status")).status_code == 200
    assert (await client.post("/audio/meeting/stop")).status_code == 409


@pytest.mark.asyncio
async def test_the_system_half_gets_its_own_level_meter(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """AC: a meeting recording that captured only the microphone looks
    identical from outside to one that works — this is what separates them."""
    recorder = MeetingRecorder(audio_settings)

    await recorder.start()
    assert recorder.system_level_db == float("-inf")

    fake_system_source.deliver(recorder._start_time, fill=0.5)

    assert recorder.system_level_db > -12.0
    assert recorder.level_db == float("-inf")

    recorder.cleanup()


@pytest.mark.asyncio
async def test_the_recorder_reports_the_endpoint_the_source_chose(
    audio_settings, fake_system_source, fake_microphone_stream
):
    recorder = MeetingRecorder(audio_settings)

    assert recorder.system_endpoint is None

    await recorder.start()

    assert recorder.system_endpoint == "Headset [Loopback]"

    recorder.cleanup()
    _wait_for_devices(recorder)

    assert recorder.system_endpoint is None


@pytest.mark.asyncio
async def test_start_surfaces_the_real_device_failure(audio_settings, fake_microphone_stream):
    """JS-78: a device that failed says why, instead of naming the platform.

    The factory used to swallow every construction failure into None, so the
    caller's "requires Windows or macOS" reached a user already on Windows --
    a message that is both false and unactionable.
    """
    recorder = MeetingRecorder(audio_settings)
    failure = SystemAudioUnavailableError("no loopback device on the default render endpoint")

    with patch("app.audio.meeting_recorder.create_system_audio_source", side_effect=failure):
        with pytest.raises(SystemAudioUnavailableError, match="default render endpoint"):
            await recorder.start()

    assert recorder.is_recording is False
    fake_microphone_stream.assert_not_called()


@pytest.mark.anyio
async def test_meeting_start_501_carries_the_device_reason(client):
    """The 501 body is what the user actually reads (JS-78)."""

    class _Broken:
        is_recording = False
        is_busy = False

        async def start(self):
            raise SystemAudioUnavailableError(
                "System audio capture could not be opened on this machine: "
                "no loopback device on the default render endpoint"
            )

    app.dependency_overrides[get_meeting_recorder] = lambda: _Broken()

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 501
    assert "default render endpoint" in resp.json()["detail"]
    assert "requires Windows or macOS" not in resp.json()["detail"]


@pytest.mark.asyncio
async def test_meeting_stop_leaves_the_event_loop_free(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """JS-81: two resamples, a mix and a wave write must not run on the loop.

    This is the headline case -- a 45-minute call is ~86 MB written with every
    other endpoint blocked behind it, including `/health` and the meeting
    status the widget polls twice a second.

    The assemble is slowed by a known interval and the floor set proportional
    to it. The original `ticks > 0` was satisfied by a single bare
    `await asyncio.sleep(0)` in front of the inline call, which leaves every
    millisecond of the write on the loop -- JS-97.
    """
    recorder = MeetingRecorder(audio_settings)
    await recorder.start()
    _feed_microphone(recorder, 40)
    for i in range(40):
        fake_system_source.deliver(recorder._start_time + i * BLOCK_FRAMES / 48000)

    blocked_seconds = 0.2
    assemble_and_write = recorder._assemble_and_write

    def slow_assemble_and_write(*args):
        time.sleep(blocked_seconds)
        return assemble_and_write(*args)

    recorder._assemble_and_write = slow_assemble_and_write

    ticks = 0

    async def competitor():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    race = asyncio.ensure_future(competitor())
    audio_path = (await recorder.stop()).path
    race.cancel()

    assert audio_path.exists()
    assert ticks > 100, (
        f"the loop ticked {ticks} times while stop() blocked for "
        f"{blocked_seconds}s -- the assemble and write are still on the event loop"
    )


@pytest.mark.asyncio
async def test_start_leaves_the_event_loop_free(audio_settings, fake_microphone_stream):
    """JS-99: the Windows COM enumeration and the macOS helper handshake are
    seconds of work, and neither may run on the event loop.

    The factory is slowed by a known interval and the floor set proportional
    to it, the same shape `test_meeting_stop_leaves_the_event_loop_free`
    uses: `ticks > 0` is satisfied by one bare `await asyncio.sleep(0)` in
    front of an inline open, which leaves every millisecond of it on the loop.
    """
    blocked_seconds = 0.2
    source = _FakeSystemAudioSource()

    def slow_factory(settings):
        time.sleep(blocked_seconds)
        return source

    recorder = MeetingRecorder(audio_settings)
    ticks = 0

    async def competitor():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    with patch("app.audio.meeting_recorder.create_system_audio_source", slow_factory):
        race = asyncio.ensure_future(competitor())
        await recorder.start()
        race.cancel()

    recorder.cleanup()
    _wait_for_devices(recorder)

    assert ticks > 100, (
        f"the loop ticked {ticks} times while start() blocked for "
        f"{blocked_seconds}s — the device open is still on the event loop"
    )


@pytest.mark.asyncio
async def test_every_device_call_happens_on_one_thread_that_is_not_the_loop(
    audio_settings, blocking_system_source, recording_microphone_stream
):
    """ADR 048: one thread owns every meeting device handle.

    PortAudio's WASAPI host API initialises COM on the calling thread, so an
    instance created on one thread and terminated on another splits an
    apartment — a Windows-only failure that would never reproduce on CI. The
    thread identity is the cheapest available alarm for it.
    """
    blocking_system_source.release_open.set()
    recorder = MeetingRecorder(audio_settings)

    await recorder.start()
    _feed_microphone(recorder, 4)
    await recorder.stop()

    device_idents = set(blocking_system_source.idents) | set(
        recording_microphone_stream.idents
    )

    assert len(device_idents) == 1, (
        f"the device calls ran on {len(device_idents)} threads: "
        f"{blocking_system_source.calls} / {recording_microphone_stream.calls}"
    )
    assert threading.get_ident() not in device_idents, (
        "a device call ran on the event-loop thread"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_system_source", ["construction", "start"], indirect=True)
async def test_a_stop_during_the_open_releases_what_the_open_created(
    audio_settings, blocking_system_source, recording_microphone_stream
):
    """A stop arriving mid-open must not reach handles that do not exist yet.

    On PR #80 it did: the release ran before the open had assigned anything,
    so `source.stop()` was observed before `source.start()` and the
    microphone stream was left open with `is_recording` false.

    The stop is a command on the same queue, so it is answered *after* the
    open it arrived behind: the start succeeds, and the stop finds a capture
    with no audio in it.
    """
    recorder = MeetingRecorder(audio_settings)
    starting = asyncio.ensure_future(recorder.start())
    await asyncio.to_thread(blocking_system_source.opening_started.wait, 5.0)

    stopping = asyncio.ensure_future(recorder.stop())
    await asyncio.sleep(0)

    blocking_system_source.release_open.set()
    await starting
    with pytest.raises(MeetingCaptureAbortedError, match="No audio data captured"):
        await stopping
    _wait_for_devices(recorder)

    assert recorder.is_recording is False
    assert recorder.is_busy is False
    assert blocking_system_source.stopped is True
    assert blocking_system_source.calls.index("start") < blocking_system_source.calls.index(
        "stop"
    ), f"the source was stopped before it was started: {blocking_system_source.calls}"
    assert [stream.closes for stream in recording_microphone_stream.streams] == [1]


@pytest.mark.asyncio
async def test_cleanup_during_the_open_returns_at_once_and_leaks_nothing(
    audio_settings, blocking_system_source, recording_microphone_stream
):
    """`cleanup()` runs on the event loop from the lifespan drain, so it
    submits the release and returns rather than waiting for it.

    The discard is a command queued behind the open, so the start it raced
    still completes — and is then torn down, which is what "leaks nothing"
    means here.
    """
    recorder = MeetingRecorder(audio_settings)
    starting = asyncio.ensure_future(recorder.start())
    await asyncio.to_thread(blocking_system_source.opening_started.wait, 5.0)

    threading.Timer(0.5, blocking_system_source.release_open.set).start()
    began = time.monotonic()
    recorder.cleanup()
    elapsed = time.monotonic() - began

    await starting
    _wait_for_devices(recorder)

    assert elapsed < 0.1, (
        f"cleanup() took {elapsed:.3f}s against an open blocked for 0.5s — it "
        f"waited for the release instead of submitting it"
    )
    assert blocking_system_source.stopped is True
    assert [stream.closes for stream in recording_microphone_stream.streams] == [1]


@pytest.mark.asyncio
async def test_a_cancelled_start_leaves_no_device_open(
    audio_settings, blocking_system_source, recording_microphone_stream
):
    """Awaiting a thread does not cancel it, so cancelling `start()` cannot
    stop the open — the open's own publish check is what prevents the leak."""
    recorder = MeetingRecorder(audio_settings)
    starting = asyncio.ensure_future(recorder.start())
    await asyncio.to_thread(blocking_system_source.opening_started.wait, 5.0)

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting

    blocking_system_source.release_open.set()
    _wait_for_devices(recorder)

    assert recorder.is_busy is False
    assert blocking_system_source.stopped is True
    assert [stream.closes for stream in recording_microphone_stream.streams] == [1]


@pytest.mark.asyncio
async def test_a_slow_open_adds_no_leading_silence_to_the_wav(
    audio_settings, fake_microphone_stream
):
    """The clock starts when capture is live, not when the open began.

    Neither assertion reads `recorder._start_time`: every other test in this
    module delivers blocks at `recorder._start_time + ...` and is therefore
    anchored to the value under test, which is why a 2 s open added 2 s of
    leading silence to every meeting WAV on PR #80 with the suite green. The
    span this test compares against is the one it measures itself.
    """
    open_block_seconds = 0.5
    source = _FakeSystemAudioSource()

    def slow_factory(settings):
        time.sleep(open_block_seconds)
        return source

    recorder = MeetingRecorder(audio_settings)
    with patch("app.audio.meeting_recorder.create_system_audio_source", slow_factory):
        await recorder.start()

    started_at = time.monotonic()
    for i in range(24):
        source.deliver(started_at + i * BLOCK_FRAMES / 48000, fill=0.5)
    time.sleep(0.55)
    stopped_at = time.monotonic()
    audio_path = (await recorder.stop()).path

    measured_span = stopped_at - started_at
    with wave.open(str(audio_path), "rb") as wf:
        rate = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    normalized = samples.astype(np.float32) / 32768.0
    duration = len(normalized) / rate
    audible = np.flatnonzero(np.abs(normalized) > 10 ** (-40 / 20))

    assert abs(duration - measured_span) < 0.15, (
        f"the WAV is {duration:.3f}s long against a measured span of "
        f"{measured_span:.3f}s — the {open_block_seconds}s open is in the file"
    )
    assert audible.size > 0
    assert audible[0] / rate < 0.15, (
        f"the first audible sample is at {audible[0] / rate:.3f}s — the open "
        f"was written into the file as leading silence"
    )


@pytest.mark.asyncio
async def test_a_second_start_during_the_open_opens_nothing(
    audio_settings, recording_microphone_stream
):
    """Whether a start may proceed is answered on the thread that owns it.

    Both starts are commands on one queue, so the second runs after the
    first has published and is refused by the recorder itself rather than by
    a check the loop made before submitting.
    """
    source = _BlockingSystemAudioSource()
    constructions = 0

    def counting_factory(settings):
        nonlocal constructions
        constructions += 1
        return source.construct(settings)

    recorder = MeetingRecorder(audio_settings)
    with patch("app.audio.meeting_recorder.create_system_audio_source", counting_factory):
        first = asyncio.ensure_future(recorder.start())
        second = asyncio.ensure_future(recorder.start())
        await asyncio.to_thread(source.opening_started.wait, 5.0)
        source.release_open.set()
        await first
        with pytest.raises(MeetingCaptureAbortedError):
            await second

    recorder.cleanup()
    _wait_for_devices(recorder)

    assert constructions == 1, f"{constructions} system sources were opened, not 1"
    assert len(recording_microphone_stream.streams) == 1


def test_the_device_thread_runs_commands_in_submission_order(audio_settings):
    """A command queued behind a running one never overtakes it.

    ADR 048 names the executor's FIFO order a deliberate dependency rather
    than an accident: every release that is submitted and not awaited is
    correct only because it cannot run before the command already occupying
    the worker. Nothing else in this suite fails when that stops being true —
    the rows that describe the ordering are refused by a state guard first,
    so they go red without ever reaching their order assertion.

    Both commands are submitted straight to the executor: this pins the queue
    itself, not any recorder state layered on top of it.
    """
    recorder = MeetingRecorder(audio_settings)
    order: list[str] = []
    occupying = threading.Event()
    release = threading.Event()

    def occupy_the_worker():
        occupying.set()
        release.wait(5.0)
        order.append("queued first")

    recorder._devices.submit(occupy_the_worker)
    assert occupying.wait(5.0), "the device thread never picked up the first command"
    recorder._devices.submit(order.append, "queued second")
    release.set()
    _wait_for_devices(recorder)

    assert order == ["queued first", "queued second"], (
        f"the device thread ran its commands as {order}, out of submission order"
    )


@pytest.mark.asyncio
async def test_a_start_arriving_during_the_release_cannot_take_the_recorder(
    audio_settings, recording_microphone_stream
):
    """A finished meeting is not free for the taking while its devices close.

    `stop()` holds both handles for the whole release — up to 1.5 s on macOS.
    Reporting the recorder idle there let a second start pass the guard, reset
    both block lists and cost the caller the meeting that had just ended.

    The racing start is not refused: it is queued behind the release and
    opens its own devices after it. So the assertion is an *order*, taken
    from the one list every device call on this source appends to — which is
    what pins ADR 048's "one worker, therefore one order".
    """
    source = _BlockingSystemAudioSource(block_on="stop")

    def counting_factory(settings):
        return source.construct(settings)

    source.release_open.set()
    recorder = MeetingRecorder(audio_settings)
    with patch("app.audio.meeting_recorder.create_system_audio_source", counting_factory):
        await recorder.start()
        expected_frames = _deliver_over_a_real_span(recorder, source, 6)
        _feed_microphone(recorder, 6)
        await asyncio.sleep(0.2)

        stopping = asyncio.ensure_future(recorder.stop())
        await asyncio.to_thread(source.closing_started.wait, 5.0)

        assert recorder.is_busy is True, (
            "the recorder reported itself free while it still held both devices"
        )
        assert recorder.is_recording is False

        racing = asyncio.ensure_future(recorder.start())
        await asyncio.sleep(0)
        source.release_close.set()

        audio_path = (await stopping).path
        await racing

    recorder.cleanup()
    _wait_for_devices(recorder)

    assert source.calls == [
        "construct",
        "start",
        "stop",
        "construct",
        "start",
        "stop",
    ], (
        f"the racing start's open interleaved with the release: {source.calls}"
    )
    assert [stream.closes for stream in recording_microphone_stream.streams] == [1, 1]
    with wave.open(str(audio_path), "rb") as wf:
        assert wf.getnframes() >= expected_frames


@pytest.mark.asyncio
async def test_a_cleanup_racing_the_release_does_not_cost_the_finished_meeting(
    audio_settings, recording_microphone_stream
):
    """The blocks come out in the hold that ends the capture, not after the release.

    Anything that resets the buffers during the release — a shutdown drain
    here, a racing start in the test above — would otherwise leave `stop()`
    raising `No audio data captured` for a meeting that was really captured.
    """
    source = _BlockingSystemAudioSource(block_on="stop")
    source.release_open.set()
    recorder = MeetingRecorder(audio_settings)
    delivered = 6

    with patch("app.audio.meeting_recorder.create_system_audio_source", source.construct):
        await recorder.start()
        expected_frames = _deliver_over_a_real_span(recorder, source, delivered)
        _feed_microphone(recorder, delivered)
        await asyncio.sleep(0.2)

        stopping = asyncio.ensure_future(recorder.stop())
        await asyncio.to_thread(source.closing_started.wait, 5.0)

        recorder.cleanup()
        source.release_close.set()
        audio_path = (await stopping).path

    _wait_for_devices(recorder)

    with wave.open(str(audio_path), "rb") as wf:
        frames = wf.getnframes()

    assert frames >= expected_frames, (
        f"the WAV holds {frames} frames against {delivered} blocks delivered — "
        f"the buffers were harvested after the release"
    )



@pytest.mark.asyncio
async def test_every_state_transition_happens_on_the_device_thread(
    audio_settings, blocking_system_source, recording_microphone_stream
):
    """ADR 048, amended: the owner thread owns the state, not only the handles.

    Five of the six lifecycle transitions used to run on the event loop while
    the handles they describe lived on another thread, and every defect three
    review rounds produced sat on that seam. This is the alarm for a
    transition drifting back onto the loop.
    """
    recorder = MeetingRecorder(audio_settings)
    transition_idents: list[int] = []
    transition = recorder._transition

    def recording_transition(state):
        transition_idents.append(threading.get_ident())
        transition(state)

    recorder._transition = recording_transition

    blocking_system_source.release_open.set()
    await recorder.start()
    _feed_microphone(recorder, 4)
    _deliver_over_a_real_span(recorder, blocking_system_source, 4)
    await asyncio.sleep(0.05)
    await recorder.stop()

    blocking_system_source.opening_started.clear()
    blocking_system_source.release_open.clear()
    starting = asyncio.ensure_future(recorder.start())
    await asyncio.to_thread(blocking_system_source.opening_started.wait, 5.0)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    blocking_system_source.release_open.set()
    _wait_for_devices(recorder)

    recorder.cleanup()
    _wait_for_devices(recorder)

    device_idents = set(blocking_system_source.idents) | set(
        recording_microphone_stream.idents
    )

    assert len(set(transition_idents)) == 1, (
        f"the lifecycle was written from {len(set(transition_idents))} threads"
    )
    assert set(transition_idents) == device_idents, (
        "the state was written from a thread other than the one holding the devices"
    )
    assert threading.get_ident() not in set(transition_idents), (
        "a lifecycle transition ran on the event-loop thread"
    )


@pytest.mark.asyncio
async def test_the_recorder_is_busy_from_the_moment_a_start_is_submitted(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """A queued command leaves the state untouched, and `is_busy` must still
    say so — `router.py`'s dictation guard is the caller that pays for a
    false negative, with a second stream on the same microphone."""
    recorder = MeetingRecorder(audio_settings)
    gate = _hold_the_owner_thread(recorder)

    starting = asyncio.ensure_future(recorder.start())
    await asyncio.sleep(0)

    assert recorder.is_busy is True, (
        "the recorder reported itself free while a start was already queued"
    )
    assert recorder.is_recording is False

    gate.set()
    await starting
    recorder.cleanup()
    _wait_for_devices(recorder)

    assert recorder.is_busy is False


@pytest.mark.asyncio
async def test_an_abandoned_start_cannot_tear_down_a_later_recording(
    audio_settings, recording_microphone_stream
):
    """An abandonment names the capture it may tear down.

    A start that fails sends its abandonment behind whatever is already
    queued, so it can reach the recorder after a *different* start has
    claimed and published it. The session token is what stops it there.
    """
    source_b = _FakeSystemAudioSource(endpoint_name="Second [Loopback]")
    attempts = 0

    def factory(settings):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("the first device is gone")
        return source_b

    recorder = MeetingRecorder(audio_settings)
    with patch("app.audio.meeting_recorder.create_system_audio_source", factory):
        first = asyncio.ensure_future(recorder.start())
        second = asyncio.ensure_future(recorder.start())
        with pytest.raises(OSError):
            await first
        await second

    _wait_for_devices(recorder)

    assert recorder.is_recording is True, (
        "the failed start's abandonment tore down the recording that replaced it"
    )
    assert recorder.system_endpoint == "Second [Loopback]"
    assert source_b.stopped is False
    assert [stream.closes for stream in recording_microphone_stream.streams] == [0]

    recorder.cleanup()
    _wait_for_devices(recorder)


@pytest.mark.asyncio
async def test_a_cancelled_stop_still_returns_the_recorder_to_idle(
    audio_settings, recording_microphone_stream
):
    """FastAPI cancels the endpoint task when a client disconnects, and a
    stop that never finishes returning to idle would leave every later start
    answering 409 for the rest of the process."""
    source = _BlockingSystemAudioSource(block_on="stop")
    source.release_open.set()
    recorder = MeetingRecorder(audio_settings)

    with patch("app.audio.meeting_recorder.create_system_audio_source", source.construct):
        await recorder.start()
        _feed_microphone(recorder, 4)
        _deliver_over_a_real_span(recorder, source, 4)
        await asyncio.sleep(0.05)

        stopping = asyncio.ensure_future(recorder.stop())
        await asyncio.to_thread(source.closing_started.wait, 5.0)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping

        source.release_close.set()
        _wait_for_devices(recorder)

    assert recorder.is_busy is False, (
        "the cancelled stop left the recorder permanently busy"
    )
    assert source.stopped is True
    assert [stream.closes for stream in recording_microphone_stream.streams] == [1]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        ("router-not-recording", 409),
        ("recorder-not-recording", 409),
        ("no-audio", 410),
    ],
)
async def test_every_refused_meeting_stop_leaves_nothing_recording(
    client,
    audio_settings,
    fake_system_source,
    fake_microphone_stream,
    outcome,
    expected_status,
):
    """The widget takes its indicator down on both refusals, so every one of
    them has to mean nothing is being recorded.

    The two codes are not two wordings: 409 is a stop that found nothing
    running, 410 is a meeting that ran and captured nothing. Nothing links
    `src/widget/meeting-toggle.ts` to this router, which is why the backend
    half of that contract is pinned here rather than left as prose on the
    other side of a process boundary.
    """
    recorder = MeetingRecorder(audio_settings)
    app.dependency_overrides[get_meeting_recorder] = lambda: recorder

    if outcome == "recorder-not-recording":
        _hold_the_owner_thread_until_queued(recorder, 2)
        recorder._submit_on_devices(lambda: None)
    elif outcome == "no-audio":
        await recorder.start()

    resp = await client.post("/audio/meeting/stop")

    assert resp.status_code == expected_status, resp.json()
    assert recorder.is_recording is False, (
        f"the {outcome} refusal was answered while the recorder was still recording"
    )

    recorder.cleanup()
    _wait_for_devices(recorder)


@pytest.mark.anyio
async def test_dictation_start_is_409_while_a_meeting_start_is_in_flight(client):
    """AC: the mutual-exclusion guards ask whether the devices are spoken
    for, which is true for the whole open — `is_recording` is not."""
    from app.audio import get_active_meeting_recorder, get_recorder

    dictation = _FakeRecorder()
    app.dependency_overrides[get_recorder] = lambda: dictation
    app.dependency_overrides[get_active_meeting_recorder] = lambda: _FakeRecorder(
        is_recording=False, is_busy=True
    )

    resp = await client.post("/audio/start")

    assert resp.status_code == 409
    assert dictation.started is False

    meeting = _FakeRecorder(is_recording=False, is_busy=True)
    app.dependency_overrides[get_meeting_recorder] = lambda: meeting

    second = await client.post("/audio/meeting/start")

    assert second.status_code == 409
    assert meeting.started is False


@pytest.mark.anyio
async def test_meeting_stop_during_the_open_is_409_not_500(client):
    """A stop with no file to return is the caller's situation, not a crash.

    This also closes a hole that predates JS-99 on the same line: a stop that
    captured nothing raised a bare RuntimeError and reached the client as 500.
    """

    class _Opening(_FakeRecorder):
        def __init__(self):
            super().__init__(is_recording=False, is_busy=True)

        async def stop(self):
            self.is_busy = False
            raise MeetingCaptureAbortedError(
                "The meeting recording was still opening its devices"
            )

    app.dependency_overrides[get_meeting_recorder] = lambda: _Opening()

    resp = await client.post("/audio/meeting/stop")

    assert resp.status_code == 409
    assert "opening its devices" in resp.json()["detail"]


@pytest.mark.anyio
async def test_a_stop_cancelled_while_still_queued_releases_the_devices_anyway(
    audio_settings, fake_system_source, recording_microphone_stream
):
    """AC: an abandoned stop request still releases both devices.

    Reloading the widget window is a client disconnect, and Starlette
    cancels the endpoint task on one. A queued release that the cancel
    withdraws leaves the microphone and the render endpoint open while the
    recorder reports itself idle.
    """
    recorder = MeetingRecorder(audio_settings)
    await recorder.start()
    _feed_microphone(recorder, 4)
    _deliver_over_a_real_span(recorder, fake_system_source, 4)
    await asyncio.sleep(0.05)

    gate = _hold_the_owner_thread(recorder)
    stopping = asyncio.ensure_future(recorder.stop())
    await asyncio.sleep(0)

    assert fake_system_source.stopped is False, (
        "the release ran before the cancel — this row is about a queued command"
    )
    assert recorder._devices_in_flight == 1

    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    await asyncio.sleep(0.05)

    gate.set()
    _wait_for_devices(recorder)

    assert fake_system_source.stopped is True
    assert [stream.closes for stream in recording_microphone_stream.streams] == [1]
    assert recorder.is_recording is False
    assert recorder.is_busy is False
    assert recorder._devices_in_flight == 0


@pytest.mark.anyio
async def test_a_start_cancelled_while_still_queued_opens_and_is_then_torn_down(
    audio_settings, recording_microphone_stream
):
    """AC: no abandoned request leaves the recorder permanently spoken for.

    A start whose command the cancel withdraws never decrements the
    in-flight counter, and `is_busy` then answers every later recording
    request with a 409 until the app restarts.
    """
    source = _FakeSystemAudioSource()
    constructions: list[object] = []

    def _construct(settings):
        constructions.append(settings)
        return source

    recorder = MeetingRecorder(audio_settings)
    with patch("app.audio.meeting_recorder.create_system_audio_source", _construct):
        gate = _hold_the_owner_thread(recorder)
        starting = asyncio.ensure_future(recorder.start())
        await asyncio.sleep(0)

        assert constructions == [], (
            "the open ran before the cancel — this row is about a queued command"
        )

        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        await asyncio.sleep(0.05)

        gate.set()
        _wait_for_devices(recorder)

    assert len(constructions) == 1
    assert source.stopped is True
    assert [stream.closes for stream in recording_microphone_stream.streams] == [1]
    assert recorder.is_busy is False
    assert recorder._devices_in_flight == 0


@pytest.mark.anyio
async def test_a_cancelled_command_that_fails_reaches_no_exception_handler(
    audio_settings, fake_system_source, recording_microphone_stream
):
    """AC: a detached command that then fails puts nothing on the loop's
    exception handler.

    A stop cancelled at quit on a meeting that captured nothing raises
    `MeetingCaptureAbortedError`, and a future whose exception nobody reads
    reports itself at ERROR with a traceback once it is collected.
    """
    collected: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda loop, context: collected.append(context)
    )
    recorder = MeetingRecorder(audio_settings)
    await recorder.start()

    gate = _hold_the_owner_thread(recorder)
    stopping = asyncio.ensure_future(recorder.stop())
    await asyncio.sleep(0)
    stopping.cancel()
    try:
        await stopping
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.05)

    gate.set()
    _wait_for_devices(recorder)
    await asyncio.sleep(0.05)

    del stopping
    gc.collect()
    await asyncio.sleep(0)

    assert [context.get("message") for context in collected] == []
    assert fake_system_source.stopped is True


@pytest.mark.anyio
async def test_a_cap_hit_while_the_stop_is_queued_is_reported(client, tmp_path):
    """AC: a raw-store cap reached at any point before the capture ended is
    reported, including in the window between the request and the release."""

    class _TruncatingStop(_FakeRecorder):
        def __init__(self):
            super().__init__(is_recording=True)

        async def stop(self):
            self.is_recording = False
            path = tmp_path / "meeting_capped.wav"
            path.write_bytes(b"")
            return MeetingRecording(path=path, duration_seconds=3.0, truncated=True)

    app.dependency_overrides[get_meeting_recorder] = lambda: _TruncatingStop()

    resp = await client.post("/audio/meeting/stop")

    assert resp.status_code == 200
    assert resp.json()["truncated"] is True


@pytest.mark.asyncio
async def test_an_abandoned_stop_still_writes_the_meeting(
    audio_settings, recording_microphone_stream
):
    """Every obligation of a capture except answering the request survives a
    client disconnect: the file, its audio, both handles and the idle state.

    ADR 048 lists the six and says which party owns each. The five the
    recorder owns are asserted here in one test, so an obligation added later
    either extends this row or is visibly missing from it.
    """
    source = _BlockingSystemAudioSource()
    source.release_open.set()
    recorder = MeetingRecorder(audio_settings)

    with patch("app.audio.meeting_recorder.create_system_audio_source", source.construct):
        await recorder.start()
        _feed_microphone(recorder, 6)
        _deliver_over_a_real_span(recorder, source, 6)
        await asyncio.sleep(0.2)

        gate = _hold_the_owner_thread(recorder)
        stopping = asyncio.ensure_future(recorder.stop())
        await asyncio.sleep(0)

        assert recorder._devices_in_flight == 1, (
            "the stop was already running, so this proves nothing about a queued one"
        )
        assert source.stopped is False

        stopping.cancel()
        try:
            await stopping
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.05)

        measured_span = time.monotonic() - recorder._start_time
        gate.set()
        _wait_for_devices(recorder)
        _wait_for_writes(recorder)

    written = list(audio_settings.temp_dir.glob("meeting_*.wav"))
    assert len(written) == 1, (
        f"the abandoned stop left {len(written)} meeting files — the capture was "
        f"harvested into an object the cancelled task dropped"
    )

    rate, signal = _wav_signal(written[0])
    duration = len(signal) / rate
    assert abs(duration - measured_span) < 0.15, (
        f"the WAV is {duration:.3f}s long against a measured span of "
        f"{measured_span:.3f}s"
    )
    assert np.max(np.abs(signal)) > 10 ** (-40 / 20), (
        "the file holds a header and no audio"
    )

    assert source.stopped is True
    assert [stream.closes for stream in recording_microphone_stream.streams] == [1]
    assert recorder.is_recording is False
    assert recorder.is_busy is False
    assert recorder._devices_in_flight == 0


@pytest.mark.asyncio
async def test_every_written_meeting_names_its_path_in_the_log(
    audio_settings, fake_system_source, fake_microphone_stream, caplog
):
    """Nothing reads the stop response's filename, so the log line is how a
    meeting recording is found — including one whose requester is gone."""
    recorder = MeetingRecorder(audio_settings)

    with caplog.at_level(logging.INFO, logger="app.audio.meeting_recorder"):
        await recorder.start()
        _feed_microphone(recorder, 4)
        _deliver_over_a_real_span(recorder, fake_system_source, 4)
        await asyncio.sleep(0.1)
        answered = (await recorder.stop()).path

        await recorder.start()
        _feed_microphone(recorder, 4)
        _deliver_over_a_real_span(recorder, fake_system_source, 4)
        await asyncio.sleep(0.1)

        gate = _hold_the_owner_thread(recorder)
        stopping = asyncio.ensure_future(recorder.stop())
        await asyncio.sleep(0)
        stopping.cancel()
        try:
            await stopping
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.05)
        gate.set()
        _wait_for_devices(recorder)
        _wait_for_writes(recorder)

    written = {path.name for path in audio_settings.temp_dir.glob("meeting_*.wav")}
    abandoned = written - {answered.name}
    assert len(abandoned) == 1
    abandoned_name = abandoned.pop()

    named = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert any(answered.name in message for message in named), (
        "the answered meeting was written with nothing in the log naming it"
    )
    assert any(abandoned_name in message for message in named), (
        "the abandoned meeting was written with nothing in the log naming it"
    )
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ], "nothing told the user their stop was answered by no one"


@pytest.mark.asyncio
async def test_the_meeting_file_is_written_off_the_device_thread(
    audio_settings, monkeypatch, recording_microphone_stream
):
    """A 45-minute call is ~86 MB to resample, mix and write, and the device
    thread is the queue every later device command sits in."""
    source = _BlockingSystemAudioSource()
    source.release_open.set()
    recorder = MeetingRecorder(audio_settings)
    writing_idents: list[int] = []
    real_write_wav = importlib.import_module("app.audio.meeting_recorder").write_wav

    def recording_write_wav(*args, **kwargs):
        writing_idents.append(threading.get_ident())
        return real_write_wav(*args, **kwargs)

    monkeypatch.setattr(
        "app.audio.meeting_recorder.write_wav", recording_write_wav
    )

    with patch("app.audio.meeting_recorder.create_system_audio_source", source.construct):
        await recorder.start()
        _feed_microphone(recorder, 4)
        _deliver_over_a_real_span(recorder, source, 4)
        await asyncio.sleep(0.1)
        await recorder.stop()
        _wait_for_devices(recorder)
        _wait_for_writes(recorder)

    assert len(writing_idents) == 1
    assert writing_idents[0] not in set(source.idents), (
        "the write ran on the thread that owns the device handles"
    )
    assert writing_idents[0] != threading.get_ident()


@pytest.mark.anyio
async def test_a_meeting_write_does_not_make_the_recorder_busy(
    client, audio_settings, fake_system_source, fake_microphone_stream, monkeypatch
):
    """The write holds no device, so counting it would refuse every dictation
    start for the length of an 86 MB write."""
    from app.audio import get_active_meeting_recorder, get_recorder

    release_write = threading.Event()
    real_write_wav = importlib.import_module("app.audio.meeting_recorder").write_wav

    def parked_write_wav(*args, **kwargs):
        assert release_write.wait(timeout=5.0), "the test never released the write"
        return real_write_wav(*args, **kwargs)

    monkeypatch.setattr("app.audio.meeting_recorder.write_wav", parked_write_wav)

    recorder = MeetingRecorder(audio_settings)
    dictation = _FakeRecorder()
    app.dependency_overrides[get_active_meeting_recorder] = lambda: recorder
    app.dependency_overrides[get_recorder] = lambda: dictation

    await recorder.start()
    _feed_microphone(recorder, 4)
    _deliver_over_a_real_span(recorder, fake_system_source, 4)
    await asyncio.sleep(0.1)

    stopping = asyncio.ensure_future(recorder.stop())
    await asyncio.sleep(0)
    await asyncio.to_thread(_wait_for_devices, recorder)

    assert recorder.is_busy is False, (
        "the recorder reported itself busy for the whole write"
    )
    assert recorder._devices_in_flight == 0
    assert (await client.post("/audio/start")).status_code == 200

    release_write.set()
    await stopping
    _wait_for_writes(recorder)


@pytest.mark.asyncio
async def test_a_finished_sessions_callback_cannot_charge_the_next_meeting(
    tmp_path, fake_microphone_stream
):
    """A block belongs to the session that registered the callback.

    The macOS reader is abandoned after a bounded half-second join while it
    still holds a live reference to the sink, so a block from a finished
    meeting reaching the next one is a documented outcome of that code rather
    than a preemption hypothesis.
    """
    settings = AudioSettings(
        sample_rate=16000,
        channels=1,
        temp_dir=tmp_path / "tmp",
        meeting_max_raw_bytes=BLOCK_FRAMES * 4,
    )
    source = _FakeSystemAudioSource()
    recorder = MeetingRecorder(settings)

    with patch("app.audio.meeting_recorder.create_system_audio_source", lambda _: source):
        await recorder.start()
        finished_sink = source.on_block
        _feed_microphone(recorder, 1)
        await asyncio.sleep(0.05)
        await recorder.stop()
        _wait_for_devices(recorder)
        _wait_for_writes(recorder)

        await recorder.start()
        finished_sink(time.monotonic(), np.zeros(BLOCK_FRAMES * 4, dtype=np.float32))
        _feed_microphone(recorder, 1)
        await asyncio.sleep(0.05)
        second = await recorder.stop()
        _wait_for_devices(recorder)
        _wait_for_writes(recorder)

    assert second.truncated is False, (
        "a block from the finished meeting was charged to the running one, and "
        "its own next block found the cap already reached"
    )
    assert second.path.exists()


@pytest.mark.anyio
async def test_a_meeting_that_captured_nothing_answers_410_and_a_double_stop_answers_409(
    client, tmp_path
):
    """`MeetingCaptureEmptyError` is a subclass, so its clause has to come
    first — below the base class it would never be reached and the widget
    would call a lost recording a double click."""

    class _Empty(_FakeRecorder):
        def __init__(self):
            super().__init__(is_recording=True)

        async def stop(self):
            self.is_recording = False
            raise MeetingCaptureEmptyError("No audio data captured")

    class _NotRecording(_FakeRecorder):
        def __init__(self):
            super().__init__(is_recording=True)

        async def stop(self):
            self.is_recording = False
            raise MeetingCaptureAbortedError("Not recording")

    app.dependency_overrides[get_meeting_recorder] = _Empty
    assert (await client.post("/audio/meeting/stop")).status_code == 410

    app.dependency_overrides[get_meeting_recorder] = _NotRecording
    assert (await client.post("/audio/meeting/stop")).status_code == 409


def test_the_recorder_abc_declares_only_what_both_recorders_honour():
    """The base class must not advertise a signature a subclass breaks.

    `stop()` used to be declared `-> Path` on `AudioRecorder` while
    `MeetingRecorder.stop()` answers a `MeetingRecording`, so code written
    against the abstraction — `path = await recorder.stop()` then
    `path.name` — worked for dictation and raised `AttributeError` for a
    meeting. Nothing consumes the two polymorphically, so this compares the
    resolved return annotation of every abstract member against both
    implementations and fails the moment one of them stops matching.
    """
    from app.audio.base import AudioRecorder
    from app.audio.recorder import MicrophoneRecorder

    abstract_names = sorted(AudioRecorder.__abstractmethods__)
    assert abstract_names == ["duration_seconds", "is_recording", "level_db", "start"], (
        "an abstract member was added or removed — check both recorders still "
        "return the same thing for it before widening this list"
    )

    def _returns(owner: type, name: str):
        member = owner.__dict__.get(name) or getattr(owner, name)
        function = member.fget if isinstance(member, property) else member
        return typing.get_type_hints(function).get("return")

    for name in abstract_names:
        declared = _returns(AudioRecorder, name)
        for implementation in (MicrophoneRecorder, MeetingRecorder):
            assert _returns(implementation, name) == declared, (
                f"{implementation.__name__}.{name} does not return what "
                f"AudioRecorder.{name} declares"
            )

    assert "stop" not in abstract_names
    assert typing.get_type_hints(MicrophoneRecorder.stop)["return"] is Path
    assert typing.get_type_hints(MeetingRecorder.stop)["return"] is MeetingRecording


@pytest.mark.anyio
async def test_a_meeting_whose_file_cannot_be_written_answers_507_and_is_already_idle(
    client, audio_settings, fake_system_source, fake_microphone_stream
):
    """The real write path fails, and the widget must be told the call ended.

    `_end_capture` releases both handles and returns the recorder to `IDLE`
    before the write is even submitted, so a write failure never means "still
    recording". It used to surface as an uncaught 500, which the widget cannot
    tell from an unreachable backend: the indicator stayed lit and the tray
    stayed flagged after the meeting had ended, clearing only on a second
    click that drew a 409. The `temp_dir` is removed after the capture has
    started, which is the disk failure itself rather than a mocked rejection.
    """
    recorder = MeetingRecorder(audio_settings)
    app.dependency_overrides[get_meeting_recorder] = lambda: recorder
    try:
        await recorder.start()
        _wait_for_devices(recorder)
        _feed_microphone(recorder, 3)
        _deliver_over_a_real_span(recorder, fake_system_source, 3)
        shutil.rmtree(audio_settings.temp_dir)

        resp = await client.post("/audio/meeting/stop")

        assert resp.status_code == 507
        detail = resp.json()["detail"]
        assert detail, "the 507 answer names no cause at all"
        assert "The call ended" not in detail, (
            "the widget writes the sentence the user reads and appends this "
            f"detail as its cause, so the two are read twice: {detail}"
        )
        assert recorder.is_busy is False
        assert recorder.is_recording is False
        assert fake_system_source.stopped is True
    finally:
        recorder.cleanup()


def test_a_failed_wav_write_constructs_no_half_built_wave_object(tmp_path):
    """`wave.open(path)` builds a `Wave_write` around the open it performs.

    When that open fails — the disk-full and vanished-`temp_dir` cases the 507
    answer exists for — the half-constructed object survives in the raised
    exception's traceback and its `__del__` raises `AttributeError: _file`
    into the unraisable hook, printing a second, misleading traceback beside
    the real error. `write_wav` opens the file itself so nothing is built.
    """
    gc.collect()
    before = sum(1 for obj in gc.get_objects() if isinstance(obj, wave.Wave_write))

    with pytest.raises(OSError):
        write_wav(
            tmp_path / "gone" / "meeting.wav",
            np.zeros(16, dtype=np.float32),
            16000,
            channels=1,
        )

    alive = sum(1 for obj in gc.get_objects() if isinstance(obj, wave.Wave_write))
    assert alive == before


def test_a_failed_write_raises_the_507_error_and_not_the_bare_os_error(audio_settings):
    """The router branches on the type, so the type is what has to change.

    Left as the `OSError` the wave module raises, the stop endpoint answers
    500 and `src/widget/meeting-toggle.ts` keeps the indicator lit.
    """
    recorder = MeetingRecorder(audio_settings)
    try:
        captured = _CapturedMeeting(
            microphone_blocks=[],
            system_blocks=[],
            system_rate=SYSTEM_RATE,
            recording_start=0.0,
            recording_stop=1.0,
            truncated=False,
        )
        with pytest.raises(MeetingWriteFailedError) as raised:
            recorder._write_captured_meeting(captured)
        assert isinstance(raised.value.__cause__, OSError)
        assert not isinstance(raised.value, MeetingCaptureAbortedError)
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_cleanup_retires_both_executors_and_refuses_a_later_start(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """`cleanup()` is terminal: the app is exiting, and a start that reopened
    a device behind it would hold one past the process the user closed."""
    already_running = set(threading.enumerate())
    recorder = MeetingRecorder(audio_settings)

    await recorder.start()
    _feed_microphone(recorder, 4)
    _deliver_over_a_real_span(recorder, fake_system_source, 4)
    await asyncio.sleep(0.1)
    await recorder.stop()
    _wait_for_writes(recorder)

    recorder.cleanup()
    _wait_for_devices(recorder)
    _wait_for_writes(recorder)

    with pytest.raises(RuntimeError):
        await recorder.start()

    assert recorder._devices_in_flight == 0
    lingering = [
        thread.name
        for thread in threading.enumerate()
        if thread not in already_running
        and thread.name.startswith(("meeting-devices", "meeting-writer"))
    ]
    assert lingering == [], f"{lingering} outlived the cleanup that retired them"


@pytest.mark.asyncio
async def test_far_side_audio_during_the_microphone_open_reaches_the_file(audio_settings):
    """The far side is live from the moment the system source starts, and the
    microphone open is an unbounded PortAudio device open behind it.

    Two things have to hold for that audio to survive: `_store` admits it,
    and the recording clock starts at the system source rather than at the
    publish — otherwise `place_on_timeline` clips every sample that precedes
    `recording_start`.
    """
    open_block_seconds = 0.5
    source = _FakeSystemAudioSource()
    opening_started = threading.Event()
    release_open = threading.Event()
    delivered_blocks = 20

    def deliver_during_the_open() -> None:
        assert opening_started.wait(timeout=5.0), "the microphone never opened"
        began = time.monotonic()
        for index in range(delivered_blocks):
            source.deliver(began + index * BLOCK_FRAMES / SYSTEM_RATE, fill=0.9)

    recorder = MeetingRecorder(audio_settings)
    deliverer = threading.Thread(target=deliver_during_the_open)
    deliverer.start()
    threading.Timer(open_block_seconds, release_open.set).start()

    with patch("app.audio.meeting_recorder.create_system_audio_source", lambda _: source):
        with patch(
            "app.audio.meeting_recorder.sd.InputStream",
            functools.partial(_BlockingMicrophoneStream, opening_started, release_open),
        ):
            await recorder.start()
            deliverer.join(timeout=5.0)
            audio_path = (await recorder.stop()).path
            _wait_for_writes(recorder)

    rate, signal = _wav_signal(audio_path)
    audible = np.flatnonzero(np.abs(signal) > 10 ** (-40 / 20))

    assert audible.size > 0, (
        "the far side spoke through the whole microphone open and the file is silent"
    )
    assert audible[0] / rate < 0.15, (
        f"the first audible sample is at {audible[0] / rate:.3f}s — the far side's "
        f"audio was clipped off the front of the timeline"
    )
    assert len(signal) >= delivered_blocks * BLOCK_FRAMES * rate // SYSTEM_RATE


@pytest.mark.asyncio
async def test_a_start_that_fails_after_cleanup_reports_the_device_error(
    audio_settings, fake_microphone_stream
):
    """GitHub review iteration 4, finding 1 — first half.

    The abandonment `start()` sends behind a failed open is unsendable once
    `cleanup()` has retired the owner thread, and the caller must still be
    told what actually went wrong: a `/meeting/start` racing the lifespan
    drain answers 501, not 500.
    """
    recorder = MeetingRecorder(audio_settings)

    with patch("app.audio.meeting_recorder.create_system_audio_source", return_value=None):
        gate = _hold_the_owner_thread(recorder)
        starting = asyncio.ensure_future(recorder.start())
        await asyncio.sleep(0)

        recorder.cleanup()
        gate.set()

        with pytest.raises(SystemAudioUnavailableError):
            await starting

    _wait_for_devices(recorder)


@pytest.mark.asyncio
async def test_a_start_cancelled_after_cleanup_still_arrives_as_a_cancellation(
    audio_settings, fake_microphone_stream
):
    """GitHub review iteration 4, finding 1 — second half.

    A cancellation replaced by an ordinary exception stops the task being
    treated as cancelled, which is the failure `_detachable_result` goes to
    some length to avoid everywhere else in this module.
    """
    recorder = MeetingRecorder(audio_settings)

    with patch("app.audio.meeting_recorder.create_system_audio_source", return_value=None):
        gate = _hold_the_owner_thread(recorder)
        starting = asyncio.ensure_future(recorder.start())
        await asyncio.sleep(0)

        recorder.cleanup()
        starting.cancel()

        outcome = "the start returned normally"
        try:
            await starting
        except asyncio.CancelledError:
            outcome = "cancelled"
        except BaseException as raised:
            outcome = repr(raised)

        gate.set()

    assert outcome == "cancelled", (
        f"the cancellation was replaced by {outcome}, so the task is no longer "
        f"treated as cancelled"
    )
    _wait_for_devices(recorder)


@pytest.mark.asyncio
async def test_a_new_meeting_cannot_clear_the_previous_stops_truncation(
    tmp_path, fake_system_source, fake_microphone_stream
):
    """GitHub review iteration 4, finding 2.

    The write outlives `_end_capture`, so a second meeting can be accepted
    while the first stop is still assembling its answer. The live flag is
    that second meeting's; the answer belongs to the first.
    """
    settings = AudioSettings(
        sample_rate=16000,
        channels=1,
        temp_dir=tmp_path / "tmp",
        meeting_max_raw_bytes=BLOCK_FRAMES * 4 * 3,
    )
    recorder = MeetingRecorder(settings)

    await recorder.start()
    _feed_microphone(recorder, 50)
    await asyncio.sleep(0.02)
    recording = await recorder.stop()
    _wait_for_devices(recorder)
    _wait_for_writes(recorder)

    assert recording.truncated is True

    await recorder.start()

    assert recorder._truncated is False, (
        "the second meeting did not reset the live flag, so this proves nothing"
    )
    assert recording.truncated is True, (
        "a truncated recording stopped reporting its dropped audio as soon as "
        "the next meeting started"
    )
    recorder.cleanup()
    _wait_for_devices(recorder)


@pytest.mark.anyio
async def test_a_stop_accepted_during_the_open_reports_the_duration_it_captured(
    client, audio_settings
):
    """GitHub review iteration 4, finding 3.

    The stop guard is `is_busy`, so a stop issued during the microphone open
    is queued behind it and harvests a real capture — the far side is already
    live and `_store` admits it. The harvest clears the live clock before the
    endpoint reads it, so the response cannot be built from it.

    The window is asserted through `_state`, not through `duration_seconds`.
    `_start_time` is published the instant the system source starts, which is
    before the microphone open this test parks, so the live duration is
    already counting here — it merely reads back as `0.0` while less than one
    `time.monotonic()` tick has passed, which on Windows is 15.625 ms. Under
    full-suite load that tick rolls over and the old assertion failed on a
    capture that was behaving exactly as intended.
    """
    opening_started = threading.Event()
    release_open = threading.Event()
    source = _FakeSystemAudioSource()
    recorder = MeetingRecorder(audio_settings)
    app.dependency_overrides[get_meeting_recorder] = lambda: recorder

    with patch(
        "app.audio.meeting_recorder.create_system_audio_source", lambda _: source
    ), patch(
        "app.audio.meeting_recorder.sd.InputStream",
        functools.partial(_BlockingMicrophoneStream, opening_started, release_open),
    ):
        starting = asyncio.ensure_future(recorder.start())
        await asyncio.to_thread(opening_started.wait, 5.0)

        assert recorder._state is MeetingState.STARTING, (
            "the open had already finished, so this proves nothing about the window"
        )
        for index in range(8):
            source.deliver(time.monotonic() + index * BLOCK_FRAMES / SYSTEM_RATE)

        stopping = asyncio.ensure_future(client.post("/audio/meeting/stop"))
        await asyncio.sleep(0.3)
        release_open.set()
        await starting
        resp = await stopping
        _wait_for_writes(recorder)

    reported = resp.json()["duration_seconds"]

    assert resp.status_code == 200
    assert reported > 0.25, (
        f"a stop accepted during the open reported {reported}s for a capture "
        f"that really ran"
    )


@pytest.mark.anyio
async def test_a_start_refused_while_the_devices_are_busy_does_not_claim_a_recording(
    client,
):
    """GitHub review iteration 4, finding 4.

    `is_busy` covers `STARTING`, `STOPPING` and a merely queued command, and
    the widget shows this detail verbatim. Telling the user a recording is in
    progress while one is being torn down is the wrong half of the truth.
    """
    app.dependency_overrides[get_meeting_recorder] = lambda: _FakeRecorder(
        is_recording=False, is_busy=True
    )

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 409
    assert resp.json()["detail"] == MEETING_BUSY_DETAIL
    assert "releasing its devices" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_a_failed_start_clears_both_level_meters(audio_settings):
    """GitHub review iteration 4, finding 5.

    `_store` admits blocks from the moment the token is claimed, so the far
    side has usually written a real dBFS value by the time the microphone
    open fails. A live `system_level_db` beside `is_recording: false` is the
    single wrong answer that meter exists to prevent.
    """
    source = _FakeSystemAudioSource()

    def _open_fails(**kwargs):
        source.deliver(time.monotonic(), fill=0.5)
        raise OSError("the microphone is in use by another application")

    recorder = MeetingRecorder(audio_settings)

    with patch(
        "app.audio.meeting_recorder.create_system_audio_source", lambda _: source
    ), patch("app.audio.meeting_recorder.sd.InputStream", _open_fails):
        with pytest.raises(OSError):
            await recorder.start()

    _wait_for_devices(recorder)

    assert recorder.is_recording is False
    assert recorder.system_endpoint is None
    assert recorder.system_level_db == float("-inf"), (
        "the status reports a live far side while nothing is being captured"
    )
    assert recorder.level_db == float("-inf")


@pytest.mark.asyncio
async def test_an_abandoned_stop_does_not_promise_a_file(
    audio_settings, fake_system_source, fake_microphone_stream, caplog
):
    """GitHub review iteration 4, finding 6.

    A cancellation landing on the first await leaves the harvest still queued,
    and the harvest can end in `MeetingCaptureEmptyError` with nothing
    written. An operator chasing a missing meeting must not be sent looking
    for a path that was never logged.
    """
    recorder = MeetingRecorder(audio_settings)

    await recorder.start()
    gate = _hold_the_owner_thread(recorder)
    stopping = asyncio.ensure_future(recorder.stop())
    await asyncio.sleep(0)
    stopping.cancel()

    with caplog.at_level(logging.WARNING):
        try:
            await stopping
        except asyncio.CancelledError:
            pass

    gate.set()
    _wait_for_devices(recorder)
    _wait_for_writes(recorder)

    abandoned = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "abandoned" in record.getMessage()
    ]

    assert list(audio_settings.temp_dir.glob("meeting_*.wav")) == [], (
        "the capture produced a file, so this proves nothing about the promise"
    )
    assert abandoned, "the abandoned stop was not reported at all"
    assert all("if the capture produced a file" in message for message in abandoned)
    assert not any("still being written" in message for message in abandoned), (
        "the warning promises a file that this run never produced"
    )


@pytest.mark.asyncio
async def test_a_status_read_during_the_open_raises_a_reloaded_widgets_indicator(
    audio_settings, client
):
    """GitHub review iteration 5, finding 1.

    `syncMeetingIndicator` runs once, at widget load, and returns early when
    the status agrees with the window's own state (`src/widget/widget.ts`).
    A widget loading during the multi-second device open therefore gets one
    chance to raise the indicator ADR 040 obligation 2 requires, and this is
    the status it gets — while far-side audio is already being kept.

    The elapsed clock is the same answer read twice: the widget starts it at
    `now - duration_seconds`, so a status reporting a live meeting with a
    duration of `0.0` puts that meeting's start at the reload instant.
    """
    source = _FakeSystemAudioSource()
    opening_started = threading.Event()
    release_open = threading.Event()
    system_started_at: list[float] = []
    open_source = source.start

    def start_and_note(on_block) -> None:
        open_source(on_block)
        system_started_at.append(time.monotonic())

    source.start = start_and_note
    recorder = MeetingRecorder(audio_settings)
    app.dependency_overrides[get_meeting_recorder] = lambda: recorder

    with patch("app.audio.meeting_recorder.create_system_audio_source", lambda _: source):
        with patch(
            "app.audio.meeting_recorder.sd.InputStream",
            functools.partial(_BlockingMicrophoneStream, opening_started, release_open),
        ):
            starting = asyncio.ensure_future(recorder.start())
            await asyncio.sleep(0)
            assert opening_started.wait(timeout=5.0), "the microphone never opened"
            source.deliver(time.monotonic(), fill=0.9)
            await asyncio.sleep(0.2)

            status = (await client.get("/audio/meeting/status")).json()
            read_at = time.monotonic()

            release_open.set()
            await starting

    await recorder.stop()
    _wait_for_writes(recorder)

    assert status["is_recording"] is True, (
        "the widget reads is_recording: false, finds it already agrees with its own "
        "inactive state and returns — the indicator stays dark for the whole meeting"
    )
    assert status["duration_seconds"] >= 0.1, (
        "the widget would start the meeting's elapsed clock at the reload instant"
    )
    drift = read_at - status["duration_seconds"] - system_started_at[0]
    assert abs(drift) < 0.05, (
        f"the indicator's clock would start {drift:+.3f}s away from the instant the "
        f"far side began being recorded"
    )


@pytest.mark.asyncio
async def test_an_abandoned_start_clears_both_level_meters(audio_settings):
    """GitHub review iteration 5, finding 2.

    `_discard_capture` is the sibling of the failed-open branch
    `test_a_failed_start_clears_both_level_meters` pins, and it is reached
    with real values on both meters: a start whose devices opened and took
    blocks, then had its request cancelled. Either the two paths leave the
    same recorder or the status answers a live far side beside
    `system_endpoint: null`.
    """
    source = _FakeSystemAudioSource()
    opening_started = threading.Event()
    release_open = threading.Event()
    recorder = MeetingRecorder(audio_settings)

    with patch("app.audio.meeting_recorder.create_system_audio_source", lambda _: source):
        with patch(
            "app.audio.meeting_recorder.sd.InputStream",
            functools.partial(_BlockingMicrophoneStream, opening_started, release_open),
        ):
            starting = asyncio.ensure_future(recorder.start())
            await asyncio.sleep(0)
            assert opening_started.wait(timeout=5.0), "the microphone never opened"
            source.deliver(time.monotonic(), fill=0.9)
            _feed_microphone(recorder, 1)

            assert recorder.system_level_db > float("-inf")
            assert recorder.level_db > float("-inf")

            starting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await starting

            release_open.set()
            _wait_for_devices(recorder)

    assert recorder.is_busy is False
    assert recorder.is_recording is False
    assert recorder.duration_seconds == 0.0
    assert recorder.system_endpoint is None
    assert recorder.system_level_db == float("-inf"), (
        "the status reports the discarded session's far side beside a null endpoint"
    )
    assert recorder.level_db == float("-inf"), (
        "the status reports the discarded session's microphone level"
    )


@pytest.mark.asyncio
async def test_a_second_meeting_cannot_rewrite_what_the_first_stop_answers(
    tmp_path, fake_system_source, fake_microphone_stream
):
    """The stop answer describes its own capture, not the recorder's live state.

    The recorder stops being busy as soon as the harvest returns, so a whole
    second meeting can start and finish while the first file is still being
    written. This drives that interleaving for real: the first write is held
    open until the second meeting's harvest has run, which is the moment any
    value kept on the recorder would have been overwritten.
    """
    settings = AudioSettings(
        sample_rate=16000,
        channels=1,
        temp_dir=tmp_path / "tmp",
        meeting_max_raw_bytes=BLOCK_FRAMES * 4 * 3,
    )
    recorder = MeetingRecorder(settings)
    write_started = threading.Event()
    release_write = threading.Event()
    write_the_wav = recorder._assemble_and_write

    def held_open_write(*args):
        if not write_started.is_set():
            write_started.set()
            release_write.wait(5.0)
        return write_the_wav(*args)

    recorder._assemble_and_write = held_open_write

    await recorder.start()
    _feed_microphone(recorder, 50)
    _deliver_over_a_real_span(recorder, fake_system_source, 4)
    await asyncio.sleep(0.3)
    long_truncated_stop = asyncio.ensure_future(recorder.stop())
    await asyncio.to_thread(write_started.wait, 5.0)

    assert recorder.is_busy is False, (
        "the recorder stayed busy through the write, so no second meeting could "
        "reach the fields this test is about and it proves nothing"
    )

    await recorder.start()
    _feed_microphone(recorder, 1)
    _deliver_over_a_real_span(recorder, fake_system_source, 1)
    short_clean_stop = asyncio.ensure_future(recorder.stop())
    await asyncio.to_thread(_wait_for_devices, recorder)

    release_write.set()
    first = await long_truncated_stop
    second = await short_clean_stop

    assert first.truncated is True, (
        "the 45-minute meeting that dropped audio at the cap answered "
        "truncated=False, because the meeting that started during its write "
        "reset the flag it was reading"
    )
    assert second.truncated is False
    assert first.duration_seconds > second.duration_seconds, (
        f"the first stop reported {first.duration_seconds:.3f}s against the "
        f"second's {second.duration_seconds:.3f}s — it answered with the other "
        "meeting's clock"
    )
    assert first.path != second.path
    assert first.path.exists()


def test_a_microphone_stream_whose_stop_raises_is_still_closed(caplog):
    """A `stop()` that raises must not cost the `close()`.

    Nothing else holds a reference by the time `_release_devices` runs and
    `sounddevice._StreamBase` has no finalizer, so a skipped `close()` holds
    that PortAudio stream for the life of the process — and every meeting
    after it adds another.
    """

    class _UnpluggedStream:
        def __init__(self):
            self.closes = 0

        def stop(self):
            raise OSError("Device unavailable")

        def close(self):
            self.closes += 1

    stream = _UnpluggedStream()

    with caplog.at_level(logging.WARNING, logger="app.audio.meeting_recorder"):
        _release_devices(stream, None)

    assert stream.closes == 1, (
        "the stream was never closed after its stop() raised — the microphone "
        "stays claimed until the app restarts"
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any("Stopping the meeting microphone stream failed" in m for m in messages), (
        f"the failing call is not identifiable from the log: {messages}"
    )


@pytest.mark.asyncio
async def test_a_second_cleanup_returns_and_a_later_request_is_refused_not_a_500(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """Shutdown answers in the endpoints' own vocabulary.

    `cleanup()` retires the owner thread, and `main.py` may call it again;
    a start or a stop landing in that window must reach the router as the
    409 the widget handles rather than a bare executor error as a 500.
    """
    recorder = MeetingRecorder(audio_settings)

    recorder.cleanup()
    _wait_for_devices(recorder)
    recorder.cleanup()

    with pytest.raises(MeetingCaptureAbortedError):
        await recorder.start()
    with pytest.raises(MeetingCaptureAbortedError):
        await recorder.stop()


@pytest.mark.asyncio
async def test_a_completed_stop_clears_both_level_meters(
    audio_settings, fake_system_source, fake_microphone_stream
):
    """GitHub review iteration 8, finding 1.

    The abandoned-start and failed-start paths both return the meters to
    silence, and `_discard_capture`'s docstring gives the reason: a status
    read cannot report a session's levels next to `system_endpoint: null`.
    A normal stop is the fourth path to that same state and must agree.
    """
    recorder = MeetingRecorder(audio_settings)

    try:
        await recorder.start()
        _wait_for_devices(recorder)
        _feed_microphone(recorder, 5)
        _deliver_over_a_real_span(recorder, fake_system_source, 5)

        assert recorder.level_db > float("-inf")
        assert recorder.system_level_db > float("-inf")

        await recorder.stop()

        assert recorder.is_recording is False
        assert recorder.system_endpoint is None
        assert recorder.level_db == float("-inf"), (
            "the status reports the ended meeting's microphone level beside "
            "system_endpoint: null"
        )
        assert recorder.system_level_db == float("-inf"), (
            "the status reports the ended meeting's far-side level beside "
            "system_endpoint: null"
        )
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
async def test_a_status_during_the_microphone_open_names_what_it_is_capturing(
    audio_settings
):
    """GitHub review iteration 8, finding 2.

    `is_recording` goes true the instant the system source starts, seconds
    before the microphone stream is up on Windows. `system_endpoint` is
    published in that same lock hold, so the window in which the status
    announces a live meeting it cannot name does not exist.
    """
    source = _FakeSystemAudioSource(endpoint_name="Speakers [Loopback]")
    recorder = MeetingRecorder(audio_settings)
    observed: dict[str, object] = {}

    def _open(**kwargs):
        observed["is_recording"] = recorder.is_recording
        observed["endpoint"] = recorder.system_endpoint
        return MagicMock()

    try:
        with patch(
            "app.audio.meeting_recorder.create_system_audio_source", lambda _: source
        ), patch("app.audio.meeting_recorder.sd.InputStream", _open):
            await recorder.start()

        assert observed["is_recording"] is True
        assert observed["endpoint"] == "Speakers [Loopback]", (
            "the status announces a live meeting whose capture target is null "
            "for the whole microphone open"
        )
    finally:
        recorder.cleanup()


def test_a_failed_write_carries_the_cause_and_not_a_second_sentence(audio_settings):
    """GitHub review iteration 8, finding 3.

    `src/widget/meeting-toggle.ts` writes the sentence the user reads and
    appends this message as its cause. A message that is itself a sentence
    reaches the user as the same statement twice.
    """
    recorder = MeetingRecorder(audio_settings)
    captured = _CapturedMeeting(
        microphone_blocks=[],
        system_blocks=[],
        system_rate=SYSTEM_RATE,
        recording_start=0.0,
        recording_stop=1.0,
        truncated=False,
    )

    try:
        with patch.object(
            recorder, "_assemble_and_write", side_effect=OSError("No space left on device")
        ):
            with pytest.raises(MeetingWriteFailedError) as raised:
                recorder._write_captured_meeting(captured)
        assert str(raised.value) == "No space left on device"

        with patch.object(recorder, "_assemble_and_write", side_effect=OSError()):
            with pytest.raises(MeetingWriteFailedError) as unnamed:
                recorder._write_captured_meeting(captured)
        assert str(unnamed.value) == "OSError", (
            "the user is told the recording could not be saved and given "
            "nothing after the colon"
        )
    finally:
        recorder.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["microphone", "system"])
async def test_a_stop_landing_between_the_two_lock_holds_leaves_the_meter_silent(
    audio_settings, fake_system_source, fake_microphone_stream, side
):
    """GitHub review iteration 9, finding 1.

    A callback keeps its block under one lock hold and publishes its level
    under a second one. `test_a_completed_stop_clears_both_level_meters`
    drives the callback from the test thread, so the gap between those two
    holds never opens. Here the callback really is parked inside it, on its
    own thread, while the stop harvests and clears everything — which is the
    interleaving PortAudio produces when it preempts the callback thread.
    """
    recorder = MeetingRecorder(audio_settings)
    reached_the_gap = threading.Event()
    stop_completed = threading.Event()
    real_store = recorder._store

    def _park_between_the_holds(token, blocks, arrival, mono):
        kept = real_store(token, blocks, arrival, mono)
        if kept:
            reached_the_gap.set()
            assert stop_completed.wait(timeout=5.0), "the stop never completed"
        return kept

    try:
        await recorder.start()
        _wait_for_devices(recorder)
        _feed_microphone(recorder, 3)
        _deliver_over_a_real_span(recorder, fake_system_source, 3)
        time.sleep(0.05)

        token = recorder._session_token
        block = np.full(BLOCK_FRAMES, 0.4, dtype=np.float32)
        if side == "microphone":
            racing = functools.partial(
                recorder._microphone_callback,
                token,
                block.reshape(BLOCK_FRAMES, 1),
                BLOCK_FRAMES,
                None,
                MagicMock(),
            )
        else:
            racing = functools.partial(
                recorder._system_callback, token, time.monotonic(), block
            )

        recorder._store = _park_between_the_holds
        callback_thread = threading.Thread(target=racing, name="parked-callback")
        callback_thread.start()
        assert reached_the_gap.wait(timeout=5.0), "the callback never kept its block"

        await recorder.stop()
        _wait_for_writes(recorder)

        stop_completed.set()
        callback_thread.join(timeout=5.0)
        assert not callback_thread.is_alive()

        assert recorder.is_recording is False
        assert recorder.system_endpoint is None
        assert recorder.level_db == float("-inf"), (
            "a callback resuming after the stop republished the ended meeting's "
            "microphone level beside system_endpoint: null"
        )
        assert recorder.system_level_db == float("-inf"), (
            "a callback resuming after the stop republished the ended meeting's "
            "far-side level beside system_endpoint: null"
        )
    finally:
        stop_completed.set()
        recorder.cleanup()


@pytest.mark.asyncio
async def test_far_side_audio_arriving_during_the_open_is_not_trimmed_away(
    audio_settings, fake_microphone_stream
):
    """GitHub review iteration 9, finding 2.

    The macOS helper streams continuously once its header is out, so the pipe
    can already hold a block by the time `source.start()` returns. If the
    recording anchor is read after that return, every such block carries
    `arrival < recording_start` and `place_on_timeline` trims it off the
    front. The anchor is taken before the start, so the audio survives.
    """
    source = _FakeSystemAudioSource()
    open_duration = 0.06
    delivered = 8
    first_arrival: list[float] = []

    def _start(on_block):
        source.on_block = on_block
        source.started = True
        first_arrival.append(time.monotonic())
        on_block(first_arrival[0], np.full(BLOCK_FRAMES, 0.5, dtype=np.float32))
        time.sleep(open_duration)

    source.start = _start
    recorder = MeetingRecorder(audio_settings)

    try:
        with patch(
            "app.audio.meeting_recorder.create_system_audio_source", lambda _: source
        ):
            await recorder.start()
        _wait_for_devices(recorder)

        for index in range(1, delivered):
            source.deliver(first_arrival[0] + index * BLOCK_FRAMES / SYSTEM_RATE)
        time.sleep(0.25)

        audio_path = (await recorder.stop()).path
        _wait_for_writes(recorder)

        with wave.open(str(audio_path), "rb") as wf:
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        block_samples = BLOCK_FRAMES * audio_settings.sample_rate // SYSTEM_RATE
        assert int(np.count_nonzero(samples)) >= (delivered - 1) * block_samples, (
            "the far-side audio delivered while the source was still starting "
            "was trimmed off the front of the recording"
        )
    finally:
        recorder.cleanup()

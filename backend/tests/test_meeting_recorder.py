"""Spec 066: MeetingRecorder, the platform factory, and the meeting endpoints.

Every device is stubbed. `pyaudiowpatch` is a Windows-only wheel that cannot
be installed on the ubuntu CI runner at all, so the Windows source's
block-handling logic is exercised against a fake module injected into
`sys.modules` — only real-device behaviour is left to the `[win]` checklist.
"""

from __future__ import annotations

import importlib
import sys
import time
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.audio import get_active_recorder, get_meeting_recorder
from app.audio.config import AudioSettings
from app.audio.meeting_recorder import MeetingRecorder
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


class FakeSystemAudioSource(SystemAudioSource):
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
    source = FakeSystemAudioSource()
    with patch("app.audio.meeting_recorder.create_system_audio_source", return_value=source):
        yield source


@pytest.fixture
def fake_microphone_stream():
    with patch("app.audio.meeting_recorder.sd.InputStream") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


def _feed_microphone(recorder: MeetingRecorder, count: int, fill: float = 0.3) -> None:
    for _ in range(count):
        block = np.full((BLOCK_FRAMES, 1), fill, dtype=np.float32)
        recorder._microphone_callback(block, BLOCK_FRAMES, None, MagicMock())




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
    audio_path = await recorder.stop()

    assert recorder.truncated is True
    assert audio_path.exists()




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
    audio_path = await recorder.stop()

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
    audio_path = await recorder.stop()

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
    audio_path = await recorder.stop()

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

    assert recorder.is_recording is False
    assert fake_system_source.stopped is True
    fake_microphone_stream.return_value.stop.assert_called_once()
    fake_microphone_stream.return_value.close.assert_called_once()


def test_cleanup_noop_when_never_started(audio_settings):
    recorder = MeetingRecorder(audio_settings)

    recorder.cleanup()

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


def test_factory_returns_none_when_the_windows_device_lookup_fails(monkeypatch):
    """The factory answers None rather than raising, so the caller decides
    what an absent source means."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules, "app.audio.windows_loopback", _module_raising_on_construction()
    )

    assert create_system_audio_source(AudioSettings()) is None


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
def acknowledged_meeting_consent():
    """The disclosure is a first-run gate, not the subject of most of these
    tests — the ones that are unset it explicitly."""
    from app.preferences import user_settings

    user_settings.update_user_settings({"meeting_consent_acknowledged": True})


class _StubRecorder:
    def __init__(self, is_recording: bool = False):
        self.is_recording = is_recording
        self.duration_seconds = 0.0
        self.level_db = float("-inf")
        self.system_level_db = float("-inf")
        self.system_endpoint = None
        self.truncated = False
        self.started = False

    async def start(self):
        self.started = True
        self.is_recording = True

    async def stop(self):
        self.is_recording = False
        raise AssertionError("this stub is not expected to stop")


@pytest.mark.anyio
async def test_meeting_start_returns_501_where_no_system_source_exists(client):
    """AC: the endpoint answers 501 naming the platform limitation."""

    class _Unavailable(_StubRecorder):
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
    meeting = _StubRecorder()
    app.dependency_overrides[get_meeting_recorder] = lambda: meeting
    app.dependency_overrides[get_active_recorder] = lambda: _StubRecorder(is_recording=True)

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 409
    assert meeting.started is False


@pytest.mark.anyio
async def test_audio_start_is_409_while_a_meeting_is_recording(client):
    """AC: the mirror-image guard."""
    from app.audio import get_active_meeting_recorder, get_recorder

    dictation = _StubRecorder()
    app.dependency_overrides[get_recorder] = lambda: dictation
    app.dependency_overrides[get_active_meeting_recorder] = lambda: _StubRecorder(
        is_recording=True
    )

    resp = await client.post("/audio/start")

    assert resp.status_code == 409
    assert dictation.started is False


@pytest.mark.anyio
async def test_meeting_start_is_409_when_already_recording(client):
    app.dependency_overrides[get_meeting_recorder] = lambda: _StubRecorder(is_recording=True)

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 409


@pytest.mark.anyio
async def test_meeting_stop_without_start_is_409(client):
    app.dependency_overrides[get_meeting_recorder] = lambda: _StubRecorder()

    resp = await client.post("/audio/meeting/stop")

    assert resp.status_code == 409


@pytest.mark.anyio
async def test_meeting_status_reports_idle(client):
    app.dependency_overrides[get_meeting_recorder] = lambda: _StubRecorder()

    resp = await client.get("/audio/meeting/status")

    assert resp.status_code == 200
    assert resp.json()["is_recording"] is False


@pytest.mark.anyio
async def test_meeting_stop_reports_the_filename_and_truncation(client, tmp_path):
    class _Stopping(_StubRecorder):
        def __init__(self):
            super().__init__(is_recording=True)
            self.truncated = True

        async def stop(self):
            self.is_recording = False
            path = tmp_path / "meeting_abc123.wav"
            path.write_bytes(b"")
            return path

    app.dependency_overrides[get_meeting_recorder] = lambda: _Stopping()

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

    class _Stopping(_StubRecorder):
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

    app.dependency_overrides[get_recorder] = lambda: _StubRecorder()

    resp = await client.request("POST" if path == "/audio/start" else "GET", path)

    assert resp.status_code == 200
    assert set(resp.json()) == {"is_recording", "duration_seconds", "level_db"}


@pytest.mark.anyio
async def test_the_meeting_status_reports_the_endpoint_and_the_system_level(client):
    """AC: the meeting responses grew by exactly two fields."""
    recorder = _StubRecorder()
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
    recorder = _StubRecorder()
    app.dependency_overrides[get_meeting_recorder] = lambda: recorder

    with patch("app.audio.meeting_recorder.create_system_audio_source") as factory:
        resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 403
    assert recorder.started is False
    factory.assert_not_called()


@pytest.mark.anyio
async def test_meeting_start_is_never_403_once_acknowledged(client):
    app.dependency_overrides[get_meeting_recorder] = lambda: _StubRecorder()

    resp = await client.post("/audio/meeting/start")

    assert resp.status_code == 200


@pytest.mark.anyio
async def test_meeting_stop_and_status_are_not_behind_the_consent_gate(client):
    """A recording already running must always be stoppable and visible."""
    from app.preferences import user_settings

    user_settings.update_user_settings({"meeting_consent_acknowledged": False})
    app.dependency_overrides[get_meeting_recorder] = lambda: _StubRecorder()

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

    assert recorder.system_endpoint is None

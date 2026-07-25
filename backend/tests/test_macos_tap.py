"""Spec 074: the macOS system-audio source, at the Python boundary.

The helper binary cannot be compiled or run here — no macOS hardware, no Swift
toolchain — so everything below drives `MacOSTapSource` against a fake process
whose stdout follows the contract written down in `app.audio.macos_tap`'s
module docstring and in the helper's own header comment. What stays unverified
is the Swift, and only the Swift.
"""

from __future__ import annotations

import io
import json
import sys
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.audio.config import AudioSettings
from app.audio.macos_tap import MacOSTapSource, parse_tap_header, resolve_audio_tap_path
from app.audio.meeting_recorder import MeetingRecorder
from app.audio.system_source import SystemAudioUnavailableError, create_system_audio_source

BLOCK_FRAMES = 64


@pytest.fixture
def tap_settings(tmp_path):
    return AudioSettings(
        sample_rate=16000,
        channels=1,
        temp_dir=tmp_path / "tmp",
        meeting_block_frames=BLOCK_FRAMES,
    )


def header_line(
    sample_rate: int = 48000,
    channels: int = 2,
    fmt: str = "f32le",
    tap_stream_index: int = 0,
) -> bytes:
    return (
        json.dumps(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "format": fmt,
                "tap_stream_index": tap_stream_index,
            }
        ).encode()
        + b"\n"
    )


class FakeTapProcess:
    """A helper process whose whole life is a byte string on stdout."""

    def __init__(self, stdout: bytes, returncode: int = 0, stderr: bytes = b""):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode


def tap_stdout(blocks: int, channels: int = 2, fill: float = 0.25, **header) -> bytes:
    frames = np.full(blocks * BLOCK_FRAMES * channels, fill, dtype="<f4")
    return header_line(channels=channels, **header) + frames.tobytes()


def run_source(settings: AudioSettings, process: FakeTapProcess) -> list[np.ndarray]:
    source = MacOSTapSource(settings, Path("/nonexistent/justsay-audiotap"))
    received: list[np.ndarray] = []
    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: received.append(mono))
        source._reader.join(timeout=2.0)
        source.stop()
    return received




def test_the_header_is_parsed_rather_than_assumed(tap_settings):
    """AC: `native_sample_rate` comes back as 48000 — a value that appears
    nowhere in AudioSettings, so it cannot have come from the config."""
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    process = FakeTapProcess(tap_stdout(blocks=3))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None)
        source._reader.join(timeout=2.0)
        source.stop()

    assert source.native_sample_rate == 48000
    assert tap_settings.sample_rate == 16000


def test_k_frames_of_stdout_become_exactly_k_over_block_frames_mono_blocks(tap_settings):
    """AC: block count and block length both follow from the header."""
    received = run_source(tap_settings, FakeTapProcess(tap_stdout(blocks=5)))

    assert len(received) == 5
    assert {len(block) for block in received} == {BLOCK_FRAMES}


def test_a_trailing_partial_block_is_dropped_rather_than_padded(tap_settings):
    stdout = tap_stdout(blocks=2) + np.full(BLOCK_FRAMES, 0.1, dtype="<f4").tobytes()

    received = run_source(tap_settings, FakeTapProcess(stdout))

    assert len(received) == 2


def test_interleaved_stereo_is_downmixed_to_mono(tap_settings):
    frames = np.tile(
        np.array([1.0, 0.0], dtype="<f4"), BLOCK_FRAMES
    )
    process = FakeTapProcess(header_line(channels=2) + frames.tobytes())

    received = run_source(tap_settings, process)

    assert len(received) == 1
    assert received[0].tolist() == pytest.approx([0.5] * BLOCK_FRAMES)


def test_a_mono_helper_stream_is_passed_through(tap_settings):
    received = run_source(tap_settings, FakeTapProcess(tap_stdout(blocks=2, channels=1)))

    assert len(received) == 2
    assert {len(block) for block in received} == {BLOCK_FRAMES}


def test_the_endpoint_is_named_for_the_status_response(tap_settings):
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    assert source.endpoint_name




@pytest.mark.parametrize(
    "stdout",
    [
        b"",
        b"not json at all\n",
        json.dumps({"sample_rate": 48000, "channels": 2, "format": "s16le"}).encode() + b"\n",
        json.dumps({"sample_rate": 48000, "format": "f32le"}).encode() + b"\n",
        json.dumps({"sample_rate": 0, "channels": 2, "format": "f32le"}).encode() + b"\n",
        b'"a bare string"\n',
        json.dumps({"sample_rate": 48000, "channels": 2, "format": "f32le"}).encode() + b"\n",
        json.dumps(
            {"sample_rate": 48000, "channels": 2, "format": "f32le", "tap_stream_index": -1}
        ).encode()
        + b"\n",
        json.dumps(
            {"sample_rate": 48000, "channels": 2, "format": "f32le", "tap_stream_index": None}
        ).encode()
        + b"\n",
    ],
)
def test_a_malformed_or_missing_header_is_an_unavailable_source(tap_settings, stdout):
    """AC: never a crash, and never a source that silently delivers nothing."""
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    process = FakeTapProcess(stdout)

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        with pytest.raises(SystemAudioUnavailableError):
            source.start(lambda arrival, mono: None)

    assert process.terminated is True


def test_a_helper_that_cannot_be_launched_is_an_unavailable_source(tap_settings):
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with patch("app.audio.macos_tap.subprocess.Popen", side_effect=OSError("no such file")):
        with pytest.raises(SystemAudioUnavailableError, match="could not be started"):
            source.start(lambda arrival, mono: None)


def test_parse_tap_header_reads_the_documented_line():
    assert parse_tap_header(header_line(44100, 1)) == (44100, 1)


def test_a_header_without_the_tap_stream_index_is_refused(tap_settings):
    """The one thing the byte stream cannot reveal: a helper reading buffer 0
    blindly captures the headset microphone instead of the system output, at the
    same rate and channel count. A helper that does not name the buffer it
    derived is not trusted to be reading the tap at all."""
    stdout = (
        json.dumps({"sample_rate": 48000, "channels": 2, "format": "f32le"}).encode() + b"\n"
    )
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=FakeTapProcess(stdout)):
        with pytest.raises(SystemAudioUnavailableError, match="which input buffer"):
            source.start(lambda arrival, mono: None)


def test_a_non_zero_tap_stream_index_is_accepted():
    """The headset case is the normal case, not an error: the sub-device's own
    input streams come first, so the tap sits at a non-zero index."""
    assert parse_tap_header(header_line(48000, 2, tap_stream_index=1)) == (48000, 2)


def test_the_helper_is_launched_with_the_configured_block_size(tap_settings):
    source = MacOSTapSource(tap_settings, Path("/opt/justsay/justsay-audiotap"))
    process = FakeTapProcess(tap_stdout(blocks=1))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process) as popen:
        source.start(lambda arrival, mono: None)
        source.stop()

    assert popen.call_args[0][0] == [
        str(Path("/opt/justsay/justsay-audiotap")),
        "--block-frames",
        str(BLOCK_FRAMES),
    ]




@pytest.mark.asyncio
async def test_a_helper_that_dies_mid_capture_leaves_a_wav_and_a_prompt_stop(tap_settings):
    """AC: `stop()` returns in under 2 s and the WAV holds the blocks that did
    arrive, within one block of what the helper managed to write."""
    delivered_blocks = 6
    process = FakeTapProcess(
        tap_stdout(blocks=delivered_blocks), returncode=3, stderr=b"tap died\n"
    )
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    recorder = MeetingRecorder(tap_settings)

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process), patch(
        "app.audio.meeting_recorder.create_system_audio_source", return_value=source
    ), patch("app.audio.meeting_recorder.sd.InputStream") as stream:
        stream.return_value = MagicMock()
        await recorder.start()
        source._reader.join(timeout=2.0)
        time.sleep(0.05)
        started = time.monotonic()
        audio_path = await recorder.stop()
        elapsed = time.monotonic() - started

    assert elapsed < 2.0

    with wave.open(str(audio_path), "rb") as wf:
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    block_samples = BLOCK_FRAMES * tap_settings.sample_rate // 48000
    expected = delivered_blocks * block_samples

    assert abs(int(np.count_nonzero(samples)) - expected) <= block_samples


def test_stop_is_safe_before_start(tap_settings):
    MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap")).stop()


def test_stop_terminates_the_helper(tap_settings):
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    process = FakeTapProcess(tap_stdout(blocks=1))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None)
        source.stop()

    assert process.terminated is True




def test_an_override_wins_unconditionally():
    override = Path("/somewhere/else/justsay-audiotap")

    assert resolve_audio_tap_path(Path("/anything"), override) == override


def test_a_frozen_sidecar_resolves_to_its_sibling_resource():
    """AC: `…/Resources/justsay-backend/justsay-backend` →
    `…/Resources/justsay-audiotap`."""
    resources = Path("/Applications/JustSay.app/Contents/Resources")
    executable = resources / "justsay-backend" / "justsay-backend"

    resolved = resolve_audio_tap_path(executable, None)

    assert resolved == resources / "justsay-audiotap"


def test_a_dev_tree_resolves_to_the_swiftpm_build_output():
    resolved = resolve_audio_tap_path(Path("/usr/bin/python3"), None)

    assert resolved.parts[-4:] == ("JustSayAudioTap", ".build", "release", "justsay-audiotap")


def test_resolve_audio_tap_path_executes_nothing(tmp_path):
    """AC: locating the helper never runs it — a path that does not exist is
    still resolved, and no subprocess is created."""
    with patch("app.audio.macos_tap.subprocess.Popen") as popen:
        resolve_audio_tap_path(tmp_path / "justsay-backend" / "justsay-backend", None)

    popen.assert_not_called()




def test_the_factory_builds_a_macos_source_on_darwin(tap_settings, tmp_path):
    """AC: the darwin branch is covered on the ubuntu runner, with the platform
    string injected rather than read."""
    settings = tap_settings.model_copy(
        update={"meeting_macos_tap_path": tmp_path / "justsay-audiotap"}
    )

    source = create_system_audio_source(settings, platform_name="darwin")

    assert isinstance(source, MacOSTapSource)


def test_the_factory_returns_none_on_a_platform_with_no_source(tap_settings):
    assert create_system_audio_source(tap_settings, platform_name="linux") is None


def test_the_factory_swallows_a_construction_failure_into_none(tap_settings):
    with patch("app.audio.macos_tap.MacOSTapSource", side_effect=RuntimeError("boom")):
        assert create_system_audio_source(tap_settings, platform_name="darwin") is None


def test_the_factory_reads_sys_platform_when_none_is_injected(monkeypatch, tap_settings):
    monkeypatch.setattr(sys, "platform", "linux")

    assert create_system_audio_source(tap_settings) is None

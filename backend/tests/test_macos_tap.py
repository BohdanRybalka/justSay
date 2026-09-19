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
import logging
import os
import sys
import threading
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.audio import macos_tap
from app.audio.config import AudioSettings
from app.audio.macos_tap import (
    _STDERR_MAX_LINE_BYTES,
    _STDERR_TAIL_LINES,
    MacOSTapSource,
    parse_tap_header,
    resolve_audio_tap_path,
)
from app.audio.meeting_recorder import MeetingRecorder
from app.audio.system_source import SystemAudioUnavailableError, create_system_audio_source

BLOCK_FRAMES = 64
NEWLINE = b"\n"

def stream_cut(tail_bytes: int, channels: int = 2) -> str:
    return (
        f"the macOS system-audio helper stopped delivering usable audio -- its "
        f"stream ended {tail_bytes} bytes into a "
        f"{BLOCK_FRAMES * channels * 4}-byte block"
    )


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


class _GateNeverOpened(BaseException):
    """A test forgot to open the stdout gate, said where the reader cannot eat it.

    Derived from `BaseException` rather than from `Exception` because the read
    happens inside `_deliver_until_refused`, whose whole point is that nothing
    foreign escapes it: an `AssertionError` from here was caught there and
    turned into "the macOS system-audio capture failed with an unexpected
    AssertionError", so a forgotten gate failed five seconds later on a
    mismatched list of reasons instead of saying what went wrong.
    """


class _GatedStdout(io.BytesIO):
    """Stdout that holds the audio body back until the gate is opened.

    The header still arrives immediately, which is what `MacOSTapSource.start`
    blocks on, but the blocks arrive only when the test says so. A real
    helper's audio arrives over the length of a call; an ungated BytesIO
    hands every block a near-identical arrival, which is not what a call
    looks like on the timeline.

    The wait's result is raised on rather than discarded: a gate that is never
    opened has to fail the test that forgot to open it, not delay every read
    by the timeout and then hand the body over anyway.
    """

    def __init__(self, data: bytes):
        super().__init__(data)
        self.gate = threading.Event()

    def read(self, size: int = -1) -> bytes:
        if not self.gate.wait(timeout=5.0):
            raise _GateNeverOpened("the test never opened the stdout gate")
        return super().read(size)


class _FakeTapProcess:
    """A helper process whose whole life is a byte string on stdout.

    `reaped` is the one thing here that is not a convenience. A real `Popen`
    answers `poll()` with None for a child that has died and not yet been
    reaped, and that is precisely the state a crashed helper is in when the
    reader observes it: stdout hit EOF microseconds ago. `wait()` is what
    turns the state into a number. A fake whose `poll()` always answered a
    code made that difference invisible, so no test here could see the one
    helper death this module can actually observe.
    """

    def __init__(
        self,
        stdout: bytes,
        returncode: int = 0,
        stderr: bytes = b"",
        gated: bool = False,
        reaped: bool = True,
    ):
        self.stdout = _GatedStdout(stdout)
        if not gated:
            self.stdout.gate.set()
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.reaped = reaped
        self.exit_observed = threading.Event()

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        self.reaped = True
        self.exit_observed.set()
        return self.returncode

    def poll(self) -> int | None:
        self.exit_observed.set()
        return self.returncode if self.reaped else None


def tap_stdout(blocks: int, channels: int = 2, fill: float = 0.25, **header) -> bytes:
    frames = np.full(blocks * BLOCK_FRAMES * channels, fill, dtype="<f4")
    return header_line(channels=channels, **header) + frames.tobytes()


def tap_stdout_cut(blocks: int, tail_bytes: int, channels: int = 2) -> bytes:
    """A helper stream whose last bytes stop part-way through a block.

    The one framing failure this side can observe, written as the bytes that
    produce it rather than injected by patching the deinterleave. A helper
    writing whole blocks through `writeAll` cannot leave a partial block
    behind and carry on -- it retries until every byte is out, or stops -- so
    any tail at all is where the evidence is, whatever its length, and
    `tests/test_cross_language_contracts` is what holds the Swift to that.
    """
    return tap_stdout(blocks=blocks, channels=channels) + b"\x00" * tail_bytes


@pytest.fixture
def tap_source(tap_settings):
    """Build sources that are stopped in teardown, whatever the assertions do.

    `source.stop()` as the last statement of a test is skipped by the first
    failing assertion above it, and what that leaks is a helper thread still
    spinning inside `os.write` and two descriptors nobody closes. The next
    test in the file then fails for a reason that is not its own.
    """
    sources: list[MacOSTapSource] = []

    def build() -> MacOSTapSource:
        source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
        sources.append(source)
        return source

    yield build
    for source in sources:
        source.stop()


class _FakeClock:
    """A monotonic clock this test advances by hand.

    `time.monotonic()` is what the deadline arithmetic under test is made of,
    and reading it for real makes the assertion a measurement: on a Windows
    runner it can answer the same value twice in a row, and the float sum then
    lands a rounding error above an exact budget.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def read(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def helper_pipe():
    """A real OS pipe whose ends are closed in teardown.

    A `BytesIO` cannot fill, so it cannot show a helper's write ending rather
    than blocking; a real pipe can, and leaks two descriptors when an
    assertion above the cleanup fails.
    """
    opened: list = []

    def build():
        read_fd, write_fd = os.pipe()
        reader = open(read_fd, "rb")
        opened.append((reader, write_fd))
        return reader, write_fd

    yield build
    for reader, write_fd in opened:
        try:
            reader.close()
        except OSError:
            pass
        try:
            os.close(write_fd)
        except OSError:
            pass


def run_source(settings: AudioSettings, process: _FakeTapProcess) -> list[np.ndarray]:
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
    process = _FakeTapProcess(tap_stdout(blocks=3))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None)
        source._reader.join(timeout=2.0)
        source.stop()

    assert source.native_sample_rate == 48000
    assert tap_settings.sample_rate == 16000


def test_k_frames_of_stdout_become_exactly_k_over_block_frames_mono_blocks(tap_settings):
    """AC: block count and block length both follow from the header."""
    received = run_source(tap_settings, _FakeTapProcess(tap_stdout(blocks=5)))

    assert len(received) == 5
    assert {len(block) for block in received} == {BLOCK_FRAMES}


def test_a_trailing_partial_block_is_dropped_rather_than_padded(tap_settings):
    stdout = tap_stdout(blocks=2) + np.full(BLOCK_FRAMES, 0.1, dtype="<f4").tobytes()

    received = run_source(tap_settings, _FakeTapProcess(stdout))

    assert len(received) == 2


def test_interleaved_stereo_is_downmixed_to_mono(tap_settings):
    frames = np.tile(
        np.array([1.0, 0.0], dtype="<f4"), BLOCK_FRAMES
    )
    process = _FakeTapProcess(header_line(channels=2) + frames.tobytes())

    received = run_source(tap_settings, process)

    assert len(received) == 1
    assert received[0].tolist() == pytest.approx([0.5] * BLOCK_FRAMES)


def test_a_mono_helper_stream_is_passed_through(tap_settings):
    received = run_source(tap_settings, _FakeTapProcess(tap_stdout(blocks=2, channels=1)))

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
    process = _FakeTapProcess(stdout)

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

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=_FakeTapProcess(stdout)):
        with pytest.raises(SystemAudioUnavailableError, match="which input buffer"):
            source.start(lambda arrival, mono: None)


def test_a_non_zero_tap_stream_index_is_accepted():
    """The headset case is the normal case, not an error: the sub-device's own
    input streams come first, so the tap sits at a non-zero index."""
    assert parse_tap_header(header_line(48000, 2, tap_stream_index=1)) == (48000, 2)


def test_the_helper_is_launched_with_the_configured_block_size(tap_settings):
    source = MacOSTapSource(tap_settings, Path("/opt/justsay/justsay-audiotap"))
    process = _FakeTapProcess(tap_stdout(blocks=1))

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
    process = _FakeTapProcess(
        tap_stdout(blocks=delivered_blocks),
        returncode=3,
        stderr=b"tap died\n",
        gated=True,
    )
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    recorder = MeetingRecorder(tap_settings)

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process), patch(
        "app.audio.meeting_recorder.create_system_audio_source", return_value=source
    ), patch("app.audio.meeting_recorder.sd.InputStream") as stream:
        stream.return_value = MagicMock()
        await recorder.start()
        process.stdout.gate.set()
        source._reader.join(timeout=2.0)
        time.sleep(0.05)
        started = time.monotonic()
        audio_path = (await recorder.stop()).path
        elapsed = time.monotonic() - started

    assert elapsed < 2.0

    with wave.open(str(audio_path), "rb") as wf:
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    block_samples = BLOCK_FRAMES * tap_settings.sample_rate // 48000
    expected = delivered_blocks * block_samples

    assert abs(int(np.count_nonzero(samples)) - expected) <= block_samples


def test_stop_is_safe_before_start(tap_settings):
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with patch("app.audio.macos_tap.subprocess.Popen") as popen:
        source.stop()
        source.stop()

    assert popen.call_args_list == [], (
        "stopping a source that never started must not reach for a helper process "
        f"to tear down: {popen.call_args_list}"
    )


def test_stop_terminates_the_helper(tap_settings):
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    process = _FakeTapProcess(tap_stdout(blocks=1))

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


def test_the_factory_reports_why_a_construction_failed(tap_settings):
    """A darwin machine that cannot open its tap is not an unsupported platform.

    This case asserted the failure was swallowed into None until JS-78, which
    is how the caller came to tell a macOS user that meeting recording
    requires macOS.
    """
    with patch("app.audio.macos_tap.MacOSTapSource", side_effect=RuntimeError("boom")):
        with pytest.raises(SystemAudioUnavailableError, match="boom"):
            create_system_audio_source(tap_settings, platform_name="darwin")


def test_the_factory_reads_sys_platform_when_none_is_injected(monkeypatch, tap_settings):
    monkeypatch.setattr(sys, "platform", "linux")

    assert create_system_audio_source(tap_settings) is None


_STDERR_LINE_BYTES = 1024
_STDERR_LINE_COUNT = 1024


def _stderr_writer(write_fd: int, last_line: bytes) -> None:
    """Write 1 MB of log lines into the helper's end of a real OS pipe."""
    filler = b"x" * (_STDERR_LINE_BYTES - 1) + b"\n"
    with os.fdopen(write_fd, "wb") as stream:
        for _ in range(_STDERR_LINE_COUNT - 1):
            stream.write(filler)
        stream.write(last_line)


@pytest.mark.timeout(30)
def test_a_helper_writing_more_stderr_than_the_pipe_holds_is_not_blocked(tap_settings):
    """AC: the drain runs for the life of the process, not only after it dies.

    The timeout marker is the whole point of the regression: without the drain
    this test does not fail, it parks the writer forever and takes the pytest
    run with it. `pytest-timeout` turns that hang back into a failure with a
    stack, which is what the docstring below always claimed happened.

    The pipe here is a real `os.pipe()`, so its buffer is the OS one -- a few
    tens of kilobytes. A writer pushing 1 MB through it finishes only if
    something is reading concurrently; without the drain thread it parks in
    `write` forever, which is the defect ADR 052 records (the Swift helper
    would stop producing audio instead of stopping its logging).

    The tail assertion is what stops this passing vacuously: a platform whose
    pipe buffer swallowed the whole megabyte would let the writer finish with
    no reader at all, and could not also produce the last line here.

    The drain is joined before that assertion because the writer finishing says
    nothing about the reader having caught up: the writer's last `write`
    returns as soon as the bytes are in the pipe. Windows happened to win that
    race and Linux did not, which is how this test passed locally and failed on
    CI at `9209027`. The join makes the order a fact rather than a scheduling
    accident -- the writer closes its end, so the drain reaches EOF and returns.
    """
    read_fd, write_fd = os.pipe()
    last_line = b"y" * (_STDERR_LINE_BYTES - 1) + b"\n"
    process = _FakeTapProcess(tap_stdout(blocks=1))
    process.stderr = os.fdopen(read_fd, "rb")
    writer = threading.Thread(
        target=_stderr_writer, args=(write_fd, last_line), name="fake-helper", daemon=True
    )
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None)
        writer.start()
        writer.join(timeout=5.0)
        writer_finished = not writer.is_alive()
        source._stderr_reader.join(timeout=5.0)
        source._reader.join(timeout=2.0)
        source.stop()

    assert writer_finished, (
        "the helper's 1 MB of stderr filled the pipe and blocked its writer, "
        "which is exactly what the drain thread exists to prevent"
    )
    assert source._stderr_text().endswith(last_line.decode().rstrip()), (
        "the drain read the pipe but kept nothing, so the assertion above "
        "would have passed on a pipe buffer large enough to swallow 1 MB"
    )


@pytest.mark.timeout(30)
def test_a_nonzero_exit_is_logged_with_what_the_helper_wrote(
    tap_settings, caplog, monkeypatch
):
    """AC: the exit diagnostic still carries the helper's own stderr text.

    Draining into a bounded buffer replaced the read-it-all-at-exit call, so
    this pins that the replacement did not turn the error log into an empty
    tail -- the only place a user-visible reason for a failed capture appears.

    The stderr here is a real `os.pipe()` whose writer holds its line back
    until the exit has been observed, and that ordering is the test rather than
    a detail. Against an `io.BytesIO` the line is in the buffer before the
    reader thread even starts, so the log carries it whether or not
    `_read_blocks` joins the drain first -- the test passed identically with the
    join deleted, over ten consecutive runs. What the join closes is the race
    where the helper's last words arrive after its exit is observed, and the
    fake helper's `poll()` is what says that moment has come: the write is
    ordered after it by an event rather than by a sleep that a loaded runner
    can overrun. The join budget is raised for the same reason -- the default
    0.5 s is a real bound on a 2-core runner, and a red here would report a
    missing diagnostic that is really a scheduling delay.
    """
    monkeypatch.setattr(macos_tap, "_READER_JOIN_TIMEOUT_SECONDS", 5.0)
    read_fd, write_fd = os.pipe()
    process = _FakeTapProcess(tap_stdout(blocks=1), returncode=3)
    process.stderr = os.fdopen(read_fd, "rb")

    def write_after_the_exit_is_observed() -> None:
        assert process.exit_observed.wait(timeout=10.0), "the helper's exit was never read"
        os.write(write_fd, b"tap died: no permission" + NEWLINE)
        os.close(write_fd)

    writer = threading.Thread(
        target=write_after_the_exit_is_observed, name="fake-helper-stderr", daemon=True
    )
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with caplog.at_level(logging.ERROR, logger="app.audio.macos_tap"):
        with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
            source.start(lambda arrival, mono: None)
            writer.start()
            source._reader.join(timeout=10.0)
            source.stop()

    writer.join(timeout=2.0)
    messages = [record.getMessage() for record in caplog.records]
    assert any("tap died: no permission" in message for message in messages), (
        "the exit was logged before the drain had the helper's last line, so the "
        f"user is told a capture failed with no reason attached: {messages}"
    )
    assert any("exited with code 3" in message for message in messages), messages


def test_a_helper_that_exits_tells_the_recorder_the_far_side_is_gone(tap_settings):
    """AC: the helper's non-zero exit is what becomes `system_audio_ended`.

    The exit report already logged; the recorder now hears it too, and this
    drives that call site rather than the abstract sink, because the argument
    and the lock read in `_read_blocks` are the half of the acceptance
    criterion no other test reaches.
    """
    process = _FakeTapProcess(tap_stdout(blocks=1), returncode=3)
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    reported: list[str] = []

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None, reported.append)
        source._reader.join(timeout=5.0)
        source.stop()

    assert reported == ["the macOS system-audio helper exited with code 3"], (
        f"a helper that died mid-meeting told the recorder {reported}, so the "
        f"meeting keeps its indicator clean while system audio is gone"
    )


@pytest.mark.parametrize("tail_bytes", [1, 7, 8, 504])
def test_a_stream_cut_short_of_a_block_is_reported_rather_than_read_as_a_clean_end(
    tap_settings, tap_source, tail_bytes
):
    """AC: a helper stream this side cannot frame reaches the recorder.

    `_read_blocks` runs on a daemon thread nobody joins for a result, so a
    raise inside it ends system audio and tells no one: the meeting reports a
    healthy capture while only the microphone is still arriving.

    Driven by the bytes rather than by patching the deinterleave, which is
    what the four rounds before this one could not do. The deinterleave cannot
    refuse anything this module hands it -- `block_bytes` and the channel
    count come from the same header field, so every full chunk divides evenly
    by construction, and a test that injected a refusal was asserting its own
    `side_effect`. What the helper can leave behind is a tail shorter than a
    whole block, and that is what each of these lengths is.

    Parametrised because two of the four -- 8 and 504 -- are whole numbers of
    two-channel float32 frames. The check these replace read
    `len(chunk) % frame_bytes`, so a cut landing on a frame boundary, one in
    eight of them, reached nobody at all: the recording kept its healthy
    marker while the far side was gone. A single length proved nothing about
    the other seven, and the one this file happened to use was misaligned.
    """
    process = _FakeTapProcess(tap_stdout_cut(blocks=3, tail_bytes=tail_bytes))
    source = tap_source()
    received: list[np.ndarray] = []
    reported: list[str] = []

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: received.append(mono), reported.append)
        reader = source._reader
        reader.join(timeout=5.0)

    assert not reader.is_alive()
    assert len(received) == 3, "the whole blocks before the cut were dropped too"
    assert reported == [stream_cut(tail_bytes)], (
        f"a helper whose stream was cut {tail_bytes} bytes into a block told "
        f"the recorder {reported}, so the meeting keeps its indicator clean "
        f"while what it recorded of the far side ends where nobody can place it"
    )


@pytest.mark.timeout(30)
def test_a_cut_stream_leaves_the_loop_so_the_exit_report_still_runs(
    tap_settings, tap_source, caplog
):
    """Leaving the loop is what keeps the report below it reachable.

    The helper's own last words on stderr are the only channel separating a
    revoked recording permission from a Core Audio error, and they are read by
    the exit report under the `while`. A thread that died inside the loop threw
    that half away.

    The helper exits 0 here, which is what it really does on this path:
    leaving the loop closes the read end, and `flushWholeBlocks` answers a
    closed stdout with `logLine("stdout is closed")` and `exit(0)`. Pinning a
    3 instead hid the whole defect -- an exit report that returned on a zero
    dropped the stderr tail on the one path that closes the pipe itself.

    Which reason the recorder hears is asserted, not just how many: the first
    one, with the exit carried by the log line beside it. A count alone passed
    whether the sink survived the first report or was taken and cleared by it.
    """
    process = _FakeTapProcess(
        tap_stdout_cut(blocks=3, tail_bytes=7),
        returncode=0,
        stderr=b"fail: the tap was invalidated\n",
    )
    source = tap_source()
    reported: list[str] = []

    with (
        caplog.at_level(logging.ERROR, logger="app.audio.macos_tap"),
        patch("app.audio.macos_tap.subprocess.Popen", return_value=process),
    ):
        source.start(lambda arrival, mono: None, reported.append)
        source._reader.join(timeout=5.0)

    messages = [record.getMessage() for record in caplog.records]
    assert any("the tap was invalidated" in message for message in messages), (
        f"the helper exited 0 because this side closed its stdout, and the exit "
        f"report read that as nothing to say, so its own account of the framing "
        f"is gone: {messages}"
    )
    assert reported == [stream_cut(7)]


@pytest.mark.timeout(30)
def test_the_pipe_is_closed_before_the_recorder_is_told_why(tap_settings, tap_source):
    """The helper stops being wedged first, and hears the reason afterwards.

    The failure sink is `MeetingRecorder._note_incident`, which takes the
    recorder's lock and can wait behind whatever else holds it. Until the read
    end is closed the helper is parked inside `write()` on a full pipe still
    holding its Core Audio tap, so reporting first put the length of that wait
    between the refusal and the release -- for a report that needs nothing
    from the pipe.

    The sink waits for the close rather than the test timing the two, so the
    order is asserted rather than sampled: reporting first leaves the wait to
    expire.
    """
    process = _FakeTapProcess(tap_stdout_cut(blocks=1, tail_bytes=7), returncode=0)
    closed = threading.Event()
    process.stdout.close = closed.set
    source = tap_source()
    pipe_was_closed_first: list[bool] = []

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(
            lambda arrival, mono: None,
            lambda reason: pipe_was_closed_first.append(closed.wait(timeout=2.0)),
        )
        source._reader.join(timeout=5.0)

    assert pipe_was_closed_first == [True], (
        "the recorder was told why before the read end was closed, so the "
        "helper stayed wedged inside write() with its tap held for as long as "
        "the sink took"
    )


@pytest.mark.timeout(30)
def test_a_block_sink_that_raises_is_reported_once_and_keeps_delivering(
    tap_settings, tap_source
):
    """The contract both platforms are held to, on the path that is reachable.

    `MeetingRecorder._system_callback` measures the block's level and writes
    it to a spill queue; nothing in this module owns it or can promise it will
    not raise. This thread is joined for no result, so a raise out of it ends
    system audio and tells nobody -- the meeting reports a healthy capture
    while only the microphone is still arriving.

    Reported once and survived, not treated as the end of the capture. One
    transient raise out of foreign code says nothing about whether the helper
    is still producing audio, and ending delivery on it costs the far side the
    rest of the meeting. The Windows twin asserts the same three offers; this
    one ended the capture on the first raise while the base class's docstring
    said it did not.
    """
    process = _FakeTapProcess(tap_stdout(blocks=3), returncode=1)
    source = tap_source()
    reported: list[str] = []
    attempted: list[int] = []

    def raise_from_the_sink(arrival, mono):
        attempted.append(len(mono))
        raise RuntimeError("the recorder's own callback failed")

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(raise_from_the_sink, reported.append)
        reader = source._reader
        reader.join(timeout=5.0)

    assert not reader.is_alive()
    assert attempted == [BLOCK_FRAMES] * 3, (
        f"the sink was offered {len(attempted)} of 3 blocks, so one raise out "
        f"of the recorder's callback ended system audio for the whole meeting"
    )
    assert reported == [
        "the macOS system-audio capture failed with an unexpected RuntimeError",
        "the macOS system-audio helper exited with code 1",
    ], reported


@pytest.mark.timeout(30)
def test_a_repeating_block_sink_failure_costs_one_traceback_not_one_per_block(
    tap_settings, tap_source, caplog
):
    """A sink that keeps raising must not be logged once a block.

    Delivery deliberately continues, so the handler is re-entered on every
    block: at 1024-frame blocks that is ~47 traceback formats and log writes a
    second on the capture thread for the rest of the meeting, which is itself
    enough to make the capture drop audio the log then blames the helper for.
    The report is claimed once; this pins that the log is too.
    """
    process = _FakeTapProcess(tap_stdout(blocks=5), returncode=0)
    source = tap_source()

    def raise_from_the_sink(arrival, mono):
        raise RuntimeError("the recorder's own callback failed")

    with (
        caplog.at_level(logging.ERROR, logger="app.audio.system_source"),
        patch("app.audio.macos_tap.subprocess.Popen", return_value=process),
    ):
        source.start(raise_from_the_sink, lambda reason: None)
        source._reader.join(timeout=5.0)

    tracebacks = [record for record in caplog.records if record.exc_info]
    assert len(tracebacks) == 1, (
        f"five blocks into a sink that always raises cost {len(tracebacks)} "
        f"tracebacks, which a realtime capture thread pays for in dropped audio"
    )


@pytest.mark.timeout(30)
def test_a_tap_failure_sink_that_raises_still_closes_the_pipe_and_reports_the_exit(
    tap_settings, tap_source, caplog
):
    """The report is the last thing this source can do about a failed capture,
    so it cannot be the thing that stops the rest of the sequence.

    Driven through the cut stream, which is where a raise here costs
    something: the report comes first, and closing the pipe the helper is
    wedged in and reading its last words both come after it.
    """
    process = _FakeTapProcess(
        tap_stdout_cut(blocks=1, tail_bytes=7), returncode=3, stderr=b"fail: gone\n"
    )
    source = tap_source()

    def raise_from_the_failure_sink(reason):
        raise RuntimeError("the recorder refused the report")

    with (
        caplog.at_level(logging.ERROR, logger="app.audio.macos_tap"),
        patch("app.audio.macos_tap.subprocess.Popen", return_value=process),
    ):
        source.start(lambda arrival, mono: None, raise_from_the_failure_sink)
        source._reader.join(timeout=5.0)
        stdout_closed = process.stdout.closed

    messages = [record.getMessage() for record in caplog.records]
    assert stdout_closed, (
        "a raising sink left the helper writing into a pipe nobody reads, so it "
        "parks inside write() still holding its Core Audio tap"
    )
    assert any("fail: gone" in message for message in messages), messages


@pytest.mark.timeout(30)
def test_a_sink_that_raises_on_the_framing_still_hears_the_helper_s_exit(
    tap_settings, tap_source
):
    """One raising report must not cost the recorder the other reason.

    Both reasons this source can observe are stops, so they share one claim,
    and a claim spent on a report nobody received would drop the exit silently.
    `_report_capture_failure` spends a kind only when the sink took it.
    """
    process = _FakeTapProcess(tap_stdout_cut(blocks=1, tail_bytes=7), returncode=3)
    source = tap_source()
    heard: list[str] = []

    def refuse_the_first_report(reason: str) -> None:
        if not heard:
            heard.append(reason)
            raise RuntimeError("the recorder refused the report")
        heard.append(reason)

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None, refuse_the_first_report)
        source._reader.join(timeout=5.0)

    assert heard == [
        stream_cut(7),
        "the macOS system-audio helper exited with code 3",
    ], heard


@pytest.mark.timeout(30)
def test_each_recording_reports_through_a_tap_source_of_its_own(tap_settings):
    """"Once" is scoped to the source, because a source is one recording.

    `MeetingRecorder._begin_capture` calls `create_system_audio_source` every
    time it opens a meeting, so the second meeting gets an object whose
    report-once flag was never set. The test this replaces started one source
    twice and asserted the flag was cleared in between; nothing does that, and
    its Windows twin pinned a restart a terminated PyAudio instance cannot
    perform at all.
    """
    captures: list[list[str]] = []

    for _ in range(2):
        source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
        process = _FakeTapProcess(tap_stdout(blocks=1), returncode=3)
        reported: list[str] = []
        with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
            source.start(lambda arrival, mono: None, reported.append)
            source._reader.join(timeout=5.0)
            source.stop()
        captures.append(reported)

    assert captures == [
        ["the macOS system-audio helper exited with code 3"],
        ["the macOS system-audio helper exited with code 3"],
    ]


@pytest.mark.timeout(30)
def test_a_helper_that_died_but_is_not_reaped_yet_still_reports(tap_settings):
    """The one helper death this module can observe, seen the way it arrives.

    A helper that crashes closes stdout on the way down, so this side meets
    the death as EOF: `_read_exactly` comes back short, the delivery loop
    returns no reason, and the exit is read microseconds later — before the kernel has
    reaped the child. `poll()` answers None there. Returning on that answer
    left the crash reported to nobody: nothing was raised, nothing was logged,
    and the meeting went on showing a healthy system capture with only the
    microphone still arriving.
    """
    process = _FakeTapProcess(tap_stdout(blocks=1), returncode=3, reaped=False)
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    reported: list[str] = []

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None, reported.append)
        source._reader.join(timeout=5.0)
        source.stop()

    assert reported == ["the macOS system-audio helper exited with code 3"], (
        f"a helper that crashed and had not been reaped told the recorder "
        f"{reported}, so the meeting keeps its indicator clean while system "
        f"audio is gone"
    )


@pytest.mark.timeout(30)
def test_a_helper_that_stops_writing_without_exiting_is_reported_anyway(
    tap_settings, caplog
):
    """No exit code is not the same as no failure.

    The stream ended and the helper is still up, which is a capture that has
    stopped delivering whatever the process table says. The report names that
    instead of a number, and it is sent rather than skipped, because the
    number was never what the recorder needed.
    """
    process = _FakeTapProcess(tap_stdout(blocks=1), returncode=3, reaped=False)

    def never_exits(timeout=None):
        raise macos_tap.subprocess.TimeoutExpired(cmd="justsay-audiotap", timeout=timeout)

    process.wait = never_exits
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    reported: list[str] = []

    with (
        caplog.at_level(logging.ERROR, logger="app.audio.macos_tap"),
        patch("app.audio.macos_tap.subprocess.Popen", return_value=process),
    ):
        source.start(lambda arrival, mono: None, reported.append)
        source._reader.join(timeout=5.0)
        source.stop()

    assert reported == [
        "the macOS system-audio helper had not exited 0.5s after its stdout "
        "was closed"
    ], reported
    messages = [record.getMessage() for record in caplog.records]
    assert any("had not exited" in message for message in messages), messages


def test_a_reader_leaving_during_a_stop_waits_for_no_exit_code(tap_settings):
    """A deliberate stop has nothing to learn from the helper's exit.

    `stop()` kills the helper, so neither its code nor its stderr says
    anything about the capture -- and this thread is one of the ones
    `_shutdown` is joining against `_READER_JOIN_TIMEOUT_SECONDS`. A wait here
    spends that budget getting this thread classified as parked on a pipe it
    is not reading, after which both descriptors go to the detached closer and
    an otherwise clean teardown pays the whole join.
    """
    process = _FakeTapProcess(b"", returncode=3, reaped=False)
    waited: list[object] = []

    def record_the_wait(timeout=None):
        waited.append(timeout)
        return 3

    process.wait = record_the_wait
    drain = threading.Thread(target=lambda: None, name="finished-drain", daemon=True)
    drain.start()
    drain.join(timeout=5.0)

    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    reported: list[str] = []
    with source._lock:
        source._on_failure = reported.append
    source._stopping.set()

    source._report_exit(process, drain, refused=False)

    assert waited == [], "a deliberate stop waited on the helper it had just killed"
    assert reported == [], "a deliberate stop was reported to the recorder as a failure"


def test_the_exit_report_spends_one_join_budget_on_the_wait_and_the_drain(
    tap_settings, tap_source, monkeypatch
):
    """The two waits after the loop share one deadline rather than one each.

    This thread is one of the ones `_shutdown` joins against
    `_READER_JOIN_TIMEOUT_SECONDS`. A wait for the exit and a join on the drain
    that each held that whole budget put it past the deadline on a teardown
    that was going cleanly: a `stop()` landing an instruction into the wait
    found the reader still running when the shared join expired, so both pipes
    went to the detached closer and the stop paid the budget twice.

    Driven against a clock this test advances, so the two budgets are exact
    numbers rather than measurements. Read off the real clock they were not:
    `time.monotonic()` on a Windows runner can answer the same value either
    side of `deadline = time.monotonic() + budget`, and `(t + 0.1) - t` is
    0.10000000000582077 for a `t` of that size -- so the sum this asserts on
    exceeded the budget by 2e-14 and reddened CI on a correct implementation.
    A tolerance would have swallowed the defect too, since spending two full
    budgets is a factor of two rather than a rounding error. The comparison is
    exact because the clock is.
    """
    monkeypatch.setattr(macos_tap, "_READER_JOIN_TIMEOUT_SECONDS", 0.1)
    clock = _FakeClock()
    monkeypatch.setattr(macos_tap.time, "monotonic", clock.read)
    budgets: list[float] = []

    def spend_the_whole_budget(timeout=None):
        budgets.append(timeout)
        clock.advance(timeout)
        raise macos_tap.subprocess.TimeoutExpired(cmd="justsay-audiotap", timeout=timeout)

    class _RecordingDrain:
        name = "recording-drain"

        def join(self, timeout=None):
            budgets.append(timeout)

        def is_alive(self):
            return False

    process = _FakeTapProcess(b"", returncode=3, reaped=False)
    process.wait = spend_the_whole_budget
    source = tap_source()

    source._report_exit(process, _RecordingDrain(), refused=False)

    assert budgets == [0.1, 0.0], (
        f"the exit wait and the drain join were given {budgets} out of one "
        f"{macos_tap._READER_JOIN_TIMEOUT_SECONDS}s budget, so a stop "
        f"overlapping them finds this reader parked past the budget it shares "
        f"with every other reader"
    )


@pytest.mark.timeout(30)
def test_a_restarted_tap_source_reports_the_second_capture_that_failed(tap_settings):
    """`stop()` leaves this source usable, so its report-once claim cannot outlive a capture.

    Nothing here is terminated the way Windows terminates its PyAudio instance:
    `stop()` kills a helper and `start()` spawns another, clearing `_stopping`
    on the way in. A claim taken in `__init__` and never released therefore made
    every capture after the first silent -- the second meeting's helper could
    die and the recorder would be told nothing while its indicator stayed clean.
    """
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))
    captures: list[list[str]] = []

    for _ in range(2):
        process = _FakeTapProcess(tap_stdout(blocks=1), returncode=3)
        reported: list[str] = []
        with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
            source.start(lambda arrival, mono: None, reported.append)
            source._reader.join(timeout=5.0)
            source.stop()
        captures.append(reported)

    assert captures == [
        ["the macOS system-audio helper exited with code 3"],
        ["the macOS system-audio helper exited with code 3"],
    ], captures


@pytest.mark.timeout(30)
def test_a_cleared_block_sink_closes_the_pipe_the_helper_writes_into(
    tap_settings, tap_source, helper_pipe
):
    """Leaving the loop is what closes the pipe, not the reason for leaving.

    A sink cleared under the reader is one of three ways out of the delivery
    loop, and it used to be the one that skipped the close: the helper went on
    producing into a pipe nobody reads, filled it within about eight blocks and
    parked inside `write()` still holding its Core Audio tap. It survived only
    because `stop()` is today the sole writer of `_on_block = None` and
    terminates immediately afterwards — an ordering guarantee living in
    another method, which is not a property of this one.

    Driven through a real OS pipe, because a `BytesIO` cannot fill and so
    cannot show a write ending rather than blocking.
    """
    reader_end, write_fd = helper_pipe()
    stopped_writing = threading.Event()
    delivered = threading.Event()

    def keep_producing() -> None:
        block = np.zeros(BLOCK_FRAMES * 2, dtype="<f4").tobytes()
        try:
            os.write(write_fd, header_line(channels=2))
            while True:
                os.write(write_fd, block)
        except OSError:
            pass
        finally:
            stopped_writing.set()

    process = _FakeTapProcess(b"", returncode=0)
    process.stdout = reader_end
    helper = threading.Thread(target=keep_producing, name="fake-helper", daemon=True)
    source = tap_source()

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        helper.start()
        source.start(lambda arrival, mono: delivered.set(), lambda reason: None)
        reader = source._reader
        assert delivered.wait(timeout=5.0), "the fake helper delivered no block at all"
        with source._lock:
            source._on_block = None
        reader.join(timeout=5.0)
        wedged = not stopped_writing.wait(timeout=5.0)

    assert not reader.is_alive()
    assert not wedged, (
        "the helper is still blocked inside write() with its Core Audio tap "
        "held, which is where SIGTERM will find it"
    )


def test_a_helper_that_exits_cleanly_reports_no_failure(tap_settings, tap_source):
    """A helper reaching the end of its stream is not a failed capture.

    The stream ends on a block boundary here, which is the only way a helper
    that was not cut can end: `flushWholeBlocks` writes
    `pending.count - (pending.count % blockSamples)` samples and `writeAll`
    puts every one of them out, so whatever this side reads is a whole number
    of blocks and a clean end reads zero bytes. This test and the cut-stream
    one above it are the two halves of that: nothing left over is silence,
    anything left over is a cut. The version this replaces ended on half a
    block and called it clean, which is the premise that let a frame-aligned
    cut pass as an ordinary end of stream.
    """
    process = _FakeTapProcess(tap_stdout(blocks=1), returncode=0)
    source = tap_source()
    reported: list[str] = []

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None, reported.append)
        source._reader.join(timeout=5.0)

    assert reported == []


@pytest.mark.timeout(30)
def test_a_reader_that_fails_after_the_loop_still_tells_the_recorder(
    tap_settings, tap_source
):
    """The three calls under the delivery loop are inside the guard too.

    Only the loop used to be, so a raise in the close, the report or the exit
    read ended this daemon thread in exactly the silence the whole path exists
    to prevent: system audio over, `on_failure` never called, the meeting
    still reporting a healthy capture.

    `Popen.wait` is the concrete one. `_exit_code` catches only
    `TimeoutExpired`, and a child already reaped elsewhere answers
    `ChildProcessError` -- which `stop()` does from `_terminate`, on the
    recorder's device worker, concurrently with this wait by design.
    """
    process = _FakeTapProcess(tap_stdout(blocks=1), returncode=3, reaped=False)

    def reaped_by_somebody_else(timeout=None):
        raise ChildProcessError("no child processes")

    process.wait = reaped_by_somebody_else
    source = tap_source()
    reported: list[str] = []

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None, reported.append)
        reader = source._reader
        reader.join(timeout=5.0)

    assert not reader.is_alive()
    assert reported == [
        "the macOS system-audio capture failed with an unexpected ChildProcessError"
    ], (
        f"the reader thread died after the loop and told the recorder {reported}, "
        f"so the meeting keeps its indicator clean while system audio is gone"
    )


class _PipedTapProcess:
    """A helper process whose stdout and stderr are real OS pipes.

    `_FakeTapProcess`'s in-memory streams cannot reproduce anything this file
    now has to prove: a `BytesIO` never blocks its writer, never blocks a
    reader inside `readline()`, and reaches EOF the moment it is exhausted. A
    real pipe does all three, which is what a helper process is.
    """

    def __init__(self, returncode: int = 0):
        stdout_read, self.stdout_write = os.pipe()
        stderr_read, self.stderr_write = os.pipe()
        self.stdout = os.fdopen(stdout_read, "rb")
        self.stderr = os.fdopen(stderr_read, "rb")
        self.returncode = returncode
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        return None

    def wait(self, timeout=None) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def close_writes(self) -> None:
        for fd in (self.stdout_write, self.stderr_write):
            try:
                os.close(fd)
            except OSError:
                pass

    def close_reads(self) -> None:
        for stream in (self.stdout, self.stderr):
            try:
                stream.close()
            except OSError:
                pass


@pytest.mark.timeout(30)
def test_stop_returns_even_though_a_reader_is_parked_on_the_helpers_pipe(tap_settings):
    """AC: `stop()` returns while a drain thread is blocked reading stderr.

    `stop()` runs on the owner thread, inside `MeetingRecorder.stop()`'s
    `finally`. Closing a `BufferedReader` that another thread is parked inside
    `readline()` on waits for that reader's buffer lock, so a close used as the
    way to end the read hangs the owner thread instead: the meeting stays in
    `STOPPING` and no further recording can start. Terminating the helper is
    what ends the read; a reader still alive after the join keeps its stream
    open rather than being closed into that wait (ADR 052).

    The helper here never exits and never closes its write ends, so both
    readers are genuinely blocked for the whole of `stop()`.
    """
    process = _PipedTapProcess()
    os.write(process.stdout_write, header_line())
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        source.start(lambda arrival, mono: None)
        readers = [source._reader, source._stderr_reader]
        stopper = threading.Thread(target=source.stop, name="owner-stop", daemon=True)
        stopper.start()
        stopper.join(timeout=4.0)
        returned = not stopper.is_alive()

    process.close_writes()
    for reader in readers:
        reader.join(timeout=2.0)
    process.close_reads()

    assert returned, (
        "stop() was still running after 4 s, so closing the helper's pipes "
        "deadlocked on a reader parked in readline() and the meeting can never "
        "leave STOPPING"
    )


@pytest.mark.timeout(30)
def test_the_helper_can_fill_the_stderr_pipe_before_it_writes_its_header(tap_settings):
    """AC: stderr is drained from the spawn, not from after the header.

    Core Audio setup is the phase the helper logs during, and the header is
    written only once that setup has succeeded -- so the 5 s header window is
    exactly the window in which an unread pipe fills. A helper that fills it
    parks in `write()` before writing its header, and the user is then told the
    helper never answered and that it is probably a permissions problem.

    The fake helper here writes 1 MB of log lines and only then its header, so
    it completes only if something read stderr while the header was still
    being waited for.
    """
    process = _PipedTapProcess()
    filler = b"x" * (_STDERR_LINE_BYTES - 1) + b"\n"

    def _helper() -> None:
        for _ in range(_STDERR_LINE_COUNT):
            os.write(process.stderr_write, filler)
        os.write(process.stdout_write, header_line())

    helper = threading.Thread(target=_helper, name="fake-helper", daemon=True)
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        helper.start()
        source.start(lambda arrival, mono: None)
        readers = [source._reader, source._stderr_reader]
        source.stop()

    helper.join(timeout=2.0)
    process.close_writes()
    for reader in readers:
        reader.join(timeout=2.0)
    process.close_reads()

    assert source.native_sample_rate == 48000


@pytest.mark.timeout(30)
def test_a_startup_failure_reports_what_the_helper_wrote(tap_settings):
    """AC: a refused start carries the helper's own words, not a guess.

    13 of the helper's 15 stderr sites are `fail(...)` on the way out, and that
    line is the only thing separating a denied recording permission from a Core
    Audio error. Every startup failure path -- header timeout, non-JSON header,
    wrong format, missing `tap_stream_index` -- used to discard it unread.
    """
    process = _PipedTapProcess(returncode=1)
    os.write(process.stderr_write, b"fail: system audio recording permission denied\n")
    process.close_writes()
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
        with pytest.raises(SystemAudioUnavailableError, match="permission denied") as caught:
            source.start(lambda arrival, mono: None)

    process.close_reads()

    assert "before writing its header" in str(caught.value)


def test_a_helper_that_never_writes_a_newline_cannot_grow_the_buffer_without_bound(
    tap_settings,
):
    """AC: the drain buffer is bounded in bytes, not only in lines.

    `maxlen` bounds how many objects the deque holds, and a stream with no
    newline in it is one object -- `readline()` returns only when it finds the
    newline or reaches EOF, so a helper logging without newlines accumulates
    the whole recording into a single string. The read is capped instead.
    """
    payload = b"x" * (_STDERR_MAX_LINE_BYTES * (_STDERR_TAIL_LINES + 5))
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    source._drain_stderr(io.BytesIO(payload))
    tail = source._stderr_text()

    assert len(tail) <= _STDERR_TAIL_LINES * (_STDERR_MAX_LINE_BYTES + 1)
    assert max(len(line) for line in tail.split("\n")) <= _STDERR_MAX_LINE_BYTES


def test_the_drain_logs_a_failure_it_did_not_expect_instead_of_dying_silently(
    tap_settings, caplog
):
    """AC: a drain that stops early says so.

    `RuntimeError` from a closed buffered object, `MemoryError`, an
    `AttributeError` on a stream that is not what this module assumes -- any of
    them used to end the thread with a bare `pass`, which reinstates the very
    defect ADR 052 exists to remove and leaves no trace that it happened.
    """
    class _BrokenStream:
        def readline(self, size: int = -1) -> bytes:
            raise RuntimeError("readline of closed file")

    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with caplog.at_level(logging.WARNING, logger="app.audio.macos_tap"):
        source._drain_stderr(_BrokenStream())

    messages = [record.getMessage() for record in caplog.records]
    assert any("stopped being read" in message for message in messages), messages


@pytest.mark.timeout(30)
def test_the_exit_diagnostic_uses_the_drain_thread_it_was_handed(
    tap_settings, caplog, monkeypatch
):
    """AC: the exit log joins the drain thread it was actually given.

    `_read_blocks` used to reach for `self._stderr_reader` with no lock while
    `stop()` cleared it with none either, so a stop racing the helper's exit
    made the join disappear and the log reported an empty tail -- the empty
    diagnostic the join was added to close. The thread is handed to the reader
    as an argument instead, so there is no shared field to race on.

    Clearing the field is how that race is reproduced, and the field is put
    back before `stop()` runs. `stop()` builds `_shutdown`'s reader list from
    it, and `_shutdown` closes a pipe only when nothing is parked on it: a
    version of this test that left the field cleared would be closing the
    helper's stderr under a live drain thread -- the deadlock ADR 052 exists to
    prevent, disarmed inside the test that exists to protect it. It passed only
    because the writer usually finished first, and under load it hung instead
    of failing.
    """
    monkeypatch.setattr(macos_tap, "_READER_JOIN_TIMEOUT_SECONDS", 5.0)
    read_fd, write_fd = os.pipe()
    process = _FakeTapProcess(tap_stdout(blocks=1), returncode=3, gated=True)
    process.stderr = os.fdopen(read_fd, "rb")

    def write_after_the_exit_is_observed() -> None:
        assert process.exit_observed.wait(timeout=10.0), "the helper's exit was never read"
        os.write(write_fd, b"tap died: no permission" + NEWLINE)
        os.close(write_fd)

    writer = threading.Thread(
        target=write_after_the_exit_is_observed, name="fake-helper-stderr", daemon=True
    )
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with caplog.at_level(logging.ERROR, logger="app.audio.macos_tap"):
        with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
            source.start(lambda arrival, mono: None)
            drain = source._stderr_reader
            source._stderr_reader = None
            writer.start()
            process.stdout.gate.set()
            source._reader.join(timeout=10.0)
            source._stderr_reader = drain
            source.stop()

    writer.join(timeout=2.0)
    messages = [record.getMessage() for record in caplog.records]
    assert any("tap died: no permission" in message for message in messages), messages


class _ParkedReader:
    """A thread parked until the test releases it, standing in for a reader.

    A drain blocked inside `readline()` on a pipe whose writer has not gone
    away is exactly this: alive, holding a stream's buffer lock, and not
    leaving on its own.
    """

    def __init__(self, name: str):
        self.released = threading.Event()
        self.thread = threading.Thread(
            target=lambda: self.released.wait(timeout=30.0), name=name, daemon=True
        )
        self.thread.start()


@pytest.mark.timeout(30)
def test_shutdown_spends_one_join_budget_on_all_the_readers_together(
    tap_settings, monkeypatch
):
    """AC: the join budget is shared, not handed out per reader.

    `_shutdown` runs on `MeetingRecorder`'s single device worker (ADR 048), the
    thread every lifecycle transition is serialised through, so every second it
    spends is a second the awaiting `stop()` request waits and the next
    `start()` cannot begin. A per-reader budget multiplies that by however many
    readers there are, and it kept evaluating readers after one was already
    known stuck.
    """
    monkeypatch.setattr(macos_tap, "_READER_JOIN_TIMEOUT_SECONDS", 0.5)
    process = _PipedTapProcess()
    parked = [_ParkedReader(f"parked-{index}") for index in range(4)]
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    started = time.monotonic()
    source._shutdown(process, [reader.thread for reader in parked])
    elapsed = time.monotonic() - started

    for reader in parked:
        reader.released.set()
        reader.thread.join(timeout=5.0)
    process.close_writes()
    process.close_reads()

    assert elapsed < 1.0, (
        f"_shutdown took {elapsed:.2f}s over four parked readers, so each one "
        "is being given the full join budget on the thread that owns the app"
    )


@pytest.mark.timeout(30)
def test_a_pipe_left_open_for_a_parked_reader_is_closed_once_it_leaves(
    tap_settings, monkeypatch
):
    """AC: a stuck reader delays the close instead of leaking the descriptor.

    `stop()` drops `_process`, `_reader` and `_stderr_reader` before
    `_shutdown` runs, so a `_shutdown` that returned without closing left two
    pipe descriptors with nothing in the object able to reach them again --
    open until the `Popen` was collected, which a thread still parked on the
    pipe prevents. The wait is handed to a thread of its own so it can be
    unbounded without freezing the caller.
    """
    monkeypatch.setattr(macos_tap, "_READER_JOIN_TIMEOUT_SECONDS", 0.2)
    process = _PipedTapProcess()
    parked = _ParkedReader("parked-drain")
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    source._shutdown(process, [parked.thread])

    assert not process.stdout.closed, (
        "the pipe was closed while a reader was still parked on it, which is "
        "the close-into-a-blocked-read deadlock ADR 052 records"
    )
    parked.released.set()
    parked.thread.join(timeout=5.0)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not (process.stdout.closed and process.stderr.closed):
        time.sleep(0.02)
    process.close_writes()

    assert process.stdout.closed and process.stderr.closed, (
        "the reader left and nothing ever closed the helper's pipes, so two "
        "descriptors leak for every meeting that stopped with a stuck reader"
    )


def test_one_long_line_costs_the_buffer_one_entry_not_twenty_five(tap_settings):
    """AC: what is bounded is what is kept, not what is read.

    Reading in `_STDERR_MAX_LINE_BYTES` slices and appending every slice turns
    a single 100 KB Core Audio dump into 25 deque entries, which evicts the
    `fail(...)` line the buffer exists to preserve -- the buffer doing the
    opposite of its job. The remainder of an over-long line is read (the pipe
    still has to drain) and dropped.
    """
    filler = b"x" * (_STDERR_MAX_LINE_BYTES * 25) + NEWLINE
    lines = [b"the first thing the helper said" + NEWLINE, filler]
    lines += [b"noise %d" % index + NEWLINE for index in range(_STDERR_TAIL_LINES - 3)]
    lines.append(b"fail: system audio recording permission denied" + NEWLINE)
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    source._drain_stderr(io.BytesIO(b"".join(lines)))
    tail = source._stderr_text().split(NEWLINE.decode())

    assert tail[0] == "the first thing the helper said", (
        f"the long line evicted everything written before it: {tail[:3]}"
    )
    assert tail[-1].startswith("fail:"), tail[-1]
    assert len(tail) == _STDERR_TAIL_LINES


@pytest.mark.timeout(30)
def test_a_deliberate_stop_is_not_reported_as_a_failed_capture(tap_settings, caplog):
    """AC: `stop()` kills the helper, so its exit code is not a diagnostic.

    `_terminate` is what ends the helper, and a terminated child exits
    non-zero, so the exit path fired on every normal stop -- telling the user a
    meeting had been recorded without system audio when nothing had gone wrong.
    It also put the block reader inside a join while `_shutdown` was joining
    the block reader, spending the shutdown's whole budget on the thread that
    owns the app and then classifying it as parked on a pipe it had finished
    reading.
    """
    process = _PipedTapProcess(returncode=-15)
    os.write(process.stdout_write, header_line())
    source = MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap"))

    with caplog.at_level(logging.ERROR, logger="app.audio.macos_tap"):
        with patch("app.audio.macos_tap.subprocess.Popen", return_value=process):
            source.start(lambda arrival, mono: None)
            readers = [source._reader, source._stderr_reader]
            source.stop()
            process.close_writes()
            for reader in readers:
                reader.join(timeout=5.0)

    process.close_reads()
    messages = [record.getMessage() for record in caplog.records]
    assert not any("recorded without system audio" in message for message in messages), (
        f"a deliberate stop was reported to the user as a failed capture: {messages}"
    )

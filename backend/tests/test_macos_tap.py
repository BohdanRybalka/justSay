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


class _GatedStdout(io.BytesIO):
    """Stdout that holds the audio body back until the gate is opened.

    The header still arrives immediately, which is what `MacOSTapSource.start`
    blocks on, but the blocks arrive only when the test says so. A real
    helper's audio arrives over the length of a call; an ungated BytesIO
    hands every block a near-identical arrival, which is not what a call
    looks like on the timeline.

    The wait's result is asserted rather than discarded: a gate that is never
    opened has to fail the test that forgot to open it, not delay every read
    by the timeout and then hand the body over anyway.
    """

    def __init__(self, data: bytes):
        super().__init__(data)
        self.gate = threading.Event()

    def read(self, size: int = -1) -> bytes:
        assert self.gate.wait(timeout=5.0), "the test never opened the stdout gate"
        return super().read(size)


class _FakeTapProcess:
    """A helper process whose whole life is a byte string on stdout."""

    def __init__(
        self, stdout: bytes, returncode: int = 0, stderr: bytes = b"", gated: bool = False
    ):
        self.stdout = _GatedStdout(stdout)
        if not gated:
            self.stdout.gate.set()
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.exit_observed = threading.Event()

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        return self.returncode

    def poll(self) -> int:
        self.exit_observed.set()
        return self.returncode


def tap_stdout(blocks: int, channels: int = 2, fill: float = 0.25, **header) -> bytes:
    frames = np.full(blocks * BLOCK_FRAMES * channels, fill, dtype="<f4")
    return header_line(channels=channels, **header) + frames.tobytes()


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
    MacOSTapSource(tap_settings, Path("/nonexistent/justsay-audiotap")).stop()


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
        assert process.exit_observed.wait(timeout=10.0), "the helper's exit was never polled"
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
        assert process.exit_observed.wait(timeout=10.0), "the helper's exit was never polled"
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

    `_shutdown` runs inline on the event-loop thread -- `pipeline/router.py`
    awaits `recorder.start()` without a `to_thread` -- so every second it
    spends is a second the whole backend is frozen, which is the freeze JS-99
    already filed. A per-reader budget multiplies that by however many readers
    there are, and it kept evaluating readers after one was already known
    stuck.
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
    while time.monotonic() < deadline and not process.stdout.closed:
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

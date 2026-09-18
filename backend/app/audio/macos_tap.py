"""macOS system audio, from the bundled `justsay-audiotap` helper (ADR 041).

Built from `macos/JustSayAudioTap`; its Swift header is the other half of this
contract, so changing one alone makes the other wrong.
    stdout: {"sample_rate":48000,"channels":2,"format":"f32le","tap_stream_index":0}\\n
            then raw interleaved little-endian float32 frames, forever
    stderr: log lines, one per line
    SIGTERM: flush whole blocks, exit 0
"""

from __future__ import annotations

import collections
import json
import logging
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from app.audio.analysis import interleaved_buffer_to_mono
from app.audio.config import AudioSettings
from app.audio.system_source import (
    SAMPLE_BYTES,
    SAMPLE_DTYPE,
    BlockSink,
    CaptureFailure,
    FailureSink,
    SystemAudioSource,
    SystemAudioUnavailableError,
)

log = logging.getLogger(__name__)

TAP_EXECUTABLE_NAME = "justsay-audiotap"
SIDECAR_DIRECTORY_NAME = "justsay-backend"
SAMPLE_FORMAT = "f32le"
ENDPOINT_NAME = "macOS system audio"

_TERMINATE_TIMEOUT_SECONDS = 0.5
_KILL_TIMEOUT_SECONDS = 0.5
_READER_JOIN_TIMEOUT_SECONDS = 0.5
_HEADER_TIMEOUT_SECONDS = 5.0
_STDERR_TAIL_LINES = 20
_STDERR_MAX_LINE_BYTES = 4096

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEV_TAP_PATH = (
    _REPO_ROOT / "macos" / "JustSayAudioTap" / ".build" / "release" / TAP_EXECUTABLE_NAME
)


def resolve_audio_tap_path(executable: Path, override: Path | None) -> Path:
    """Where the tap helper lives, without executing anything.

    `override` wins unconditionally; a frozen sidecar resolves to its sibling
    `…/Resources/justsay-audiotap`, anything else to the SwiftPM build output.
    """
    if override is not None:
        return Path(override)
    if executable.parent.name == SIDECAR_DIRECTORY_NAME:
        return executable.parent.parent / TAP_EXECUTABLE_NAME
    return _DEV_TAP_PATH


def _start_reader(target, name: str) -> threading.Thread:
    """A started daemon thread, handed back so the caller can shut it down.

    Whoever tears the helper down must know which readers are still parked
    inside a read before closing any pipe (ADR 052).
    """
    reader = threading.Thread(target=target, name=name, daemon=True)
    reader.start()
    return reader


def _await_header(reader: threading.Thread, header: list[bytes]) -> bytes:
    """The helper's first line, or a raise once `_HEADER_TIMEOUT_SECONDS` passes.

    Bounded because the caller is reached from an `async def`. On the timeout
    path `reader` is still parked in `readline()`, so that pipe must not be closed.
    """
    reader.join(timeout=_HEADER_TIMEOUT_SECONDS)
    if not header:
        raise SystemAudioUnavailableError(
            f"The macOS audio helper did not answer within {_HEADER_TIMEOUT_SECONDS:.0f}s. "
            "It may be waiting on a system-audio recording permission that was never granted."
        )
    return header[0]


def parse_tap_header(line: bytes) -> tuple[int, int]:
    """The helper's first stdout line, as (sample_rate, channels)."""
    if not line:
        raise SystemAudioUnavailableError(
            "The macOS system-audio helper exited before writing its header"
        )
    try:
        header = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SystemAudioUnavailableError(
            f"The macOS system-audio helper wrote a header that is not JSON: {line!r}"
        ) from e
    if not isinstance(header, dict):
        raise SystemAudioUnavailableError(
            f"The macOS system-audio helper wrote a non-object header: {header!r}"
        )
    if header.get("format") != SAMPLE_FORMAT:
        raise SystemAudioUnavailableError(
            f"The macOS system-audio helper reported sample format "
            f"{header.get('format')!r}, not {SAMPLE_FORMAT!r}"
        )
    try:
        sample_rate = int(header["sample_rate"])
        channels = int(header["channels"])
    except (KeyError, TypeError, ValueError) as e:
        raise SystemAudioUnavailableError(
            f"The macOS system-audio helper header is missing a usable rate or "
            f"channel count: {header!r}"
        ) from e
    if sample_rate <= 0 or channels <= 0:
        raise SystemAudioUnavailableError(
            f"The macOS system-audio helper reported {sample_rate} Hz and "
            f"{channels} channels"
        )
    try:
        tap_stream_index = int(header["tap_stream_index"])
    except (KeyError, TypeError, ValueError) as e:
        raise SystemAudioUnavailableError(
            f"The macOS system-audio helper did not say which input buffer it "
            f"reads, so there is nothing proving it is the tap and not a "
            f"microphone: {header!r}"
        ) from e
    if tap_stream_index < 0:
        raise SystemAudioUnavailableError(
            f"The macOS system-audio helper reported input buffer "
            f"{tap_stream_index}, which does not exist"
        )
    return sample_rate, channels


class MacOSTapSource(SystemAudioSource):
    """Everything the Mac is playing, read as float32 frames off a helper's stdout."""

    _capture_name = "the macOS system-audio capture"

    def __init__(self, settings: AudioSettings, tap_path: Path):
        super().__init__()
        self._settings = settings
        self._tap_path = Path(tap_path)
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stderr_lock = threading.Lock()
        self._stopping = threading.Event()
        self._stderr_tail: collections.deque[str] = collections.deque(
            maxlen=_STDERR_TAIL_LINES
        )
        self._on_block: BlockSink | None = None
        self._native_sample_rate = settings.sample_rate
        self._channels = 1

    @property
    def native_sample_rate(self) -> int:
        return self._native_sample_rate

    @property
    def endpoint_name(self) -> str:
        return ENDPOINT_NAME

    def start(self, on_block: BlockSink, on_failure: FailureSink | None = None) -> None:
        self._stopping.clear()
        process = self._spawn()
        stderr_reader = _start_reader(
            lambda: self._drain_stderr(process.stderr), "macos-audio-tap-stderr"
        )
        header: list[bytes] = []
        header_reader = _start_reader(
            lambda: header.append(process.stdout.readline()), "macos-audio-tap-header"
        )
        readers = [stderr_reader, header_reader]
        try:
            self._native_sample_rate, self._channels = parse_tap_header(
                _await_header(header_reader, header)
            )
        except SystemAudioUnavailableError as e:
            self._shutdown(process, readers)
            raise SystemAudioUnavailableError(self._failure_message(str(e))) from e
        except Exception:
            self._shutdown(process, readers)
            raise

        with self._lock:
            self._on_block = on_block
        self._begin_failure_reports(on_failure)
        self._process = process
        self._stderr_reader = stderr_reader
        self._reader = _start_reader(
            lambda: self._read_blocks(process, stderr_reader), "macos-audio-tap"
        )
        log.info(
            "macOS system-audio tap started: %d Hz, %d ch",
            self._native_sample_rate,
            self._channels,
        )

    def _spawn(self) -> subprocess.Popen:
        try:
            return subprocess.Popen(
                [str(self._tap_path), "--block-frames", str(self._settings.meeting_block_frames)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            raise SystemAudioUnavailableError(
                f"The macOS system-audio helper at {self._tap_path} could not be started: {e}"
            ) from e

    def _read_blocks(
        self, process: subprocess.Popen, stderr_reader: threading.Thread
    ) -> None:
        """Frames until the helper stops producing them, then why it stopped.

        Runs on a daemon thread nobody joins, so every step is inside the catch.
        The pipe is closed on every way out of the loop, before anything is reported.
        """
        try:
            reason = self._deliver_until_refused(process)
            self._stop_reading(process)
            if reason is not None:
                self._report_capture_failure(reason, CaptureFailure.STOPPED)
            self._report_exit(process, stderr_reader, refused=reason is not None)
        except Exception as failure:
            self._report_callback_failure(failure)

    def _deliver_until_refused(self, process: subprocess.Popen) -> str | None:
        """Blocks to the sink until the helper stops, or why reading stopped. Never raises.

        None when the helper ran out cleanly or the sink was cleared, otherwise the
        sentence the recorder hears. A raise out of the sink does not leave the loop.
        """
        block_bytes = self._settings.meeting_block_frames * self._channels * SAMPLE_BYTES
        stdout = process.stdout
        try:
            while True:
                chunk = _read_exactly(stdout, block_bytes)
                if len(chunk) < block_bytes:
                    return self._ran_out(len(chunk), block_bytes)
                with self._lock:
                    sink = self._on_block
                if sink is None:
                    return None
                arrival = time.monotonic()
                mono = interleaved_buffer_to_mono(chunk, self._channels, SAMPLE_DTYPE)
                self._deliver_to_sink(sink, arrival, mono)
        except Exception as failure:
            log.exception("Reading the macOS system-audio helper failed")
            return (
                f"the macOS system-audio capture failed with an unexpected "
                f"{type(failure).__name__}"
            )

    def _ran_out(self, partial_block_bytes: int, block_bytes: int) -> str | None:
        """Why the helper's stream ended, when it ended short of a block.

        The helper writes whole blocks only, so any tail at all is proof the
        stream was cut; None for a clean end and for a deliberate `stop()`.
        """
        if not partial_block_bytes or self._stopping.is_set():
            return None
        return (
            f"the macOS system-audio helper stopped delivering usable audio -- its "
            f"stream ended {partial_block_bytes} bytes into a {block_bytes}-byte block"
        )

    def _stop_reading(self, process: subprocess.Popen) -> None:
        """Close the pipe the helper is writing into, so it exits.

        Safe from this thread and only from this thread: it is the sole reader
        of that pipe and has already left the read (ADR 052).
        """
        try:
            if process.stdout is not None:
                process.stdout.close()
        except OSError:
            log.debug("Closing the macOS system-audio helper's stdout failed", exc_info=True)

    def _report_exit(
        self,
        process: subprocess.Popen,
        stderr_reader: threading.Thread,
        *,
        refused: bool,
    ) -> None:
        """Why the helper is gone, in its own words, once it actually is.

        Waits for the exit status rather than sampling it, silent during a
        deliberate stop, and silent on a zero exit unless the framing was refused.
        """
        if self._stopping.is_set():
            return
        deadline = time.monotonic() + _READER_JOIN_TIMEOUT_SECONDS
        code = _exit_code(process, deadline)
        if self._stopping.is_set() or (code == 0 and not refused):
            return
        stderr_reader.join(timeout=max(0.0, deadline - time.monotonic()))
        gone = (
            f"exited with code {code}"
            if code is not None
            else f"had not exited {_READER_JOIN_TIMEOUT_SECONDS:.1f}s after its "
            f"stdout was closed"
        )
        log.error(
            "The macOS system-audio helper %s, so this meeting is being "
            "recorded without system audio: %s",
            gone,
            self._stderr_text(),
        )
        self._report_capture_failure(
            f"the macOS system-audio helper {gone}", CaptureFailure.STOPPED
        )

    def _drain_stderr(self, stream: object) -> None:
        """Read the helper's stderr from the moment it is spawned, until EOF.

        A pipe nobody reads fills and the helper blocks inside its own write
        (ADR 052). Never close this stream from another thread: it deadlocks.
        """
        if stream is None:
            return
        try:
            continuation = False
            while True:
                chunk = stream.readline(_STDERR_MAX_LINE_BYTES)
                if not chunk:
                    return
                truncated = continuation
                continuation = not chunk.endswith(b"\n")
                if truncated:
                    continue
                text = chunk.decode("utf-8", errors="replace").rstrip()
                if text:
                    with self._stderr_lock:
                        self._stderr_tail.append(text)
        except (OSError, ValueError):
            log.debug(
                "The macOS system-audio helper's stderr ended while it was being read",
                exc_info=True,
            )
        except Exception:
            log.warning(
                "The macOS system-audio helper's stderr stopped being read, so a "
                "failed capture will be reported without the helper's own words",
                exc_info=True,
            )

    def _stderr_text(self) -> str:
        """The buffered tail of what the helper wrote to stderr."""
        with self._stderr_lock:
            return "\n".join(self._stderr_tail)

    def _failure_message(self, reason: str) -> str:
        """A startup failure, carrying whatever the helper said about it.

        The helper's stderr is the only channel separating a refused recording
        permission from a Core Audio error (ADR 052).
        """
        tail = self._stderr_text()
        if not tail:
            return reason
        return f"{reason} The helper wrote: {tail}"

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            self._on_block = None
        self._end_failure_reports()
        process = self._process
        self._process = None
        readers = [t for t in (self._reader, self._stderr_reader) if t is not None]
        self._reader = None
        self._stderr_reader = None

        if process is not None:
            self._shutdown(process, readers)

    def _shutdown(
        self, process: subprocess.Popen, readers: Sequence[threading.Thread]
    ) -> None:
        """End the helper, then close its pipes once nothing is reading them.

        Killing it is what ends every read; closing a pipe to end one deadlocks
        (ADR 052). Readers still parked at the deadline keep their pipes open.
        """
        self._terminate(process)
        deadline = time.monotonic() + _READER_JOIN_TIMEOUT_SECONDS
        stuck = [reader for reader in readers if not _joined(reader, deadline)]
        if stuck:
            log.warning(
                "The macOS system-audio helper's pipes stay open until %s "
                "finishes reading",
                ", ".join(reader.name for reader in stuck),
            )
            _close_when_idle(process, stuck)
            return
        _close_streams(process)

    def _terminate(self, process: subprocess.Popen) -> None:
        try:
            process.terminate()
            try:
                process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_KILL_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            log.warning("Stopping the macOS system-audio helper failed", exc_info=True)


def _exit_code(process: subprocess.Popen, deadline: float) -> int | None:
    """The helper's exit status, waited for until `deadline`, or None.

    `deadline` is shared with the stderr drain's join that follows rather than
    being a budget of its own.
    """
    try:
        return process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        return None


def _joined(reader: threading.Thread, deadline: float) -> bool:
    """True once `reader` has finished, within a budget shared with its peers."""
    reader.join(timeout=max(0.0, deadline - time.monotonic()))
    return not reader.is_alive()


def _close_streams(process: subprocess.Popen) -> None:
    """Close both of the helper's pipes, with nothing reading either."""
    for stream in (process.stdout, process.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


def _close_when_idle(
    process: subprocess.Popen, readers: Sequence[threading.Thread]
) -> None:
    """Close the helper's pipes once the readers still parked on them are out.

    The wait is unbounded and runs on a new daemon thread, never the caller's:
    `stop()` runs on the thread that owns the recording.
    """

    def _wait_then_close() -> None:
        for reader in readers:
            reader.join()
        _close_streams(process)

    threading.Thread(
        target=_wait_then_close, name="macos-audio-tap-cleanup", daemon=True
    ).start()


def _read_exactly(stream: object, size: int) -> bytes:
    """`size` bytes, or fewer once the stream cannot supply them.

    How short the answer is is the evidence the caller reads: the helper writes
    whole blocks, so any shorter tail says the stream was cut.
    """
    parts: list[bytes] = []
    remaining = size
    while remaining > 0:
        try:
            chunk = stream.read(remaining)
        except (OSError, ValueError):
            break
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


"""macOS system audio, read from the bundled Core Audio tap helper.

The helper is `macos/JustSayAudioTap`, built as `justsay-audiotap` and signed
inside the app bundle. Its contract, which is also written down in the helper's
own header comment:

    justsay-audiotap --block-frames <N>

    stdout: {"sample_rate":48000,"channels":2,"format":"f32le","tap_stream_index":0}\\n
            then raw interleaved little-endian float32 frames, forever
    stderr: log lines, one per line
    SIGTERM: flush whole blocks, exit 0

Nothing in this repository compiles or runs the helper, so this docstring and
the Swift header comment are the two halves of one contract; changing either
alone makes the other wrong.

`tap_stream_index` is required here for exactly that reason. The helper reads
one buffer out of the aggregate device's input list, and buffer 0 is the wrong
one whenever the default output device is a headset that also has a microphone
— the recording then contains the microphone twice and no system audio, at the
right sample rate and channel count, so nothing about the bytes gives it away.
The field is the helper stating which buffer it derived; a helper that cannot
state it is refused at startup rather than trusted.

See docs/adr/041-macos-system-audio-comes-from-a-core-audio-tap.md.
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

from app.audio.config import AudioSettings
from app.audio.system_source import (
    BlockSink,
    FailureSink,
    SystemAudioSource,
    SystemAudioUnavailableError,
)
from app.audio.timeline import interleaved_buffer_to_mono

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

    `override` wins unconditionally. A frozen sidecar at
    `…/Resources/justsay-backend/justsay-backend` resolves to its sibling
    resource `…/Resources/justsay-audiotap`; anything else is a dev tree and
    resolves to the SwiftPM build output.
    """
    if override is not None:
        return Path(override)
    if executable.parent.name == SIDECAR_DIRECTORY_NAME:
        return executable.parent.parent / TAP_EXECUTABLE_NAME
    return _DEV_TAP_PATH


def _start_reader(target, name: str) -> threading.Thread:
    """A started daemon thread, handed back so the caller can shut it down.

    Every thread this module starts reads one of the helper's pipes, and
    whoever tears the helper down has to know which of them are still parked
    inside a read before closing anything (ADR 052). Returning the thread makes
    that a value the caller holds rather than something a function was trusted
    to append to a list it was passed.
    """
    reader = threading.Thread(target=target, name=name, daemon=True)
    reader.start()
    return reader


def _await_header(reader: threading.Thread, header: list[bytes]) -> bytes:
    """The helper's first line, bounded, or a raise saying it never arrived.

    The helper writes its header only after both `AudioHardwareCreateProcessTap`
    and the aggregate device succeed. Spawned but stalled before that -- a
    permission prompt being the obvious candidate -- a plain `readline()` never
    returns, and `MacOSTapSource.start` is reached from an `async def`, so it
    takes the whole backend with it rather than just meeting recording. Every
    other blocking call in this module is already bounded; this was the one
    that was not, and the Windows loopback source has no equivalent.

    On the timeout path `reader` is still parked inside `readline()` on the
    helper's stdout, which is why the caller holds it: that pipe must not be
    closed under a blocked read.
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

    def __init__(self, settings: AudioSettings, tap_path: Path):
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
        self._on_failure: FailureSink | None = None
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
            self._on_failure = on_failure
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

        The exit report joins the drain first, because the helper's last words
        are written on the way out and arrive after its exit is observed. That
        join is skipped once `stop()` has been entered: a deliberate stop kills
        the helper, so a non-zero code there says nothing, and this thread is
        one of the ones `_shutdown` is joining -- sitting inside a join of its
        own would spend the shutdown's whole budget and get itself classified
        as parked on a pipe it is not reading.
        """
        block_bytes = self._settings.meeting_block_frames * self._channels * 4
        stdout = process.stdout
        while True:
            chunk = _read_exactly(stdout, block_bytes)
            if chunk is None:
                break
            with self._lock:
                sink = self._on_block
            if sink is None:
                break
            sink(time.monotonic(), interleaved_buffer_to_mono(chunk, self._channels, "<f4"))

        code = process.poll()
        if code is not None and code != 0 and not self._stopping.is_set():
            stderr_reader.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)
            log.error(
                "The macOS system-audio helper exited with code %d, so this meeting "
                "is being recorded without system audio: %s",
                code,
                self._stderr_text(),
            )
            with self._lock:
                on_failure = self._on_failure
            if on_failure is not None:
                on_failure(
                    f"the macOS system-audio helper exited with code {code}"
                )

    def _drain_stderr(self, stream: object) -> None:
        """Read the helper's stderr from the moment it is spawned.

        A pipe nobody reads fills, and the helper then blocks inside its own
        write instead of producing audio (ADR 052). The read starts before the
        header read does, because the Core Audio setup the helper logs during
        is exactly the phase that precedes the header.

        What is bounded is what is kept, not what is read. Each read stops at
        `_STDERR_MAX_LINE_BYTES`, and the remainder of a line longer than that
        is read and dropped rather than kept as further entries: a single
        100 KB Core Audio dump would otherwise become 25 of the
        `_STDERR_TAIL_LINES` the buffer holds and evict the `fail(...)` line
        the buffer exists to preserve. So one written line is at most one
        entry, of at most `_STDERR_MAX_LINE_BYTES` characters -- characters,
        not bytes, because `errors="replace"` turns each undecodable byte into
        a U+FFFD that re-encodes to three.

        Returns at EOF, which is the helper's exit closing the write end --
        never a close of this stream from another thread, which would deadlock
        on the buffer lock this read holds.
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

        13 of the helper's 15 stderr sites are `fail(...)`, written on the way
        out, and that line is the only channel that separates a refused
        recording permission from a Core Audio error. A startup path that
        discards it leaves the user with a guess (ADR 052).
        """
        tail = self._stderr_text()
        if not tail:
            return reason
        return f"{reason} The helper wrote: {tail}"

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            self._on_block = None
            self._on_failure = None
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

        Killing the helper is what ends every read: the child's exit closes the
        write end and each reader sees EOF. Closing a pipe to end a read
        instead is a deadlock -- `close()` waits on the buffer lock the reader
        holds inside `readline()`, and this runs on the thread that owns the
        recording, so `MeetingRecorder.stop()` never returns and the meeting
        stays in `STOPPING` for the life of the process (ADR 052).

        The readers share one `_READER_JOIN_TIMEOUT_SECONDS` deadline rather
        than getting one each, because this runs on `MeetingRecorder`'s single
        device worker (ADR 048), which serialises every lifecycle transition:
        the time spent here is time the awaiting `stop()` request waits and the
        next `start()` cannot begin, and a per-reader budget multiplies it by
        however many readers there are. It is not the event loop that blocks --
        an earlier draft of this docstring, of ADR 052 and of the test below
        all said it was, citing an `await recorder.start()` in
        `pipeline/router.py` that does not exist; the call is
        `audio/router.py:100` and `:154`, and it awaits a future the worker
        resolves. Whatever is still parked when the deadline passes keeps its pipe
        open here and is handed to `_close_when_idle`, so a stuck reader delays
        the close instead of leaking the descriptor: `stop()` has already
        dropped every reference this object held, and without that handoff
        nothing could ever close them again.
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

    The wait is unbounded and therefore never on the caller's thread: `stop()`
    runs inside `MeetingRecorder.stop()`'s `finally` on the thread that owns
    the recording. A reader parked on a pipe whose writer is gone leaves within
    milliseconds of the helper dying; one that never leaves holds two
    descriptors, which is what the old "return and hope the `Popen` is
    collected" path did for as long as this thread would have.
    """

    def _wait_then_close() -> None:
        for reader in readers:
            reader.join()
        _close_streams(process)

    threading.Thread(
        target=_wait_then_close, name="macos-audio-tap-cleanup", daemon=True
    ).start()


def _read_exactly(stream: object, size: int) -> bytes | None:
    """Exactly `size` bytes, or None once the stream cannot supply them."""
    parts: list[bytes] = []
    remaining = size
    while remaining > 0:
        try:
            chunk = stream.read(remaining)
        except (OSError, ValueError):
            return None
        if not chunk:
            return None
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


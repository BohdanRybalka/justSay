"""The system-audio capture contract and its platform factory.

One abstract class with one implementation per platform, in the same shape as
the STT providers: nothing else in the audio package knows which OS it is
running on. See docs/adr/037-system-audio-capture-is-a-per-platform-source.md
and docs/adr/041-macos-system-audio-comes-from-a-core-audio-tap.md.
"""

from __future__ import annotations

import logging
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import ClassVar

import numpy as np

from app.audio.config import AudioSettings
from app.core.errors import ResourceUnavailableError

log = logging.getLogger(__name__)

BlockSink = Callable[[float, np.ndarray], None]

FailureSink = Callable[[str], None]

SAMPLE_DTYPE = "<f4"

SAMPLE_BYTES = 4


class CaptureFailure(Enum):
    """The ways a capture can go wrong, deduplicated apart from one another.

    A degradation is a stream still delivering while the platform says
    something is wrong with it. A raised callback is foreign code failing on
    the capture thread, which says nothing about whether the device is still
    producing audio. A stop is a source that has ended delivery for the rest
    of the recording.

    Each platform registers only what it can observe. Windows watches a
    PortAudio callback, so it sees a status flag and a raise and never a stop
    -- nothing a WASAPI callback can deliver is unreadable, since PortAudio
    derives the block length from the same mix format the stream was opened
    with. macOS reads a pipe, so it sees a raise and an end of stream and
    never a degradation.
    """

    DEGRADED = "degraded"
    CALLBACK_RAISED = "callback raised"
    STOPPED = "stopped"


class SystemAudioUnavailableError(ResourceUnavailableError):
    """This machine could not open the capture path its platform has.

    A missing loopback device, a denied macOS recording permission, a helper
    binary that would not spawn, a COM call that failed: every raise of this
    class is an open-time failure of one machine, so 503 is the honest answer
    and the next attempt may succeed. The platform having no capture path at
    all is a different fact and has its own subclass.
    """


class SystemAudioUnsupportedError(SystemAudioUnavailableError):
    """This operating system has no system-audio capture path at all.

    Permanent for as long as the app runs on this OS, which is why it keeps
    the 501 its base gave up: `ResourceUnavailableError` promises that
    retrying later may work, and here it never will. See
    docs/adr/060-a-platform-without-audio-is-not-a-broken-device.md.
    """

    status_code: ClassVar[int] = 501
    code: ClassVar[str] = "system_audio_unsupported"


class SystemAudioSource(ABC):
    """Contract: start delivering timestamped mono blocks → stop.

    ``endpoint_name`` answers before ``start()``, and both implementations
    promise that by different means: ``WindowsLoopbackSource.__init__`` opens
    PortAudio and resolves the loopback endpoint there, while ``MacOSTapSource``
    returns a constant — its tap has one name on every Mac — and does not spawn
    its helper until ``start()``. ``MeetingRecorder`` depends on the promise: it
    publishes ``source.endpoint_name`` under the lock that marks the meeting
    live and calls ``start()`` after releasing that lock, because PortAudio
    begins delivering blocks inside ``open()`` and a name published afterwards
    leaves a window answering "not recording" for audio already being kept.

    ``native_sample_rate`` carries no such promise. On macOS it reads back the
    configured rate until ``start()`` parses the tap header, so it is answerable
    only once capture is running, which is why ``MeetingRecorder`` reads it at
    stop time rather than at start.

    Reporting a failed capture is implemented here, not per platform: the two
    implementations had grown the same nine lines twice, with opposite answers
    for a sink that raises and a docstring each explaining why the other one
    was different. ``_capture_name`` is the only part of it a platform still
    owns -- it names the capture in every reason the recorder is handed.
    """

    _capture_name: ClassVar[str] = "this system-audio capture"

    def __init__(self) -> None:
        self._failure_lock = threading.Lock()
        self._failure_sink: FailureSink | None = None
        self._reported: set[CaptureFailure] = set()
        self._logged: set[CaptureFailure] = set()
        self._refusal_logged = False

    @abstractmethod
    def start(self, on_block: BlockSink, on_failure: FailureSink | None = None) -> None:
        """Begin capturing, calling `on_block(arrival_monotonic, mono_block)`.

        `on_failure(reason)` carries whatever the source can observe about a
        capture going wrong -- a helper that exited, a PortAudio status flag,
        a block that is not whole frames, or anything else the capture
        callback raised, named by its exception type. It is optional because
        both implementations are useful without it, and it is a report rather
        than a raise: the meeting keeps recording the microphone, and the
        caller decides what to tell the user. A render endpoint that goes
        quiet with no flag set is indistinguishable from a machine playing
        silence and reaches nobody.

        At most once per `CaptureFailure`, and nothing at all after `stop()`.
        The rule is implemented once, in `_report_capture_failure` below,
        rather than per platform: two copies of it drifted into opposite
        answers for a raise out of the block sink while this paragraph
        described only one of them.

        A raise out of the block sink is forgiven on both platforms. It is
        foreign code failing on the capture thread and says nothing about
        whether the device is still producing audio, so it is reported and
        delivery continues: ending a capture on it would cost the far side the
        rest of the meeting for one transient raise. `_deliver_to_sink` is
        where both sources hand a block over, so neither can end on one.

        `reason` is a diagnostic, not a sentence for a panel: it reaches
        `MeetingRecorder._note_incident`, which logs it and publishes only the
        `CaptureIncident` the user's meeting status carries.

        Neither implementation lets anything escape its capture callback, at
        the cost of a sink that raises being logged and swallowed -- a raise
        crossing a PortAudio callback or ending the macOS reader thread is
        how a failed capture reached nobody in the first place.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and release the device. Safe to call when not started."""

    @property
    @abstractmethod
    def native_sample_rate(self) -> int:
        """The rate the device actually delivers at, before any resampling."""

    @property
    @abstractmethod
    def endpoint_name(self) -> str:
        """What is being captured, named the way the user would recognise it."""

    def _begin_failure_reports(self, on_failure: FailureSink | None) -> None:
        """Take the sink this capture reports to, and its own set of claims.

        Per capture rather than per source, because `MacOSTapSource.stop()`
        leaves its object usable: it kills a helper, and `start()` spawns
        another. A claim held for the life of that source silenced every
        capture after the first. `WindowsLoopbackSource.stop()` terminates its
        PyAudio instance instead, so a second `start()` cannot open a stream
        at all and the release is unobservable there -- which is a fact about
        that one `stop()`, not a reason for the base to hold two rules.
        """
        with self._failure_lock:
            self._failure_sink = on_failure
            self._reported.clear()
            self._logged.clear()
            self._refusal_logged = False

    def _end_failure_reports(self) -> None:
        """Stop reporting, which is what makes `stop()` silent by construction."""
        with self._failure_lock:
            self._failure_sink = None

    def _report_capture_failure(self, reason: str, kind: CaptureFailure) -> None:
        """Hand the recorder a reason this capture went wrong, once per kind.

        Counted per kind rather than together, because a source raising the
        same PortAudio flag on every block must not hand the user a sentence
        per block, and must not silence the later news that it has stopped
        delivering -- which is the order the two arrive in, since the flag is
        raised on the same callback whose sink then raises.

        A kind's one report is spent only when a sink actually took it. No
        sink registered, or a sink that raised, leaves the claim standing:
        "once" means once the recorder has heard it, and a claim spent on a
        report nobody received is this machinery silencing the source on its
        own. The retry that buys is bounded by the caller -- Windows reports
        at most once per kind per callback, macOS at most twice per reader
        pass -- and costs one sink call rather than a traceback, because the
        log below is claimed separately and only once.

        The sink belongs to `MeetingRecorder`, not to this package, so a raise
        out of it is caught here rather than left to unwind a realtime
        callback or a reader thread.
        """
        with self._failure_lock:
            if kind in self._reported:
                return
            on_failure = self._failure_sink
            if on_failure is None:
                return
            self._reported.add(kind)
        try:
            on_failure(reason)
        except Exception:
            with self._failure_lock:
                self._reported.discard(kind)
                say_so = not self._refusal_logged
                self._refusal_logged = True
            if say_so:
                log.exception(
                    "The failure sink for %s raised, so this capture failure reaches "
                    "the recorder as this log line and nothing else",
                    self._capture_name,
                )

    def _claim_failure_log(self, kind: CaptureFailure) -> bool:
        """True the first time this capture logs `kind`.

        Counted apart from the report above, because a sink that refuses the
        report is not a reason to lose the diagnostic, and because a failure
        that repeats per block would otherwise spend a capture thread
        formatting the same traceback ~47 times a second for the rest of the
        meeting -- which produces the input underflow the next log line then
        blames the device for.
        """
        with self._failure_lock:
            already = kind in self._logged
            self._logged.add(kind)
        return not already

    def _report_callback_failure(self, failure: BaseException) -> None:
        """A raise on the capture thread, logged once and reported once.

        Called from inside the `except` that caught it, so the traceback comes
        from the live exception rather than from `failure`.
        """
        if self._claim_failure_log(CaptureFailure.CALLBACK_RAISED):
            log.exception("A capture callback failed on %s", self._capture_name)
        self._report_capture_failure(
            f"{self._capture_name} failed with an unexpected {type(failure).__name__}",
            CaptureFailure.CALLBACK_RAISED,
        )

    def _deliver_to_sink(self, sink: BlockSink, arrival: float, block: np.ndarray) -> None:
        """Hand one block over, reporting a raise out of the sink rather than ending on it.

        The one place either source calls `on_block`, so the contract above --
        a transient raise out of foreign code does not cost the far side the
        rest of the meeting -- is executed rather than described twice.
        """
        try:
            sink(arrival, block)
        except Exception as failure:
            self._report_callback_failure(failure)


def create_system_audio_source(
    settings: AudioSettings, platform_name: str | None = None
) -> SystemAudioSource | None:
    """The system-audio source for this platform.

    Returns None only when this platform has no capture path at all, which is
    a fact about the operating system and the same on every machine running
    it. A platform that *has* one and could not open it raises
    ``SystemAudioUnavailableError`` carrying the reason, because that is a
    fact about this machine and the user is the only person who can act on
    it. Collapsing the two into None is what made the caller tell a Windows
    user that meeting recording requires Windows.

    Both platform modules are imported here and nowhere else: `pyaudiowpatch`
    is a Windows-only wheel, and the macOS source reaches a helper binary that
    exists only inside a macOS bundle.

    `platform_name` is injectable so both branches are covered on the ubuntu
    CI runner.
    """
    platform_name = sys.platform if platform_name is None else platform_name

    try:
        if platform_name == "win32":
            from app.audio.windows_loopback import WindowsLoopbackSource

            return WindowsLoopbackSource(settings)

        if platform_name == "darwin":
            from app.audio.macos_tap import MacOSTapSource, resolve_audio_tap_path

            return MacOSTapSource(
                settings,
                resolve_audio_tap_path(
                    Path(sys.executable), settings.meeting_macos_tap_path
                ),
            )
    except SystemAudioUnavailableError:
        raise
    except Exception as failure:
        log.warning(
            "System-audio capture could not be opened on %s", platform_name, exc_info=True
        )
        raise SystemAudioUnavailableError(
            f"System audio capture could not be opened on this machine: {failure}"
        ) from failure

    return None

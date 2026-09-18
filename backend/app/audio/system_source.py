"""The system-audio capture contract and its platform factory.

One abstract class with one implementation per platform, in the same shape as
the STT providers: nothing else in the audio package knows which OS it is
running on (ADR 037, ADR 041).
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

    A degradation is a stream still delivering while the platform reports a
    problem, a raised callback is foreign code failing on the capture thread,
    and a stop has ended delivery. Each platform registers what it observes.
    """

    DEGRADED = "degraded"
    CALLBACK_RAISED = "callback raised"
    STOPPED = "stopped"


class SystemAudioUnavailableError(ResourceUnavailableError):
    """This machine could not open the capture path its platform has.

    An open-time failure of one machine — a missing loopback device, a denied
    permission, a helper that would not spawn — so 503 is honest and the next
    attempt may succeed. A platform with no path at all has its own subclass.
    """


class SystemAudioUnsupportedError(SystemAudioUnavailableError):
    """This operating system has no system-audio capture path at all.

    Permanent for as long as the app runs on this OS, which is why it answers
    501 rather than the retryable 503 its base promises (ADR 060).
    """

    status_code: ClassVar[int] = 501
    code: ClassVar[str] = "system_audio_unsupported"


class SystemAudioSource(ABC):
    """Contract: start delivering timestamped mono blocks → stop.

    ``endpoint_name`` answers before ``start()``; ``native_sample_rate`` only
    once capture is running. Failure reporting lives here rather than per
    platform, and ``_capture_name`` is the part of it a platform still owns.
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

        `on_failure(reason)` reports what the source can observe about a
        capture going wrong, at most once per `CaptureFailure` and never after
        `stop()`. A raise out of `on_block` is reported, never fatal.
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

        Per capture rather than per source, so a source reused across a
        `stop()`/`start()` pair reports again instead of staying silent.
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

        A kind's one report is spent only once a sink has actually taken it:
        no sink, or a sink that raised, leaves the claim standing. A raise out
        of the sink is caught and logged here rather than left to unwind.
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

        Claimed apart from the report above, so a sink that refuses the report
        does not also cost the diagnostic.
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

        The only route either source takes to `on_block`, so the contract
        above is executed rather than restated per platform.
        """
        try:
            sink(arrival, block)
        except Exception as failure:
            self._report_callback_failure(failure)


def create_system_audio_source(
    settings: AudioSettings, platform_name: str | None = None
) -> SystemAudioSource | None:
    """The system-audio source for this platform.

    ``None`` only when this platform has no capture path at all; a platform
    that has one and could not open it raises ``SystemAudioUnavailableError``
    carrying the reason. ``platform_name`` is injectable for tests.
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

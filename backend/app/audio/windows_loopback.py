"""WASAPI loopback capture of the default Windows render endpoint.

Imported only from `app.audio.system_source.create_system_audio_source`, and
only when `sys.platform == "win32"` — `pyaudiowpatch` is a Windows-only wheel
and must never be imported on macOS or on the ubuntu CI runner.

`pyaudiowpatch` rather than the project's `sounddevice`: PortAudio as shipped
by `sounddevice` has no WASAPI loopback flag at all, and `pyaudiowpatch`
bundles a PortAudio patched for exactly that. See
docs/adr/037-system-audio-capture-is-a-per-platform-source.md.
"""

from __future__ import annotations

import logging
import threading
import time

import pyaudiowpatch as pyaudio

from app.audio.analysis import MalformedCaptureBlockError, interleaved_buffer_to_mono
from app.audio.config import AudioSettings
from app.audio.endpoint_selection import resolve_loopback_device
from app.audio.system_source import (
    BlockSink,
    FailureSink,
    SystemAudioSource,
    SystemAudioUnavailableError,
)
from app.audio.windows_endpoints import render_endpoint_names

log = logging.getLogger(__name__)


def _find_default_loopback(audio: pyaudio.PyAudio, settings: AudioSettings) -> dict:
    """The loopback analogue of the render endpoint the meeting is playing through.

    Teams and Zoom render to the communications endpoint, which is a different
    default from the console one whenever a headset is configured for calls —
    see docs/adr/042-loopback-follows-the-communications-endpoint.md.
    """
    role_names = render_endpoint_names()
    device = resolve_loopback_device(
        role_names,
        list(audio.get_loopback_device_info_generator()),
        settings.meeting_system_endpoint_role,
    )
    if device is None:
        raise SystemAudioUnavailableError(
            f"No WASAPI loopback device found for the default render endpoints "
            f"{role_names} — none of them exposes a loopback analogue"
        )
    return device


class WindowsLoopbackSource(SystemAudioSource):
    """Captures the default render endpoint at its own native mix format.

    The device dictates rate and channel count; nothing is converted here.
    Downmixing to mono is the only work done in the callback, and resampling
    to the pipeline's rate happens later, off the realtime thread, in
    `app.audio.timeline`.
    """

    def __init__(self, settings: AudioSettings):
        self._settings = settings
        self._audio = pyaudio.PyAudio()
        try:
            device = _find_default_loopback(self._audio, settings)
        except Exception:
            self._audio.terminate()
            raise

        self._device_index = int(device["index"])
        self._channels = max(int(device["maxInputChannels"]), 1)
        self._native_sample_rate = int(device["defaultSampleRate"])
        self._endpoint_name = str(device.get("name", "Unknown endpoint"))
        self._stream: object | None = None
        self._on_block: BlockSink | None = None
        self._on_failure: FailureSink | None = None
        self._failure_reported = False
        self._status_logged = False
        self._lock = threading.Lock()
        log.info(
            "WASAPI loopback endpoint: %s (%d Hz, %d ch)",
            device.get("name", "?"),
            self._native_sample_rate,
            self._channels,
        )

    @property
    def native_sample_rate(self) -> int:
        return self._native_sample_rate

    @property
    def endpoint_name(self) -> str:
        return self._endpoint_name

    def _report_capture_failure(self, reason: str) -> None:
        """Hand the recorder the first reason this capture went wrong.

        The one report-once path this source has, shared by everything it can
        observe: a non-zero PortAudio status flag, a block that is not whole
        frames, and anything else the callback raises. All of them, once they
        happen at all, happen on every callback, and the user gets one
        sentence per recording rather than one per block. Per recording and
        not per source: `start()` clears the flag, so a source started a
        second time can report again, which is what macOS already meant by
        "once" and Windows did not.

        Reporting is all this does. Deciding whether a caller has anything to
        say once is `_claim_status_log`'s job, and it keeps its own flag,
        because this one is shared and the caller's need is not.

        The sink belongs to `MeetingRecorder`, not to this module, so a raise
        out of it is caught here rather than left to unwind a realtime
        callback: the report is the last thing this source can do about a
        capture that has already failed.
        """
        with self._lock:
            already = self._failure_reported
            self._failure_reported = True
            on_failure = self._on_failure
        if already or on_failure is None:
            return
        try:
            on_failure(reason)
        except Exception:
            log.exception(
                "The loopback failure sink raised, so this capture failure "
                "reaches the recorder as this log line and nothing else"
            )

    def _claim_status_log(self) -> bool:
        """True the first time this recording sees a PortAudio status flag.

        Its own flag rather than `_report_capture_failure`'s, which is shared
        with the malformed-block path: a malformed block arriving first used
        to claim that one and silence this log for the rest of the recording.
        The log is the only thing separating PortAudio substituting zeros
        from WASAPI handing over a genuinely silent mix, and discarding it
        cost a full diagnosis pass during spec 066, so it does not answer to
        whichever path reached a shared flag first.
        """
        with self._lock:
            already = self._status_logged
            self._status_logged = True
        return not already

    def _stop_delivering(self, reason: str) -> None:
        """End system-audio delivery, having said why once.

        Clearing the block sink is how ``stop()`` already ends delivery, and it
        is what keeps a stream that has begun producing unreadable blocks from
        spending a realtime callback on each one. The stream itself stays open
        until the recorder stops it: the meeting is still running and still
        recording the microphone.
        """
        with self._lock:
            self._on_block = None
        self._report_capture_failure(reason)

    def _report_stream_status(self, status: int) -> None:
        """Log a non-zero PortAudio status flag once per recording.

        Silence arriving from this callback has two very different causes:
        PortAudio substituting zeros on input underflow, which raises
        `paInputUnderflow` here, or WASAPI genuinely handing over a silent
        mix. They are indistinguishable in the samples themselves and this
        flag is the only thing that separates them — discarding it cost a
        full diagnosis pass during spec 066.

        The same flag is the only evidence Windows has that loopback capture
        has degraded, so it is reported to the recorder as well as logged:
        a meeting whose far side stopped arriving is news the user gets while
        the call is still running rather than when they play the file back.
        The two are deduplicated separately, because the report is shared with
        the malformed-block path and the log is not.
        """
        self._report_capture_failure(
            f"the WASAPI loopback stream reported PortAudio status {int(status)}"
        )
        if self._claim_status_log():
            log.warning(
                "WASAPI loopback stream reported PortAudio status %d "
                "(paInputUnderflow=%d) — any silence in this recording may be "
                "substituted rather than captured",
                int(status),
                pyaudio.paInputUnderflow,
            )

    def _deliver_block(self, in_data, status) -> None:
        """One callback's worth of work: report the flag, downmix, hand over."""
        arrival = time.monotonic()
        if status:
            self._report_stream_status(status)
        with self._lock:
            sink = self._on_block
        if sink is not None and in_data:
            sink(arrival, interleaved_buffer_to_mono(in_data, self._channels, "<f4"))

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """Nothing raises out of here, whatever the block or the sink does.

        PortAudio does not report an exception crossing this boundary — it
        tears the stream down, so `on_failure` is never called and the meeting
        goes on reporting a healthy capture while holding the microphone
        alone. The block sink is `MeetingRecorder._system_callback`, which
        measures the block's level and writes it to a spill queue: a caller
        this module neither owns nor can promise about, which is why the
        catch is the whole body rather than the deinterleave alone.

        Both handlers end delivery rather than continuing into whatever comes
        next, because a callback that has failed once fails on every block and
        the stream stays open for the recorder to close.
        """
        try:
            self._deliver_block(in_data, status)
        except MalformedCaptureBlockError as malformed:
            log.exception("The WASAPI loopback stream stopped delivering usable audio")
            self._stop_delivering(
                f"the WASAPI loopback stream stopped delivering usable audio — {malformed}"
            )
        except Exception as failure:
            log.exception("The WASAPI loopback callback failed")
            self._stop_delivering(
                f"the WASAPI loopback capture failed with an unexpected "
                f"{type(failure).__name__}"
            )
        return (None, pyaudio.paContinue)

    def start(self, on_block: BlockSink, on_failure: FailureSink | None = None) -> None:
        with self._lock:
            self._on_block = on_block
            self._on_failure = on_failure
            self._failure_reported = False
            self._status_logged = False
        self._stream = self._audio.open(
            format=pyaudio.paFloat32,
            channels=self._channels,
            rate=self._native_sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self._settings.meeting_block_frames,
            stream_callback=self._stream_callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        with self._lock:
            self._on_block = None
            self._on_failure = None
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                log.warning("Closing the WASAPI loopback stream failed", exc_info=True)
        try:
            self._audio.terminate()
        except Exception:
            log.warning("Terminating the loopback PyAudio instance failed", exc_info=True)

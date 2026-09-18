"""WASAPI loopback capture of the default Windows render endpoint.

Imported only from `app.audio.system_source.create_system_audio_source`, and
only when `sys.platform == "win32"` — `pyaudiowpatch` is a Windows-only wheel
and must never be imported on macOS or on the ubuntu CI runner.

`pyaudiowpatch` rather than `sounddevice`: PortAudio as shipped by
`sounddevice` has no WASAPI loopback flag at all (ADR 037).
"""

from __future__ import annotations

import logging
import threading
import time

import pyaudiowpatch as pyaudio

from app.audio.analysis import interleaved_buffer_to_mono
from app.audio.config import AudioSettings
from app.audio.endpoint_selection import resolve_loopback_device
from app.audio.system_source import (
    SAMPLE_DTYPE,
    BlockSink,
    CaptureFailure,
    FailureSink,
    SystemAudioSource,
    SystemAudioUnavailableError,
)
from app.audio.windows_endpoints import render_endpoint_names

log = logging.getLogger(__name__)


def _find_default_loopback(audio: pyaudio.PyAudio, settings: AudioSettings) -> dict:
    """The loopback analogue of the render endpoint the meeting plays through.

    Teams and Zoom render to the communications endpoint, a different default
    from the console one whenever a headset is configured for calls (ADR 042).
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

    The device dictates rate and channel count; the callback only downmixes,
    and resampling happens later in `app.audio.timeline`. `stop()` is the one
    thing that ends delivery: a status flag and a raised sink are all it sees.
    """

    _capture_name = "the WASAPI loopback capture"

    def __init__(self, settings: AudioSettings):
        super().__init__()
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

    def _report_stream_status(self, status: int) -> None:
        """Log a non-zero PortAudio status flag once per recording.

        The flag is the only thing separating substituted zeros on underflow
        from a genuinely silent WASAPI mix, and Windows' only evidence that
        loopback has degraded, so it is reported as well as logged.
        """
        self._report_capture_failure(
            f"the WASAPI loopback stream reported PortAudio status {int(status)}",
            CaptureFailure.DEGRADED,
        )
        if self._claim_failure_log(CaptureFailure.DEGRADED):
            log.warning(
                "WASAPI loopback stream reported PortAudio status %d "
                "(paInputUnderflow=%d) — any silence in this recording may be "
                "substituted rather than captured",
                int(status),
                pyaudio.paInputUnderflow,
            )

    def _deliver_block(self, in_data, status) -> None:
        """One callback's worth of work: report the flag, downmix, hand over.

        `frame_count` is not read: it is PortAudio's own count of the frames
        it built `in_data` out of, at the sample size and channel count this
        stream was opened with, so a guard against it proves nothing.
        """
        arrival = time.monotonic()
        if status:
            self._report_stream_status(status)
        with self._lock:
            sink = self._on_block
        if sink is None or not in_data:
            return
        mono = interleaved_buffer_to_mono(in_data, self._channels, SAMPLE_DTYPE)
        self._deliver_to_sink(sink, arrival, mono)

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """Nothing raises out of here, whatever the block or the sink does.

        PortAudio tears the stream down on an exception crossing this boundary
        without reporting it, so a raise is caught, reported once, and the next
        block is still delivered. Nothing here ends the capture.
        """
        try:
            self._deliver_block(in_data, status)
        except Exception as failure:
            self._report_callback_failure(failure)
        return (None, pyaudio.paContinue)

    def start(self, on_block: BlockSink, on_failure: FailureSink | None = None) -> None:
        with self._lock:
            self._on_block = on_block
        self._begin_failure_reports(on_failure)
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
        self._end_failure_reports()
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

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

import numpy as np
import pyaudiowpatch as pyaudio

from app.audio.config import AudioSettings
from app.audio.endpoint_selection import resolve_loopback_device
from app.audio.system_source import BlockSink, SystemAudioSource, SystemAudioUnavailableError
from app.audio.timeline import to_mono
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
        self._status_reported = False
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

        Silence arriving from this callback has two very different causes:
        PortAudio substituting zeros on input underflow, which raises
        `paInputUnderflow` here, or WASAPI genuinely handing over a silent
        mix. They are indistinguishable in the samples themselves and this
        flag is the only thing that separates them — discarding it cost a
        full diagnosis pass during spec 066.
        """
        with self._lock:
            already = self._status_reported
            self._status_reported = True
        if not already:
            log.warning(
                "WASAPI loopback stream reported PortAudio status %d "
                "(paInputUnderflow=%d) — any silence in this recording may be "
                "substituted rather than captured",
                int(status),
                pyaudio.paInputUnderflow,
            )

    def _stream_callback(self, in_data, frame_count, time_info, status):
        arrival = time.monotonic()
        if status:
            self._report_stream_status(status)
        with self._lock:
            sink = self._on_block
        if sink is not None and in_data:
            interleaved = np.frombuffer(in_data, dtype=np.float32)
            if self._channels > 1:
                interleaved = interleaved.reshape(-1, self._channels)
            sink(arrival, to_mono(interleaved))
        return (None, pyaudio.paContinue)

    def start(self, on_block: BlockSink) -> None:
        with self._lock:
            self._on_block = on_block
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

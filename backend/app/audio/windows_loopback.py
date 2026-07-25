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
from app.audio.system_source import BlockSink, SystemAudioSource, SystemAudioUnavailableError
from app.audio.timeline import to_mono

log = logging.getLogger(__name__)


def _find_default_loopback(audio: pyaudio.PyAudio) -> dict:
    """The loopback endpoint for whatever the machine is currently playing through.

    `get_default_wasapi_loopback()` is the direct answer; the generator is the
    fallback for the case where no default render endpoint reports a loopback
    analogue but some other endpoint does.
    """
    try:
        device = audio.get_default_wasapi_loopback()
    except Exception:
        device = None

    if device is None:
        device = next(iter(audio.get_loopback_device_info_generator()), None)

    if device is None:
        raise SystemAudioUnavailableError(
            "No WASAPI loopback device found — the default output endpoint "
            "exposes no loopback analogue"
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
            device = _find_default_loopback(self._audio)
        except Exception:
            self._audio.terminate()
            raise

        self._device_index = int(device["index"])
        self._channels = max(int(device["maxInputChannels"]), 1)
        self._native_sample_rate = int(device["defaultSampleRate"])
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

    def _stream_callback(self, in_data, frame_count, time_info, status):
        arrival = time.monotonic()
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

"""Microphone recorder using sounddevice."""

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import ClassVar

import numpy as np
import sounddevice as sd

from app.audio.analysis import rms_dbfs
from app.audio.base import AudioRecorder, write_wav
from app.audio.config import AudioSettings
from app.audio.session import SESSION_MISMATCH_DETAIL, SessionMismatchError
from app.core.errors import NotReadyError

log = logging.getLogger(__name__)


class NotRecordingError(NotReadyError):
    """A request to end a capture arrived when no capture was running.

    Named rather than a bare ``RuntimeError`` so exactly this state answers
    409: a client reads that 409 as proof its abandoned request was already
    processed. Both ``stop()`` and ``discard()`` raise it.
    """

    code: ClassVar[str] = "not_recording"



class MicrophoneRecorder(AudioRecorder):
    """Records audio from the default microphone input."""

    def __init__(self, settings: AudioSettings):
        self._settings = settings
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._session_id: str | None = None
        self._start_time: float = 0.0
        self._final_duration: float = 0.0
        self._current_level: float = float("-inf")

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        """Called by sounddevice from a separate thread for each audio block."""
        with self._lock:
            if not self._recording:
                return
            self._frames.append(indata.copy())
            self._current_level = rms_dbfs(indata)

    async def start(self, session_id: str | None = None) -> None:
        """Open the device, recording `session_id` as the capture's owner.

        An already-open device keeps the owner it was opened with and this
        call does nothing. `None` starts an unowned capture the recorder
        answers to anyone about, which curl and `smoke_sidecar.py` rely on.
        """
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._current_level = float("-inf")
            self._recording = True
            self._session_id = session_id

        try:
            self._settings.temp_dir.mkdir(parents=True, exist_ok=True)
            self._stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception:
            self.cleanup()
            raise

        self._start_time = time.monotonic()

    async def stop(self, session_id: str | None = None) -> Path:
        """Harvest the capture, refusing a session that does not own it.

        The guard and the state change are indivisible: one `with self._lock`
        acquisition, and no `await` inside any such block in this module.
        """
        with self._lock:
            if not self._recording or self._stream is None:
                raise NotRecordingError("Not recording")
            if session_id is not None and session_id != self._session_id:
                raise SessionMismatchError(SESSION_MISMATCH_DETAIL)
            self._final_duration = time.monotonic() - self._start_time
            self._recording = False
            self._session_id = None

        stream = self._stream
        self._stream = None
        try:
            stream.stop()
        finally:
            stream.close()

        with self._lock:
            frames = self._frames
            self._frames = []

        if not frames:
            raise RuntimeError("No audio data captured")

        filename = f"rec_{uuid.uuid4().hex[:12]}.wav"
        output_path = self._settings.temp_dir / filename

        return await asyncio.to_thread(self._concatenate_and_write, frames, output_path)

    async def discard(self, session_id: str | None = None) -> float:
        """End the capture and drop its frames, writing no file at all.

        The counterpart of `stop()` for a recording nobody will transcribe.
        Returns the duration that was dropped, which is all there is left to
        report about it.
        """
        with self._lock:
            if not self._recording or self._stream is None:
                raise NotRecordingError("Not recording")
            if session_id is not None and session_id != self._session_id:
                raise SessionMismatchError(SESSION_MISMATCH_DETAIL)
            dropped_seconds = time.monotonic() - self._start_time
            self._recording = False
            self._session_id = None
            stream = self._stream
            self._stream = None
            self._frames = []

        try:
            stream.stop()
        finally:
            stream.close()

        return dropped_seconds

    def _concatenate_and_write(self, frames: list[np.ndarray], output_path: Path) -> Path:
        """Concatenate the captured frames and write the WAV.

        Runs off the event loop, so a long recording does not stall every
        other endpoint for the length of the write.
        """
        return write_wav(
            output_path,
            np.concatenate(frames, axis=0),
            self._settings.sample_rate,
            self._settings.channels,
        )

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def session_id(self) -> str | None:
        """Who owns the live capture, or `None` when nothing is being recorded.

        Never a stale name: every exit path clears it inside the same locked
        block that clears `_recording`.
        """
        return self._session_id

    @property
    def duration_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def level_db(self) -> float:
        with self._lock:
            return self._current_level

    @property
    def last_duration_seconds(self) -> float:
        """Duration of the most recently completed recording, or 0.0 if never stopped."""
        return self._final_duration

    def cleanup(self) -> None:
        """Release the audio stream if one is open. Safe to call any time.

        Discards buffered frames without writing a WAV — for app shutdown or
        to roll a failed `start()` back, never as a substitute for `stop()`.
        `stop()` and `close()` get a `try` each so neither is skipped.
        """
        with self._lock:
            stream = self._stream
            self._stream = None
            self._recording = False
            self._session_id = None
            self._frames = []
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                log.warning("Stopping the dictation microphone stream failed", exc_info=True)
            try:
                stream.close()
            except Exception:
                log.warning("Closing the dictation microphone stream failed", exc_info=True)

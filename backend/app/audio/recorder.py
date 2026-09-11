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

    Named rather than raised as a bare ``RuntimeError`` so that exactly this
    state answers 409, instead of a whole coroutine body being wrapped in
    ``except RuntimeError``. That shape is the one [JS-107] was: any unrelated
    ``RuntimeError`` raised later inside the handler's call would be answered
    as "not recording", and the client reads that 409 as *proof* that its
    abandoned request was already processed.

    Both ways a capture can end -- ``stop()`` and ``discard()`` -- raise it,
    so the one state has one class and one status rather than the bare
    ``RuntimeError``/409 split it carried until spec 150.
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

        An already-open device keeps the owner it was opened with and this call
        does nothing. The id names a capture rather than a caller, so writing a
        second caller's id over a live one would hand that capture to a window
        which never opened it and could then stop or discard it — the exact
        confusion the id exists to remove. `POST /audio/start` refuses this case
        with 409 before reaching here, so no caller observes the difference.

        `None` keeps the unowned semantics every caller had before spec 119:
        the recorder answers to anyone, which is what a curl caller and
        `smoke_sidecar.py` still rely on.
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

        The guard and the state change are indivisible because they share one
        `with self._lock` acquisition and because no `await` appears inside
        any such block in this module — both facts are asserted by an AST test
        in `backend/tests/test_audio.py`, so a future edit that moves either
        out turns the suite red rather than reopening the race.
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

        The counterpart of `stop()` for a recording nobody is going to
        transcribe: an abandoned start the client reclaims, or the Settings
        microphone test, neither of which ever wanted the WAV that `stop()`
        leaves in the scratch directory for nothing to delete ([JS-122]).
        Returns the duration that was dropped, which is the only thing left to
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
        """The dictation counterpart of the meeting recorder's off-loop write.

        Smaller -- a dictation clip is seconds, not a 45-minute call -- but the
        same shape, reached from the same `async def`, so a long recording
        stalls every other endpoint for the length of the write.
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
        block that clears `_recording`, which
        `test_a_stopped_recorder_never_names_an_owner` asserts on all four.
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
        """Release the audio stream if one is open. Safe to call any time,
        including when never started. Discards buffered frames without writing
        a WAV — call on app shutdown, or to roll a failed start() back to a
        stopped state, but never as a substitute for stop().

        `stop()` and `close()` get a `try` each, as `meeting_recorder.py` does
        for the same pair: the reference is already dropped and
        `sounddevice._StreamBase` has no finalizer, so a `stop()` that raises —
        the unplugged-headset case — would otherwise skip the `close()` and
        hold that PortAudio stream for the life of the process."""
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

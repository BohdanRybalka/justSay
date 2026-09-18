"""Audio DSP helpers: dBFS level calculation and streaming silence analysis.

numpy-only by constraint: the frozen sidecar's venv holds numpy, soundfile and
sounddevice and nothing else, so anything added here that reaches for another
package breaks the packaged build (ADR 015).
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.audio.config import AudioSettings

log = logging.getLogger(__name__)

_FRAME_SECONDS = 0.030

_MIN_SPEECH_UNITS_FLOOR = 2

_DBFS_FLOOR = 1e-10


class MalformedCaptureBlockError(Exception):
    """A raw capture buffer does not divide into whole interleaved frames.

    Outside the `JustSayError` hierarchy on purpose: nothing routes it to a
    response, and a capture callback reports it through `on_failure` instead.
    """


def required_speech_units(total_unit_count: int, *, cap: int, ratio: float) -> int:
    """How many "speech" units a clip of ``total_unit_count`` units must show.

    The one rule both silence detectors scale by: proportional to clip length
    between a floor and ``cap``. ``cap`` and ``ratio`` are keyword-only.
    """
    return min(cap, max(_MIN_SPEECH_UNITS_FLOOR, math.ceil(total_unit_count * ratio)))


def to_mono(block: np.ndarray) -> np.ndarray:
    """Downmix an interleaved capture block to mono float32.

    What comes back is always writable, whatever the input's channel count or
    flags, so a read-only ``np.frombuffer`` view is safe to pass in.
    """
    array = np.asarray(block, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    mono = np.ascontiguousarray(array, dtype=np.float32)
    return mono if mono.flags.writeable else mono.copy()


def interleaved_buffer_to_mono(buffer: bytes, channels: int, dtype: str) -> np.ndarray:
    """Read a raw interleaved capture buffer and downmix it to writable mono float32.

    Raises ``MalformedCaptureBlockError`` for a buffer that is not whole
    ``channels``-wide frames or a ``channels`` below 1, ``TypeError`` for an
    unparseable ``dtype``.
    """
    if channels < 1:
        raise MalformedCaptureBlockError(
            f"a capture block cannot be read as {channels}-channel frames"
        )
    sample_dtype = np.dtype(dtype)
    sample_count, leftover_bytes = divmod(len(buffer), sample_dtype.itemsize)
    if leftover_bytes or sample_count % channels:
        raise MalformedCaptureBlockError(
            f"a {len(buffer)}-byte capture block is not a whole number of "
            f"{channels}-channel {dtype} frames"
        )
    interleaved = np.frombuffer(buffer, dtype=sample_dtype)
    if channels > 1:
        interleaved = interleaved.reshape(-1, channels)
    return to_mono(interleaved)


def to_dbfs(amplitude: float) -> float:
    """One amplitude in 0..1 expressed in dBFS.

    Every dBFS answer in the codebase comes through here; ``_DBFS_FLOOR``
    avoids ``log10(0)`` on true digital silence.
    """
    return float(20 * np.log10(max(amplitude, _DBFS_FLOOR)))


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS level of ``samples`` in dBFS.

    The one implementation, shared by the silence guard and the Mic Test
    level meter so the two cannot drift on what "level" means.
    """
    rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))
    return to_dbfs(rms)


@dataclass(frozen=True)
class SilenceAnalysis:
    peak_dbfs: float
    speech_frame_count: int
    total_frame_count: int
    is_silent: bool


def _required_speech_frames(total_frame_count: int, settings: AudioSettings) -> int:
    """Length-proportional speech-frame requirement for the energy guard.

    30 ms frames, capped by ``silence_min_speech_frames``. The rule itself is
    `required_speech_units`.
    """
    return required_speech_units(
        total_frame_count,
        cap=settings.silence_min_speech_frames,
        ratio=settings.silence_min_speech_ratio,
    )


def analyze_silence(audio_path: Path, settings: AudioSettings) -> SilenceAnalysis | None:
    """Stream ``audio_path`` in 30 ms frames and decide whether it is silent.

    Silent when the peak misses ``silence_peak_dbfs`` or too few frames clear
    the lower ``silence_frame_dbfs``. ``None`` — never a raise, never a silent
    verdict — means "skip the guard, transcribe anyway" (ADR 015).
    """
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        samplerate = info.samplerate
        frame_len = max(1, int(_FRAME_SECONDS * samplerate))

        peak = 0.0
        speech_frame_count = 0
        total_frame_count = 0
        total_samples_decoded = 0

        for block in sf.blocks(
            str(audio_path), blocksize=frame_len, dtype="float32", always_2d=True
        ):
            mono = to_mono(block)
            total_samples_decoded += mono.size
            if mono.size:
                peak = max(peak, float(np.max(np.abs(mono))))
                if rms_dbfs(mono) >= settings.silence_frame_dbfs:
                    speech_frame_count += 1
            total_frame_count += 1
    except Exception as e:
        log.warning(
            "Silence analysis could not decode %s — failing open, transcription proceeds: %s",
            audio_path, e,
        )
        return None

    decoded_ms = (total_samples_decoded / samplerate) * 1000.0 if samplerate else 0.0
    if decoded_ms < settings.silence_min_analysis_ms:
        log.warning(
            "Silence analysis: only %.1fms of audio decoded from %s (floor=%.1fms) "
            "— failing open, transcription proceeds",
            decoded_ms, audio_path, settings.silence_min_analysis_ms,
        )
        return None

    peak_dbfs = to_dbfs(peak)
    required_speech_frames = _required_speech_frames(total_frame_count, settings)
    is_silent = bool(
        peak_dbfs < settings.silence_peak_dbfs
        or speech_frame_count < required_speech_frames
    )
    return SilenceAnalysis(
        peak_dbfs=peak_dbfs,
        speech_frame_count=speech_frame_count,
        total_frame_count=total_frame_count,
        is_silent=is_silent,
    )

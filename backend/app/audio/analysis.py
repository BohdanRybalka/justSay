"""Audio DSP helpers: dBFS level calculation and streaming silence analysis.

The project's first audio-DSP module — `app.core.audio_validation` is
magic-bytes-only. Deliberately numpy-only: the backend ships as a frozen
PyInstaller sidecar (`backend/build_sidecar.spec`) whose venv contains only
numpy, soundfile and sounddevice. Real VAD libraries (`webrtcvad`,
`silero-vad`, `torch`, `scipy`) are absent and a torch/onnx-based VAD would
break the packaged build — see
docs/adr/015-pipeline-level-silence-guard.md.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.audio.config import AudioSettings

log = logging.getLogger(__name__)

_FRAME_SECONDS = 0.030  # 30 ms

# Floor for the length-proportional speech-frame requirement below. A
# module constant, deliberately NOT an AudioSettings/env-var field: it only
# matters for sub-100ms-of-speech clips, and a reviewer pass on spec 029
# specifically flagged env knobs that carry no useful tuning surface. See
# docs/adr/015-pipeline-level-silence-guard.md.
_MIN_SPEECH_FRAMES_FLOOR = 2


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS level of ``samples`` in dBFS.

    The formula previously lived inline in
    ``MicrophoneRecorder._audio_callback`` (recorder.py) — lifted here
    verbatim so the guard and the Mic Test level meter share one
    implementation and can't drift apart on what "level" means. The
    ``1e-10`` floor avoids ``log10(0)`` on true digital silence.
    """
    rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))
    # float(): np.log10 of a Python float still returns np.float64, which is
    # not `is`-identical to a plain float -- match analyze_silence's own
    # float()/bool() casting convention below.
    return float(20 * np.log10(max(rms, 1e-10)))


@dataclass(frozen=True)
class SilenceAnalysis:
    peak_dbfs: float
    speech_frame_count: int
    total_frame_count: int
    is_silent: bool


def _required_speech_frames(total_frame_count: int, settings: AudioSettings) -> int:
    """Length-proportional speech-frame requirement (spec 029, AC 25-27).

    An absolute floor (the pre-fix behaviour) cannot be satisfied by a short
    clip no matter how loud it is: a 200 ms clip has only ~7 frames total,
    so a flat requirement of 5 unconditionally called it silent. Scaling
    with clip length — capped at ``silence_min_speech_frames`` for the
    long-clip case, floored at the module constant ``_MIN_SPEECH_FRAMES_FLOOR``
    so a handful of frames still means something — fixes that without
    weakening the long-clip requirement at all (the cap keeps that regime
    byte-identical to the old absolute rule).
    """
    return min(
        settings.silence_min_speech_frames,
        max(
            _MIN_SPEECH_FRAMES_FLOOR,
            math.ceil(total_frame_count * settings.silence_min_speech_ratio),
        ),
    )


def analyze_silence(audio_path: Path, settings: AudioSettings) -> SilenceAnalysis | None:
    """Stream ``audio_path`` in 30 ms frames and decide whether it is silent.

    Channels are mean-collapsed to mono per frame. A frame counts as
    "speech" when its RMS level clears ``settings.silence_frame_dbfs``. The
    number of speech frames required is **proportional to clip length**, not
    an absolute count — see ``required_speech_frames`` below — because an
    absolute floor cannot be satisfied by a short clip no matter how loud it
    is (spec 029, Stage 3 review RED-1: a 200 ms clip has only ~7 frames
    total). Silence is declared when the file's peak absolute sample never
    clears ``settings.silence_peak_dbfs`` OR too few frames clear the
    per-frame floor — either condition alone is enough, biasing toward
    letting a hallucination through rather than discarding real speech.
    ``silence_peak_dbfs`` and ``silence_frame_dbfs`` must stay genuinely
    different (peak > frame): at equal values ``rms(frame) <= max|frame| <=
    global peak`` makes the peak check provably unreachable (RED-2) — see
    docs/adr/015-pipeline-level-silence-guard.md.

    Returns ``None`` — never raises, never reports ``is_silent=True`` — in
    two cases:
      - the file cannot be decoded at all (``/pipeline/process-file``
        accepts ``.m4a``/``.webm`` that libsndfile cannot open; treating
        "unreadable" as "silent" would silently break that tab), or
      - fewer than ``settings.silence_min_analysis_ms`` of audio were
        actually decoded (RED-3: a WAV with a valid header but truncated
        data decodes cleanly and yields a handful of samples — not enough
        for an energy heuristic to judge anything. Measured from decoded
        samples, never from ``sf.info(...).duration``: libsndfile clamps
        the declared frame count to the physical file size, so a
        declared-vs-decoded comparison would never catch this).
    Callers MUST treat ``None`` as "skip the guard, transcribe normally"
    (fail open).

    Streams via ``soundfile.blocks()`` rather than ``sf.read()`` — memory
    use stays flat regardless of upload length.
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
            mono = block.mean(axis=1)
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

    # float()/bool() here: numpy scalar types (np.float64/np.bool_) are not
    # `is`-identical to Python's builtin True/False, which would silently
    # break `assert result.is_silent is True`-style test assertions.
    peak_dbfs = float(20 * np.log10(max(peak, 1e-10)))
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

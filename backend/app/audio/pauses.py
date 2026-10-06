"""Where a long recording can be cut without cutting a word: the middle of its longest pause.

A pause is at least ``PAUSE_SECONDS`` in which no 16 ms hop holds speech. TEN VAD's probability
decides speech when the library loads; otherwise a hop louder than ``silence_frame_dbfs`` does.
"""

import numpy as np

from app.audio import vad
from app.audio.analysis import to_dbfs
from app.audio.config import AudioSettings

PAUSE_SECONDS = 0.4


def _hop_levels(samples: np.ndarray, settings: AudioSettings) -> tuple[np.ndarray, float]:
    probabilities = vad.speech_probabilities(samples, settings)
    if probabilities is not None:
        return probabilities, settings.silence_vad_probability
    hop = vad._HOP_SAMPLES
    hops = samples[: samples.size // hop * hop].reshape(-1, hop).astype(np.float64)
    levels = np.array([to_dbfs(rms) for rms in np.sqrt(np.mean(hops**2, axis=1))])
    return levels, settings.silence_frame_dbfs


def longest_pause(samples: np.ndarray, settings: AudioSettings) -> float | None:
    """Seconds into 16 kHz mono ``samples`` of the middle of their longest pause, or ``None``."""
    levels, speech_from = _hop_levels(samples, settings)
    width = round(PAUSE_SECONDS / vad.HOP_SECONDS)
    if len(levels) < width:
        return None
    quiet = np.lib.stride_tricks.sliding_window_view(levels, width).max(axis=1) < speech_from
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.astype(np.int8), [0]))))
    if not edges.size:
        return None
    starts, ends = edges[::2], edges[1::2]
    longest = int(np.argmax(ends - starts))
    middle = (starts[longest] + ends[longest] - 1) / 2
    return (middle + width / 2) * vad.HOP_SECONDS

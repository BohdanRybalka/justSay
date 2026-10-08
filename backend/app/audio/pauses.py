"""Where a long recording can be cut without cutting a word, and which stretches of it hold speech.

A pause is at least ``PAUSE_SECONDS`` in which no 16 ms hop holds speech. TEN VAD's probability
decides speech when the library loads; otherwise a hop louder than ``silence_frame_dbfs`` does.
Speech stretches follow faster-whisper's VAD defaults: 2 s of silence splits, 0.4 s pads.
"""

import numpy as np

from app.audio import vad
from app.audio.analysis import to_dbfs
from app.audio.config import AudioSettings

PAUSE_SECONDS = 0.4
MIN_SILENCE_SECONDS = 2.0
SPEECH_PAD_SECONDS = 0.4


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


def speech_spans(samples: np.ndarray, settings: AudioSettings) -> list[tuple[float, float]]:
    """``(start, end)`` seconds of the speech in 16 kHz mono ``samples``; empty when none is heard.

    Speech closer than ``MIN_SILENCE_SECONDS`` to the next is one span, and every span keeps
    ``SPEECH_PAD_SECONDS`` around it, within the audio.
    """
    levels, speech_from = _hop_levels(samples, settings)
    speech = (levels >= speech_from).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], speech, [0]))))
    runs: list[list[int]] = []
    for start, end in zip(edges[::2], edges[1::2]):
        if runs and (start - runs[-1][1]) * vad.HOP_SECONDS < MIN_SILENCE_SECONDS:
            runs[-1][1] = int(end)
        else:
            runs.append([int(start), int(end)])
    duration = samples.size / vad._HOP_SAMPLES * vad.HOP_SECONDS
    return [
        (
            max(start * vad.HOP_SECONDS - SPEECH_PAD_SECONDS, 0.0),
            min(end * vad.HOP_SECONDS + SPEECH_PAD_SECONDS, duration),
        )
        for start, end in runs
    ]

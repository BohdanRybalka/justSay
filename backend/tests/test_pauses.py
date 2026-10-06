"""Where long audio is cut — the middle of its longest pause, by TEN VAD or by level."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio import pauses, vad
from app.audio.config import AudioSettings

RATE = 16000
SETTINGS = AudioSettings()


def _probabilities(monkeypatch, values: list[tuple[float, float]]) -> np.ndarray:
    """A 10 s stretch whose hop probabilities are ``speech`` except the given ``(start, end)``."""
    hops = int(10.0 / vad.HOP_SECONDS)
    levels = np.full(hops, 0.9, dtype=np.float32)
    for start, end in values:
        levels[int(start / vad.HOP_SECONDS) : int(end / vad.HOP_SECONDS)] = 0.1
    monkeypatch.setattr(vad, "speech_probabilities", lambda samples, settings: levels)
    return np.zeros(hops * vad._HOP_SAMPLES, dtype=np.float32)


def test_the_cut_is_the_middle_of_the_longest_pause(monkeypatch):
    samples = _probabilities(monkeypatch, [(2.0, 2.6), (6.0, 7.0)])

    assert pauses.longest_pause(samples, SETTINGS) == pytest.approx(6.5, abs=0.02)


def test_a_breath_shorter_than_a_pause_is_not_one(monkeypatch):
    samples = _probabilities(monkeypatch, [(4.0, 4.0 + pauses.PAUSE_SECONDS - 0.05)])

    assert pauses.longest_pause(samples, SETTINGS) is None


def test_without_the_vad_a_quiet_stretch_is_the_pause(monkeypatch):
    monkeypatch.setattr(vad, "speech_probabilities", lambda samples, settings: None)
    loud = np.random.default_rng(0).uniform(-0.3, 0.3, 10 * RATE).astype(np.float32)
    quiet = loud.copy()
    quiet[3 * RATE : 4 * RATE] = 0.0

    assert pauses.longest_pause(loud, SETTINGS) is None
    assert pauses.longest_pause(quiet, SETTINGS) == pytest.approx(3.5, abs=0.02)


class _Library:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.destroyed = False

    def create(self, threshold: float) -> object:
        return object()

    def process(self, handle: object, hop: np.ndarray) -> float:
        assert hop.dtype == np.int16 and hop.size == vad._HOP_SAMPLES
        return next(self.values)

    def destroy(self, handle: object) -> None:
        self.destroyed = True


def test_speech_probabilities_rate_each_whole_hop_and_release_the_handle(monkeypatch):
    library = _Library([0.1, 0.8, 0.3])
    monkeypatch.setattr(vad, "_get_library", lambda: library)

    rated = vad.speech_probabilities(np.zeros(3 * vad._HOP_SAMPLES + 10, np.float32), SETTINGS)

    assert rated.tolist() == pytest.approx([0.1, 0.8, 0.3])
    assert library.destroyed


def test_speech_probabilities_abstain_without_the_library(monkeypatch):
    monkeypatch.setattr(vad, "_get_library", lambda: None)

    assert vad.speech_probabilities(np.zeros(4096, np.float32), SETTINGS) is None

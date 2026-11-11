"""A Cloud dictation in pieces — trimmed edges, cut points, piece sizes, second opinions, joins.

A scripted provider answers each piece in turn; the audio is synthetic tone and digital silence,
judged by the energy floor, so trimming, pause search and encoding run for real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import vad
from app.core.errors import ResourceUnavailableError
from app.pipeline import dictation
from app.pipeline.chunking import UnpacedPieces
from app.pipeline.dictation import plan_dictation_pieces, transcribe_dictation
from app.stt.base import STTProvider, TranscriptionResult

RATE = 16000


class _Provider(STTProvider):
    """Answers each call with the next scripted result or raises it; records what it was sent."""

    def __init__(self, answers: list[object]) -> None:
        self.answers = list(answers)
        self.sent: list[dict] = []

    @property
    def model_name(self) -> str:
        return "mock/provider"

    async def transcribe(self, audio_path: Path, language: str = "uk", **kwargs):
        self.sent.append(
            {
                "seconds": sf.info(str(audio_path)).duration,
                "language": language,
                "second_opinion": kwargs.get("second_opinion"),
                "dictation": kwargs.get("dictation"),
            }
        )
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def _energy_decides(monkeypatch):
    monkeypatch.setattr(vad, "speech_probabilities", lambda samples, settings: None)


def _recording(tmp_path: Path, *stretches: tuple[str, float]) -> Path:
    """A WAV of ``("tone", s)`` and ``("quiet", s)`` stretches in order."""
    parts = [
        0.3 * np.sin(2 * np.pi * 220 * np.arange(round(s * RATE)) / RATE)
        if kind == "tone"
        else np.zeros(round(s * RATE))
        for kind, s in stretches
    ]
    path = tmp_path / "dictation.wav"
    sf.write(str(path), np.concatenate(parts).astype(np.float32), RATE)
    return path


async def _run(provider: _Provider, path: Path, language: str = "uk") -> TranscriptionResult:
    return await transcribe_dictation(provider, path, language, None, UnpacedPieces())


def _said(text: str, doubts: int = 0) -> TranscriptionResult:
    return TranscriptionResult(text=text, doubts=doubts)


@pytest.mark.parametrize(
    ("duration", "pieces"),
    [
        (20.0, [(0.0, 20.0)]),
        (28.0, [(0.0, 28.0)]),
        (33.0, [(0.0, 23.0), (23.0, 33.0)]),
        (60.0, [(0.0, 28.0), (28.0, 50.0), (50.0, 60.0)]),
    ],
)
def test_without_a_pause_a_dictation_is_cut_hard_at_the_limit_leaving_ten_seconds(
    duration, pieces
):
    assert plan_dictation_pieces(duration, lambda start, end: None) == pieces


def test_a_cut_is_searched_from_ten_seconds_in_up_to_the_limit():
    asked: list[tuple[float, float]] = []

    def find_pause(start: float, end: float) -> float:
        asked.append((start, end))
        return start + 2.0

    pieces = plan_dictation_pieces(45.0, find_pause)

    assert asked == [(10.0, 28.0), (22.0, 35.0)]
    assert pieces == [(0.0, 12.0), (12.0, 24.0), (24.0, 45.0)]


async def test_silence_before_and_after_the_speech_is_not_sent(tmp_path):
    provider = _Provider([_said("привіт")])

    await _run(provider, _recording(tmp_path, ("quiet", 3.0), ("tone", 5.0), ("quiet", 4.0)))

    assert [round(sent["seconds"], 1) for sent in provider.sent] == [5.8]


async def test_a_long_dictation_is_cut_in_its_longest_pause_with_nothing_sent_twice(tmp_path):
    path = _recording(
        tmp_path,
        ("quiet", 2.0),
        ("tone", 11.0),
        ("quiet", 0.5),
        ("tone", 5.0),
        ("quiet", 1.2),
        ("tone", 12.0),
        ("quiet", 3.0),
    )
    provider = _Provider([_said("перша частина"), _said("друга частина")])

    result = await _run(provider, path)

    seconds = [sent["seconds"] for sent in provider.sent]
    assert len(seconds) == 2
    assert seconds[0] == pytest.approx(17.5, abs=0.15)
    assert sum(seconds) == pytest.approx(30.5, abs=0.05)
    assert result.text == "перша частина друга частина"
    assert all(sent["dictation"] and sent["language"] == "uk" for sent in provider.sent)


async def test_a_piece_heard_as_non_speech_adds_nothing_and_is_not_asked_again(tmp_path):
    provider = _Provider([_said("", doubts=1), _said("друга частина")])

    result = await _run(provider, _recording(tmp_path, ("tone", 40.0)))

    assert result.text == "друга частина"
    assert result.no_speech_prob is None
    assert [sent["second_opinion"] for sent in provider.sent] == [False, False]


async def test_a_dictation_with_no_piece_left_reads_as_no_speech(tmp_path):
    provider = _Provider([_said("", doubts=1)])

    result = await _run(provider, _recording(tmp_path, ("tone", 5.0)))

    assert result.text == ""
    assert result.no_speech_prob == 1.0


@pytest.mark.parametrize(
    ("second", "kept"),
    [
        (_said("вперед"), "вперед"),
        (_said("в керівництві", doubts=1), "в керівництві. про это"),
        (_said("", doubts=0), "в керівництві. про это"),
        (ResourceUnavailableError("Groq is overloaded right now. Try again later."),
         "в керівництві. про это"),
    ],
)
async def test_a_doubtful_piece_is_asked_once_more_and_the_surer_answer_kept(
    tmp_path, second, kept
):
    provider = _Provider([_said("в керівництві. про это", doubts=1), second])

    result = await _run(provider, _recording(tmp_path, ("tone", 5.0)))

    assert result.text == kept
    assert [sent["second_opinion"] for sent in provider.sent] == [False, True]


async def test_no_scratch_file_is_left_beside_the_recording(tmp_path):
    path = _recording(tmp_path, ("tone", 40.0))

    await _run(_Provider([_said("а"), _said("б")]), path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["dictation.wav"]


async def test_audio_that_cannot_be_cut_here_goes_whole(tmp_path, monkeypatch):
    path = _recording(tmp_path, ("tone", 40.0))

    def undecodable(source, target, rate):
        raise dictation.UndecodableAudioError("no decoder")

    monkeypatch.setattr(dictation, "decode_to_mono_wav", undecodable)
    provider = _Provider([_said("усе разом")])

    result = await _run(provider, path)

    assert result.text == "усе разом"
    assert [round(sent["seconds"]) for sent in provider.sent] == [40]

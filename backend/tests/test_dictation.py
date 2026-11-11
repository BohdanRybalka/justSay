"""A Cloud dictation in pieces — trimmed edges, cut points, piece sizes, second opinions, joins.

A scripted provider answers each piece in turn; the audio is synthetic tone and digital silence,
judged by the energy floor unless a test hands TEN VAD's part a script, so trimming, pause search
and encoding run for real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import vad
from app.core.errors import ResourceUnavailableError
from app.pipeline import chunking
from app.pipeline.chunking import UnpacedPieces
from app.pipeline.dictation import plan_dictation_pieces, transcribe_dictation
from app.stt.base import STTProvider, TranscriptionResult

RATE = 16000
THRESHOLD = 0.6


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
                "path": audio_path,
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


def _tone(seconds: float, dbfs: float = -13.5) -> np.ndarray:
    amplitude = np.sqrt(2) * 10 ** (dbfs / 20)
    return amplitude * np.sin(2 * np.pi * 220 * np.arange(round(seconds * RATE)) / RATE)


def _quiet(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * RATE))


def _recording(tmp_path: Path, *parts: np.ndarray, rate: int = RATE) -> Path:
    path = tmp_path / "dictation.wav"
    sf.write(str(path), np.repeat(np.concatenate(parts), rate // RATE).astype(np.float32), rate)
    return path


async def _run(provider: _Provider, path: Path, language: str = "uk") -> TranscriptionResult:
    return await transcribe_dictation(
        provider,
        path,
        language,
        sf.info(str(path)).duration,
        UnpacedPieces(),
        no_speech_threshold=THRESHOLD,
    )


def _said(text: str, doubts: int = 0, no_speech_prob: float | None = None) -> TranscriptionResult:
    return TranscriptionResult(text=text, doubts=doubts, no_speech_prob=no_speech_prob)


def _seconds(provider: _Provider) -> list[float]:
    return [round(sent["seconds"], 1) for sent in provider.sent]


@pytest.mark.parametrize(
    ("duration", "pieces"),
    [
        (20.0, [(0.0, 20.0)]),
        (28.01, [(0.0, 28.01)]),
        (29.0, [(0.0, 29.0)]),
        (30.0, [(0.0, 30.0)]),
        (30.5, [(0.0, 20.5), (20.5, 30.5)]),
        (33.0, [(0.0, 23.0), (23.0, 33.0)]),
        (60.0, [(0.0, 28.0), (28.0, 50.0), (50.0, 60.0)]),
    ],
)
def test_a_dictation_goes_whole_up_to_the_window_and_is_cut_hard_past_it_without_a_pause(
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


@pytest.mark.parametrize("sound", [2.0, 24.0])
async def test_a_dictation_up_to_the_window_reaches_the_provider_as_recorded(
    tmp_path, monkeypatch, sound
):
    decoded: list[Path] = []
    monkeypatch.setattr(
        chunking, "decode_to_mono_wav", lambda source, target, rate: decoded.append(source)
    )
    path = _recording(tmp_path, _quiet(3.0), _tone(sound), _quiet(3.0), rate=48000)
    provider = _Provider([_said("усе")])

    await _run(provider, path)

    assert [sent["path"] for sent in provider.sent] == [path]
    assert _seconds(provider) == [sound + 6.0]
    assert decoded == []


async def test_a_35_second_dictation_is_cut_in_its_pause(tmp_path):
    provider = _Provider([_said("перша"), _said("друга")])

    await _run(provider, _recording(tmp_path, _tone(15.0), _quiet(1.0), _tone(19.0)))

    assert _seconds(provider) == [15.5, 19.5]


async def test_a_long_dictation_is_cut_in_its_longest_pause_with_nothing_sent_twice(tmp_path):
    path = _recording(
        tmp_path,
        _quiet(2.0),
        _tone(11.0),
        _quiet(0.5),
        _tone(5.0),
        _quiet(1.2),
        _tone(12.0),
        _quiet(3.0),
    )
    provider = _Provider([_said("перша частина"), _said("друга частина")])

    result = await _run(provider, path)

    seconds = [sent["seconds"] for sent in provider.sent]
    assert seconds[0] == pytest.approx(18.1, abs=0.15)
    assert sum(seconds) == pytest.approx(31.7, abs=0.05)
    assert result.text == "перша частина друга частина"
    assert all(sent["dictation"] and sent["language"] == "uk" for sent in provider.sent)


async def test_silence_before_and_after_a_long_dictation_is_not_sent(tmp_path):
    path = _recording(
        tmp_path, _quiet(4.0), _tone(15.0), _quiet(1.0), _tone(15.0), _quiet(4.0)
    )
    provider = _Provider([_said("перша"), _said("друга")])

    await _run(provider, path)

    assert _seconds(provider) == [16.5, 16.5]


@pytest.mark.parametrize(
    "parts",
    [
        (_quiet(4.0), _tone(1.0, -55.0), _quiet(0.8), _tone(26.0, -20.0), _quiet(4.0)),
        (_quiet(4.0), _tone(26.0, -20.0), _quiet(0.8), _tone(1.0, -55.0), _quiet(4.0)),
    ],
)
async def test_a_quiet_first_or_last_word_is_kept(tmp_path, parts):
    provider = _Provider([_said("тихо")])

    await _run(provider, _recording(tmp_path, *parts))

    assert _seconds(provider) == [29.8]


async def test_what_the_voice_detector_hears_is_kept_however_quiet(tmp_path, monkeypatch):
    def voice_wherever_anything_sounds(samples, settings):
        hops = samples[: samples.size // 256 * 256].reshape(-1, 256)
        return np.where(np.abs(hops).max(axis=1) > 0, 0.9, 0.0).astype(np.float32)

    monkeypatch.setattr(vad, "speech_probabilities", voice_wherever_anything_sounds)
    path = _recording(
        tmp_path, _quiet(3.0), _tone(0.5, -80.0), _quiet(1.0), _tone(25.0), _quiet(3.0)
    )
    provider = _Provider([_said("шепіт")])

    await _run(provider, path)

    assert _seconds(provider) == [28.5]


@pytest.mark.parametrize(
    ("second", "kept"),
    [
        (_said("Дякую за перегляд.", no_speech_prob=0.1), "Дякую за перегляд."),
        (_said("Дякую за перегляд.", no_speech_prob=0.9), ""),
        (_said(""), ""),
    ],
)
async def test_confident_speech_that_sounds_like_no_speech_is_dropped_only_when_both_agree(
    tmp_path, second, kept
):
    provider = _Provider([_said("Дякую за перегляд.", no_speech_prob=0.8), second])

    result = await _run(provider, _recording(tmp_path, _tone(3.0)))

    assert result.text == kept
    assert (result.no_speech_prob is None) == bool(kept)
    assert [sent["second_opinion"] for sent in provider.sent] == [False, True]


async def test_speech_that_sounds_like_no_speech_is_kept_when_asking_again_fails(tmp_path):
    provider = _Provider(
        [_said("вперед", no_speech_prob=0.8), ResourceUnavailableError("Groq did not answer.")]
    )

    result = await _run(provider, _recording(tmp_path, _tone(3.0)))

    assert result.text == "вперед"


async def test_a_dictation_over_the_window_only_before_the_trim_goes_whole(tmp_path):
    provider = _Provider([_said("усе")])

    await _run(provider, _recording(tmp_path, _quiet(3.0), _tone(27.0), _quiet(3.0)))

    assert _seconds(provider) == [29.0]


async def test_a_long_recording_with_no_sound_at_all_is_sent_untrimmed(tmp_path):
    provider = _Provider([_said(""), _said("")])

    result = await _run(provider, _recording(tmp_path, _quiet(35.0)))

    assert sum(_seconds(provider)) == pytest.approx(35.0)
    assert result.no_speech_prob == 1.0


async def test_a_piece_without_text_adds_nothing_and_is_not_asked_again(tmp_path):
    provider = _Provider([_said(""), _said("друга частина")])

    result = await _run(provider, _recording(tmp_path, _tone(40.0)))

    assert result.text == "друга частина"
    assert result.no_speech_prob is None
    assert [sent["second_opinion"] for sent in provider.sent] == [False, False]


@pytest.mark.parametrize(
    ("answers", "no_speech_prob"),
    [
        (
            [
                _said("Дякую.", no_speech_prob=0.9),
                _said("Дякую.", no_speech_prob=0.95),
                _said("Дякую.", no_speech_prob=0.7),
                _said(""),
            ],
            0.95,
        ),
        ([_said(""), _said("")], 1.0),
    ],
)
async def test_a_dictation_with_no_piece_heard_reads_as_no_speech(
    tmp_path, answers, no_speech_prob
):
    result = await _run(_Provider(answers), _recording(tmp_path, _tone(40.0)))

    assert result.text == ""
    assert result.no_speech_prob == no_speech_prob


@pytest.mark.parametrize(
    ("second", "kept"),
    [
        (_said("вперед"), "вперед"),
        (_said("в керівництві", doubts=1), "Дякую за перегляд. про это"),
        (_said(""), ""),
        (_said("Дякую за перегляд.", no_speech_prob=0.95), ""),
        (
            ResourceUnavailableError("Groq is overloaded right now. Try again later."),
            "Дякую за перегляд. про это",
        ),
    ],
)
async def test_a_doubted_piece_is_asked_once_more_and_the_surer_answer_kept(
    tmp_path, second, kept
):
    provider = _Provider([_said("Дякую за перегляд. про это", doubts=1), second])

    result = await _run(provider, _recording(tmp_path, _tone(5.0)))

    assert result.text == kept
    assert [sent["second_opinion"] for sent in provider.sent] == [False, True]


@pytest.mark.parametrize(
    ("language", "detected", "joined"),
    [
        ("zh", None, "你好世界"),
        ("ja", None, "你好世界"),
        ("auto", "zh", "你好世界"),
        ("uk", None, "你好 世界"),
    ],
)
async def test_pieces_join_without_a_space_in_a_language_written_without_spaces(
    tmp_path, language, detected, joined
):
    answers = [
        TranscriptionResult(text="你好", detected_language=detected),
        TranscriptionResult(text="世界", detected_language=detected),
    ]

    result = await _run(_Provider(answers), _recording(tmp_path, _tone(40.0)), language)

    assert result.text == joined


async def test_no_scratch_file_is_left_beside_the_recording(tmp_path):
    path = _recording(tmp_path, _tone(40.0))

    await _run(_Provider([_said("а"), _said("б")]), path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["dictation.wav"]


async def test_audio_that_cannot_be_cut_here_goes_whole(tmp_path, monkeypatch):
    path = _recording(tmp_path, _tone(40.0), rate=48000)

    def undecodable(source, target, rate):
        raise chunking.UndecodableAudioError("no decoder")

    monkeypatch.setattr(chunking, "decode_to_mono_wav", undecodable)
    provider = _Provider([_said("усе разом")])

    result = await _run(provider, path)

    assert result.text == "усе разом"
    assert [round(sent["seconds"]) for sent in provider.sent] == [40]

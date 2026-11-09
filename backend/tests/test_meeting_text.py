"""A two-channel meeting becomes labelled turns in spoken order, with the echo cut out.

A scripted provider answers each turn in the order it is sent: the left channel's turns by
time, then the right channel's. The audio is synthetic noise on one channel at a time, so the
cutting into turns and pieces runs for real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipeline.meeting_text import (
    has_two_tracks,
    spoken_word_count,
    transcribe_meeting,
    without_repeats,
)
from app.stt.base import STTProvider, TranscriptionResult
from tests.conftest import write_meeting_wav

THRESHOLD = 0.6

pytestmark = pytest.mark.usefixtures("energy_decides_speech")


class _Provider(STTProvider):
    def __init__(self, answers: list[TranscriptionResult], *, longest: float | None = None) -> None:
        self.answers = list(answers)
        if longest is not None:
            self.longest_piece_seconds = longest
        self.sent = 0

    @property
    def model_name(self) -> str:
        return "mock/provider"

    async def transcribe(self, audio_path: Path, language: str = "uk", **kwargs):
        self.sent += 1
        return self.answers.pop(0)


class _Pacer:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def before_piece(self, index: int, total: int) -> None:
        self.events.append(("before", index, total))

    async def before_send(self) -> None:
        return None

    async def pause(self, seconds: float) -> None:
        self.events.append(("pause", seconds))

    def piece_done(self, done: int, total: int) -> None:
        self.events.append(("done", done, total))


def _said(*texts: str) -> list[TranscriptionResult]:
    return [TranscriptionResult(text=text) for text in texts]


async def _transcribe(provider: _Provider, path: Path, pacer: _Pacer | None = None):
    return await transcribe_meeting(
        provider,
        path,
        language="uk",
        no_speech_threshold=THRESHOLD,
        holds_no_speech=lambda piece: False,
        pacer=pacer or _Pacer(),
    )


@pytest.fixture
def meeting(tmp_path):
    def make(seconds: float, **ranges) -> Path:
        return write_meeting_wav(tmp_path / "meeting_0123456789ab.wav", seconds, **ranges)

    return make


@pytest.mark.asyncio
async def test_each_turn_is_labelled_and_follows_the_order_it_was_spoken(meeting, tmp_path):
    path = meeting(20.0, you=((0.0, 3.0), (12.0, 15.0)), others=((5.0, 10.0),))
    provider = _Provider(_said("привіт колего", "бувайте", "добрий день"))

    result = await _transcribe(provider, path)

    assert result.text == "You: привіт колего\nOthers: добрий день\nYou: бувайте"
    assert result.no_speech_prob is None
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.asyncio
async def test_turns_of_one_side_with_nothing_between_share_a_line(meeting):
    path = meeting(30.0, you=((0.0, 3.0), (6.0, 9.0)), others=((20.0, 23.0),))
    provider = _Provider(_said("перше", "друге", "відповідь"))

    result = await _transcribe(provider, path)

    assert result.text == "You: перше друге\nOthers: відповідь"


@pytest.mark.asyncio
async def test_one_side_speaking_alone_leaves_the_text_without_labels(meeting):
    path = meeting(20.0, you=((0.0, 3.0), (12.0, 15.0)))
    provider = _Provider(_said("перша думка", "друга думка"))

    result = await _transcribe(provider, path)

    assert result.text == "перша думка друга думка"


@pytest.mark.asyncio
async def test_a_microphone_turn_that_only_repeats_the_others_is_cut(meeting):
    path = meeting(20.0, you=((0.0, 3.0), (6.0, 10.0)), others=((6.0, 10.0),))
    provider = _Provider(
        _said(
            "привіт",
            "завтра о десятій буде нарада в офісі",
            "завтра о десятій буде нарада в офісі",
        )
    )

    result = await _transcribe(provider, path)

    assert result.text == "You: привіт\nOthers: завтра о десятій буде нарада в офісі"


@pytest.mark.asyncio
async def test_a_repeat_from_seconds_away_is_not_an_echo(meeting):
    path = meeting(60.0, you=((22.0, 26.0),), others=((6.0, 10.0),))
    provider = _Provider(
        _said("завтра о десятій буде нарада в офісі", "завтра о десятій буде нарада в офісі")
    )

    result = await _transcribe(provider, path)

    assert result.text == (
        "Others: завтра о десятій буде нарада в офісі\n"
        "You: завтра о десятій буде нарада в офісі"
    )


@pytest.mark.asyncio
async def test_a_repeat_starting_a_moment_after_the_others_stopped_is_still_an_echo(meeting):
    path = meeting(20.0, you=((10.0, 13.0),), others=((3.0, 7.5),))
    provider = _Provider(
        _said("завтра о десятій буде нарада в офісі", "завтра о десятій буде нарада в офісі")
    )

    result = await _transcribe(provider, path)

    assert result.text == "завтра о десятій буде нарада в офісі"


@pytest.mark.asyncio
async def test_speech_running_across_a_minute_mark_is_one_turn(meeting):
    path = meeting(75.0, you=((55.0, 65.0),))
    provider = _Provider(_said("довга репліка"))

    result = await _transcribe(provider, path)

    assert provider.sent == 1
    assert result.text == "довга репліка"


@pytest.mark.asyncio
async def test_a_meeting_without_speech_sends_nothing_and_is_reported_silent(meeting):
    path = meeting(10.0)
    provider = _Provider([])

    result = await _transcribe(provider, path)

    assert provider.sent == 0
    assert result.text == ""
    assert result.no_speech_prob == 1.0


@pytest.mark.asyncio
async def test_progress_counts_every_turn_as_a_piece_of_one_file(meeting):
    path = meeting(20.0, you=((0.0, 3.0), (12.0, 15.0)), others=((5.0, 10.0),))
    pacer = _Pacer()

    await _transcribe(_Provider(_said("раз", "два", "три")), path, pacer)

    assert pacer.events == [
        ("before", 0, 3),
        ("done", 1, 3),
        ("before", 1, 3),
        ("done", 2, 3),
        ("before", 2, 3),
        ("done", 3, 3),
    ]


@pytest.mark.asyncio
async def test_the_pieces_of_a_long_turn_all_count_before_the_turns_that_follow(meeting):
    path = meeting(40.0, you=((0.0, 25.0),), others=((30.0, 33.0),))
    pacer = _Pacer()
    provider = _Provider(_said("раз два", "три чотири", "п'ять"), longest=20.0)

    result = await _transcribe(provider, path, pacer)

    assert result.text == "You: раз два три чотири\nOthers: п'ять"
    assert pacer.events == [
        ("before", 0, 3),
        ("done", 1, 3),
        ("before", 1, 3),
        ("done", 2, 3),
        ("before", 2, 3),
        ("done", 3, 3),
    ]


@pytest.mark.asyncio
async def test_tokens_add_up_and_the_language_most_turns_were_heard_in_is_reported(meeting):
    path = meeting(20.0, you=((0.0, 3.0), (12.0, 15.0)), others=((5.0, 10.0),))
    provider = _Provider(
        [
            TranscriptionResult(text="раз", tokens_used=1, detected_language="uk"),
            TranscriptionResult(text="два", tokens_used=2, detected_language="en"),
            TranscriptionResult(text="три", tokens_used=3, detected_language="en"),
        ]
    )

    result = await transcribe_meeting(
        provider,
        path,
        language="auto",
        no_speech_threshold=THRESHOLD,
        holds_no_speech=lambda piece: False,
        pacer=_Pacer(),
    )

    assert result.tokens_used == 6
    assert result.detected_language == "en"


@pytest.mark.parametrize(
    ("text", "heard", "expected"),
    [
        (
            "раз два три чотири п'ять шість сімм",
            "раз два три чотири п'ять шість сім",
            "",
        ),
        (
            "ну я думаю що завтра о десятій буде нарада в офісі",
            "завтра о десятій буде нарада",
            "ну я думаю що в офісі",
        ),
        (
            "так завтра о десятій добре",
            "завтра о десятій буде нарада",
            "так завтра о десятій добре",
        ),
        ("зовсім інші слова", "завтра о десятій буде нарада", "зовсім інші слова"),
        ("", "завтра о десятій", ""),
        (
            "раз два три чотири п'ять шість сім вісім",
            "раз два три чотири п'ять шість",
            "",
        ),
        (
            "раз два три чотири п'ять шість сім вісім",
            "раз два три чотири п'ять",
            "шість сім вісім",
        ),
        ("раз два три чотири", "раз два п'ять три чотири", ""),
        (
            "ну я думаю що завтра о десятій буде в офісі",
            "завтра о десятій буде нарада",
            "ну я думаю що в офісі",
        ),
    ],
    ids=[
        "a copy with one slip",
        "a long run among own words",
        "a short run",
        "unrelated",
        "empty",
        "a copy at the threshold",
        "a run of five among own words",
        "two runs of two",
        "a run of exactly four",
    ],
)
def test_the_words_the_others_said_in_the_same_seconds_are_taken_out(text, heard, expected):
    assert without_repeats(text, heard) == expected


def test_the_labels_are_not_counted_as_words():
    assert spoken_word_count("You: раз два\nOthers: три чотири п'ять") == 5
    assert spoken_word_count("раз два три") == 3
    assert spoken_word_count("") == 0


def test_only_a_two_channel_file_at_the_piece_rate_is_a_meeting_with_sides(tmp_path):
    stereo = write_meeting_wav(tmp_path / "stereo.wav", 1.0)
    mono = tmp_path / "mono.wav"
    sf.write(str(mono), np.zeros(16000, dtype=np.float32), 16000)
    wide = tmp_path / "wide.wav"
    sf.write(str(wide), np.zeros((48000, 2), dtype=np.float32), 48000)
    not_audio = tmp_path / "notes.wav"
    not_audio.write_bytes(b"not audio")

    assert has_two_tracks(stereo) is True
    assert has_two_tracks(mono) is False
    assert has_two_tracks(wide) is False
    assert has_two_tracks(not_audio) is False
    assert has_two_tracks(tmp_path / "missing.wav") is False

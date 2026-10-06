"""Long audio in pieces — spans, seams, piece sizes, language, silence, pauses, scratch files.

A scripted provider answers each piece in turn, so the seams carry known words; the audio is
synthetic and real, so cutting and encoding run for real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import vad
from app.core.audio_formats import ffmpeg_selftest
from app.core.errors import ResourceUnavailableError
from app.pipeline import chunking
from app.pipeline.chunking import join_at_seam, transcribe_in_pieces
from app.stt.base import STTProvider, TranscriptionResult
from tests.conftest import write_aac_m4a

GROQ_FILE_LIMIT = 25 * 1000 * 1000
THRESHOLD = 0.6


class _Provider(STTProvider):
    """Answers each call with the next scripted result or raises it; records what it was sent."""

    def __init__(self, answers: list[object], *, is_local: bool = False) -> None:
        self.answers = list(answers)
        self.is_local = is_local
        self.sent: list[dict] = []

    @property
    def model_name(self) -> str:
        return "mock/provider"

    async def transcribe(self, audio_path: Path, language: str = "uk", **kwargs):
        info = sf.info(str(audio_path)) if audio_path.suffix == ".flac" else None
        self.sent.append(
            {"path": audio_path, "language": language, "seconds": info and info.duration}
        )
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class _Pacer:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def before_piece(self, index: int, total: int) -> None:
        self.events.append(("before", index, total))

    async def pause(self, seconds: float) -> None:
        self.events.append(("pause", seconds))

    def piece_done(self, done: int, total: int) -> None:
        self.events.append(("done", done, total))


def _audio(tmp_path: Path, seconds: float, rate: int = 44100, channels: int = 2) -> Path:
    path = tmp_path / "job_x.wav"
    noise = np.random.default_rng(0).uniform(-0.1, 0.1, (int(seconds * rate), channels))
    sf.write(str(path), noise.astype(np.float32), rate)
    return path


async def _run(
    provider: _Provider,
    path: Path,
    pacer: _Pacer | None = None,
    language: str = "auto",
    silent_pieces: frozenset[int] = frozenset(),
):
    def holds_no_speech(piece: Path) -> bool:
        return int(piece.stem.rsplit("piece", 1)[1]) in silent_pieces

    return await transcribe_in_pieces(
        provider,
        path,
        language=language,
        duration=None,
        no_speech_threshold=THRESHOLD,
        holds_no_speech=holds_no_speech,
        pacer=pacer or _Pacer(),
    )


@pytest.fixture
def short_pieces(monkeypatch):
    """Pieces of 30 s (15 s local) with no pause to cut in, answered by one-line transcripts."""
    monkeypatch.setattr(chunking, "CLOUD_PIECE_SECONDS", 30.0)
    monkeypatch.setattr(chunking, "LOCAL_PIECE_SECONDS", 15.0)
    monkeypatch.setattr(chunking, "OVERLAP_SECONDS", 5.0)
    monkeypatch.setattr(chunking, "MIN_CHARS_PER_MINUTE", 0)
    monkeypatch.setattr(chunking, "_pause_in", lambda mono, start, end: None)


def _pause_before(gap: float):
    asked: list[tuple[float, float]] = []

    def find_pause(start: float, end: float) -> float:
        asked.append((start, end))
        return end - gap

    return find_pause, asked


@pytest.mark.parametrize(("duration", "count"), [(20.0, 1), (600.0, 1), (601.0, 2), (3600.0, 7)])
def test_each_piece_ends_in_the_pause_found_in_its_last_tenth(duration, count):
    find_pause, asked = _pause_before(20.0)

    pieces = chunking.plan_pieces(duration, 600.0, find_pause)

    assert len(pieces) == count
    assert pieces[0].start == 0.0
    assert pieces[-1].end == pytest.approx(duration)
    assert all(p.end - p.start <= 600.0 for p in pieces)
    assert all(not p.overlaps_previous for p in pieces)
    for before, after in zip(pieces, pieces[1:]):
        assert after.start == before.end
    assert all(end - start == pytest.approx(60.0) for start, end in asked)
    limits = [min(p.start + 600.0, duration - 60.0) for p in pieces[:-1]]
    assert [round(end) for _, end in asked] == [round(limit) for limit in limits]
    assert pieces[-1].end - pieces[-1].start >= 60.0 or count == 1


def test_without_a_pause_pieces_overlap_and_are_marked_for_the_seam():
    pieces = chunking.plan_pieces(1300.0, 600.0, lambda start, end: None)

    assert [(p.start, p.end, p.overlaps_previous) for p in pieces] == [
        (0.0, 600.0, False),
        (590.0, 1190.0, True),
        (1180.0, 1300.0, True),
    ]


@pytest.mark.parametrize(
    ("text", "seconds", "broken"),
    [
        (" ".join(f"w{i:04d}" for i in range(120)), 600.0, False),
        (" ".join(f"w{i:04d}" for i in range(119)) + " abcd", 600.0, True),
        ("".join(chr(0x4E00 + i) for i in range(300)), 300.0, False),
        ("".join(chr(0x4E00 + i) for i in range(20)) * 30, 300.0, True),
        ("Тихо, тихо, тихо, тихо, тихо. " + " ".join(f"w{i}" for i in range(200)), 600, False),
        (" ".join(["we go to the cave with the scorpions now"] * 40), 600.0, True),
    ],
)
def test_a_transcript_too_sparse_or_looping_looks_broken(text, seconds, broken):
    assert chunking.looks_broken(text, seconds) is broken


@pytest.mark.parametrize(
    ("left", "right", "joined"),
    [
        (
            "We will ship the new build on Friday morn",
            "build on Friday morning and then rest.",
            "We will ship the new build on Friday morning and then rest.",
        ),
        (
            "Привіт, це тест. Ми говоримо про нарізку",
            "говоримо про нарізку довгих записів",
            "Привіт, це тест. Ми говоримо про нарізку довгих записів",
        ),
        (
            "so the plan is that we take the long road and then",
            "the plan is that, well, we take the long road and then rest",
            "so the plan is that we take the long road and then rest",
        ),
        (
            "and so we decided to go.",
            "We decided to go, and then",
            "and so we decided to go, and then",
        ),
        ("Nothing in common", "with the next one", "Nothing in common with the next one"),
        (
            "we talked about the plan",
            "and the budget was fine",
            "we talked about the plan and the budget was fine",
        ),
        ("Line one.\nLine two is", "Line two is and three", "Line one.\nLine two is and three"),
        ("so yes", "oh well um so yes then more", "so yes oh well um so yes then more"),
        ("we stop — … —", "— … — and go on", "we stop — … — — … — and go on"),
    ],
)
def test_words_heard_in_the_overlap_are_kept_once(left, right, joined):
    assert join_at_seam(left, right) == joined


def test_an_overlap_without_shared_speech_is_joined_whole_not_cut_at_a_chance_match():
    left = (
        "Сідай. Заводься. І погнали. Так, мені отуди. І ось там щось є, наче якісь дерева і там "
        "щось теж типу сміттєзвалища якогось. Хм."
    )
    right = (
        "Так, тихенько-тихенько. Ні, то скеля. А мені здалося, що там якийсь будинок на скелі "
        "побудований. І тут нічого. Мені здалось... О, чекайте, а там що?"
    )

    assert join_at_seam(left, right) == f"{left} {right}"


@pytest.mark.parametrize("distance", [25, 60])
def test_a_phrase_repeated_far_from_the_seam_is_not_taken_for_the_overlap(distance):
    before = " ".join(f"left{i}" for i in range(distance))
    after = " ".join(f"right{i}" for i in range(distance))
    left = f"{before} I don't know {before}"
    right = f"{after} I don't know {after}"

    assert join_at_seam(left, right) == f"{left} {right}"


def test_of_two_equal_runs_the_one_at_the_seam_wins():
    middle = " ".join(f"word{i}" for i in range(30))
    left = f"so I think that is great {middle} and I think that is fine"
    right = "I think that was fine and more"

    joined = join_at_seam(left, right)

    assert joined == f"so I think that is great {middle} and I think that was fine and more"


async def test_a_long_recording_goes_as_flac_pieces_joined_at_the_seams(tmp_path, short_pieces):
    path = _audio(tmp_path, 70.0)
    provider = _Provider([
        TranscriptionResult("one two three four five six", detected_language="uk"),
        TranscriptionResult("four five six seven eight nine"),
        TranscriptionResult("seven eight nine ten"),
    ])
    pacer = _Pacer()

    result = await _run(provider, path, pacer)

    assert result.text == "one two three four five six seven eight nine ten"
    assert result.detected_language == "uk"
    assert [s["language"] for s in provider.sent] == ["auto", "uk", "uk"]
    assert all(s["path"].suffix == ".flac" for s in provider.sent)
    assert [round(s["seconds"]) for s in provider.sent] == [30, 30, 20]
    assert pacer.events == [
        ("before", 0, 3), ("done", 1, 3),
        ("before", 1, 3), ("done", 2, 3),
        ("before", 2, 3), ("done", 3, 3),
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["job_x.wav"]


async def test_a_local_engine_gets_shorter_pieces(tmp_path, short_pieces):
    path = _audio(tmp_path, 40.0)
    provider = _Provider([TranscriptionResult("a")] * 4, is_local=True)

    await _run(provider, path)

    assert len(provider.sent) == 4


def _speech_with_pauses(tmp_path: Path, seconds: float, pauses: list[float]) -> Path:
    rate = 16000
    noise = np.random.default_rng(0).uniform(-0.3, 0.3, int(seconds * rate)).astype(np.float32)
    for at in pauses:
        noise[int(at * rate) : int((at + 0.6) * rate)] = 0.0
    path = tmp_path / "job_x.wav"
    sf.write(str(path), noise, rate)
    return path


async def test_pieces_are_cut_in_the_pauses_of_the_recording_and_joined_whole(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(chunking, "CLOUD_PIECE_SECONDS", 30.0)
    monkeypatch.setattr(chunking, "MIN_CHARS_PER_MINUTE", 0)
    monkeypatch.setattr(vad, "speech_probabilities", lambda samples, settings: None)
    path = _speech_with_pauses(tmp_path, 70.0, [28.0, 56.0])
    provider = _Provider([
        TranscriptionResult("we went to the cave"),
        TranscriptionResult("to the cave with scorpions"),
        TranscriptionResult("and drove away"),
    ])

    result = await _run(provider, path)

    assert [round(s["seconds"], 1) for s in provider.sent] == [28.3, 28.0, 13.7]
    assert result.text == "we went to the cave to the cave with scorpions and drove away"


async def test_a_looping_answer_is_asked_for_again_and_the_sound_one_kept(tmp_path):
    path = _audio(tmp_path, 60.0)
    loop = TranscriptionResult(" ".join(["we go to the cave with the scorpions now"] * 40))
    sound = TranscriptionResult(" ".join(f"word{i}" for i in range(60)))
    provider = _Provider([loop, sound])
    pacer = _Pacer()

    result = await _run(provider, path, pacer)

    assert result.text == sound.text
    assert len(provider.sent) == 2
    assert pacer.events == [("before", 0, 1), ("before", 0, 1), ("done", 1, 1)]


async def test_of_two_broken_answers_the_one_with_more_words_is_kept(tmp_path):
    path = _audio(tmp_path, 60.0)
    provider = _Provider([
        TranscriptionResult("only this"),
        TranscriptionResult("only this and that"),
    ])

    result = await _run(provider, path)

    assert result.text == "only this and that"


@pytest.mark.parametrize("loop_first", [True, False])
async def test_of_two_broken_answers_a_loop_never_beats_a_short_real_one(tmp_path, loop_first):
    path = _audio(tmp_path, 600.0, rate=16000, channels=1)
    real = TranscriptionResult("Okay, let us wait for the others. Anna is late again.")
    loop = TranscriptionResult("Thank you. " * 3000)
    provider = _Provider([loop, real] if loop_first else [real, loop])

    result = await _run(provider, path)

    assert result.text == real.text


async def test_of_two_answers_too_short_to_tell_apart_the_longer_is_kept(tmp_path):
    path = _audio(tmp_path, 5.0)
    provider = _Provider([TranscriptionResult(""), TranscriptionResult("Yes")])

    result = await _run(provider, path)

    assert result.text == "Yes"


async def test_when_asking_again_fails_the_first_answer_is_kept(tmp_path):
    path = _audio(tmp_path, 60.0)
    provider = _Provider([
        TranscriptionResult("only this"),
        ResourceUnavailableError("daily quota", headers={"Retry-After": "42188"}),
    ])

    result = await _run(provider, path)

    assert result.text == "only this"


async def test_the_tokens_of_both_answers_are_counted(tmp_path):
    path = _audio(tmp_path, 60.0)
    provider = _Provider([
        TranscriptionResult("only this", tokens_used=100),
        TranscriptionResult(" ".join(f"word{i}" for i in range(60)), tokens_used=300),
    ])

    result = await _run(provider, path)

    assert result.tokens_used == 400


async def test_an_answer_the_provider_calls_silent_is_not_asked_for_again(tmp_path):
    path = _audio(tmp_path, 60.0)
    provider = _Provider([TranscriptionResult("Thank you.", no_speech_prob=0.9)])

    result = await _run(provider, path)

    assert (result.text, result.no_speech_prob, len(provider.sent)) == ("", 1.0, 1)


def test_the_longest_cloud_piece_fits_groq_even_when_it_cannot_compress(tmp_path):
    mono = tmp_path / "mono.wav"
    seconds = chunking.CLOUD_PIECE_SECONDS
    noise = np.random.default_rng(1).integers(-32768, 32767, int(seconds * 16000), dtype=np.int16)
    sf.write(str(mono), noise, 16000, subtype="PCM_16")
    piece = tmp_path / "piece.flac"

    chunking._write_piece(mono, piece, 0.0, seconds)

    assert sf.info(str(piece)).duration == pytest.approx(seconds)
    assert piece.stat().st_size < GROQ_FILE_LIMIT


async def test_a_silent_piece_adds_no_text_and_does_not_set_the_language(tmp_path, short_pieces):
    path = _audio(tmp_path, 50.0)
    provider = _Provider([
        TranscriptionResult("Thank you.", detected_language="en", no_speech_prob=0.9),
        TranscriptionResult("добрий день", detected_language="uk", no_speech_prob=0.1),
    ])

    result = await _run(provider, path)

    assert (result.text, result.no_speech_prob) == ("добрий день", None)
    assert [s["language"] for s in provider.sent] == ["auto", "auto"]


async def test_all_silent_pieces_hand_back_a_result_the_pipeline_discards(tmp_path, short_pieces):
    path = _audio(tmp_path, 50.0)
    provider = _Provider([
        TranscriptionResult("Thank you.", no_speech_prob=0.9),
        TranscriptionResult("Bye.", no_speech_prob=0.7),
    ])

    result = await _run(provider, path)

    assert (result.text, result.no_speech_prob) == ("", 1.0)


async def test_a_piece_the_vad_finds_silent_is_never_sent(tmp_path, short_pieces):
    path = _audio(tmp_path, 70.0)
    provider = _Provider([
        TranscriptionResult("добрий день", detected_language="uk"),
        TranscriptionResult("до побачення"),
    ])
    pacer = _Pacer()

    result = await _run(provider, path, pacer, silent_pieces=frozenset({1}))

    assert (result.text, result.no_speech_prob) == ("добрий день до побачення", None)
    assert [s["path"].stem[-1] for s in provider.sent] == ["0", "2"]
    assert [e for e in pacer.events if e[0] == "done"] == [("done", n, 3) for n in (1, 2, 3)]


async def test_no_piece_with_speech_hands_back_certain_silence(tmp_path, short_pieces):
    path = _audio(tmp_path, 50.0)
    provider = _Provider([TranscriptionResult("Thank you.", no_speech_prob=0.2)])

    result = await _run(provider, path, silent_pieces=frozenset({1}))
    assert result.no_speech_prob is None

    provider = _Provider([])
    result = await _run(provider, path, silent_pieces=frozenset({0, 1}))
    assert (result.text, result.no_speech_prob, provider.sent) == ("", 1.0, [])


def _busy(seconds: str | None) -> ResourceUnavailableError:
    headers = {"Retry-After": seconds} if seconds is not None else None
    return ResourceUnavailableError("rate limit", headers=headers)


async def test_a_rate_limit_pauses_then_sends_the_same_piece_again(tmp_path):
    path = _audio(tmp_path, 5.0)
    provider = _Provider([_busy("7"), TranscriptionResult("after the pause")])
    pacer = _Pacer()

    result = await _run(provider, path, pacer)

    assert result.text == "after the pause"
    assert pacer.events == [("before", 0, 1), ("pause", 7.0), ("done", 1, 1)]
    assert provider.sent[0]["seconds"] == provider.sent[1]["seconds"]


@pytest.mark.parametrize(
    "answers",
    [
        [_busy(None)],
        [_busy(str(chunking.MAX_PAUSE_SECONDS + 1))],
        [_busy("1")] * (chunking.MAX_PAUSES_PER_PIECE + 1),
    ],
)
async def test_a_refusal_without_a_short_wait_ends_the_work(tmp_path, answers):
    path = _audio(tmp_path, 5.0)
    provider = _Provider(answers)

    with pytest.raises(ResourceUnavailableError):
        await _run(provider, path)

    assert provider.answers == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["job_x.wav"]


async def test_an_m4a_is_cut_into_pieces_like_any_other_recording(tmp_path, short_pieces):
    path = write_aac_m4a(tmp_path / "job_x.m4a", 70.0)
    provider = _Provider([TranscriptionResult("a")] * 3)

    await _run(provider, path)

    assert [round(s["seconds"]) for s in provider.sent] == [30, 30, 20]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["job_x.m4a"]


async def test_a_file_nothing_here_can_decode_goes_whole_in_one_request(tmp_path):
    path = tmp_path / "job_x.m4a"
    path.write_bytes(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 64)
    provider = _Provider([TranscriptionResult("whole", no_speech_prob=0.2)])
    pacer = _Pacer()

    result = await _run(provider, path, pacer, language="en")

    assert (result.text, result.no_speech_prob) == ("whole", 0.2)
    assert [(s["path"], s["language"]) for s in provider.sent] == [(path, "en")]
    assert pacer.events == [("before", 0, 1), ("done", 1, 1)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["job_x.m4a"]


def test_the_ffmpeg_selftest_passes_here_and_fails_without_pyav(monkeypatch):
    assert ffmpeg_selftest() == (True, "ok")

    monkeypatch.setitem(sys.modules, "av", None)
    ok, message = ffmpeg_selftest()

    assert ok is False
    assert message.startswith("decoding an AAC M4A raised: ")

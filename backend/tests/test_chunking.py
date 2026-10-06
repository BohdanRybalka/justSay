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

from app.core.audio_formats import ffmpeg_selftest
from app.core.errors import ResourceUnavailableError
from app.pipeline import chunking
from app.pipeline.chunking import join_at_seam, piece_spans, transcribe_in_pieces
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
    monkeypatch.setattr(chunking, "CLOUD_PIECE_SECONDS", 30.0)
    monkeypatch.setattr(chunking, "LOCAL_PIECE_SECONDS", 15.0)
    monkeypatch.setattr(chunking, "OVERLAP_SECONDS", 5.0)


@pytest.mark.parametrize(("duration", "count"), [(20.0, 1), (600.0, 1), (601.0, 2), (3600.0, 7)])
def test_spans_are_equal_overlap_and_cover_the_whole_recording(duration, count):
    spans = piece_spans(duration, 600.0)

    assert len(spans) == count
    assert spans[0][0] == 0.0
    assert spans[-1][1] == pytest.approx(duration)
    assert all(end - start <= 600.0 + 1e-9 for start, end in spans)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end - start == pytest.approx(chunking.OVERLAP_SECONDS)


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
    assert [round(s["seconds"], 2) for s in provider.sent] == [pytest.approx(26.67, abs=0.01)] * 3
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

    assert [round(s["seconds"]) for s in provider.sent] == [27, 27, 27]
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

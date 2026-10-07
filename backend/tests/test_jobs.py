"""Background file jobs — lifecycle, estimate, cancel, failure reasons, dictation first.

Most tests drive `JobQueue` with a stand-in for `process_audio` that makes the observer calls a
real run makes, when the test releases it. One runs the real pipeline over a fake provider to
prove the saved row and the clipboard.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from app.audio.vad import VadAnalysis
from app.config import settings
from app.core.audio_formats import UNREADABLE_HERE
from app.core.errors import ConfigurationError, NotReadyError, ResourceUnavailableError
from app.core.types import ProviderMode
from app.main import app
from app.pipeline import jobs, service
from app.pipeline.jobs import DictationGate, JobQueue, remove_leftover_files
from app.pipeline.jobs_router import get_job_queue
from app.pipeline.service import ProcessingResult
from app.stt.base import TranscriptionResult
from app.transcripts import history, vector_store


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _FakePipeline:
    """Stands in for `process_audio`; each call waits for `release` before it answers."""

    def __init__(
        self,
        *,
        duration: float | None = 60.0,
        outcome: object = "saved",
        pieces: int = 1,
        pause: float | None = None,
    ) -> None:
        self.duration = duration
        self.pause = pause
        self.outcome = outcome
        self.pieces = pieces
        self.piece = -1
        self.next_piece = asyncio.Event()
        self.release = asyncio.Event()
        self.transcribing = asyncio.Event()
        self.calls: list[dict] = []

    async def __call__(self, path: Path, **kwargs) -> ProcessingResult:
        self.calls.append({"path": path, **kwargs})
        observer = kwargs["observer"]
        await observer.before_transcribe("mock/model", self.duration)
        for index in range(self.pieces):
            await observer.before_piece(index, self.pieces)
            self.piece = index
            if self.pause is not None:
                await observer.pause(self.pause)
            self.transcribing.set()
            if index + 1 < self.pieces:
                await self.next_piece.wait()
                self.next_piece.clear()
            else:
                await self.release.wait()
            observer.piece_done(index + 1, self.pieces)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if self.outcome == "silence":
            return ProcessingResult("", 5, False, discarded_reason="silence")
        observer.before_save()
        if self.outcome == "saved":
            observer.saved(f"entry-{len(self.calls)}")
        return ProcessingResult("hello", 5, False, model_name="mock/model")


@pytest.fixture(autouse=True)
def _no_indexer(monkeypatch):
    monkeypatch.setattr(vector_store, "run_background_indexer", AsyncMock())


@pytest.fixture(autouse=True)
def fakes(monkeypatch) -> dict[str, _FakePipeline]:
    """Each file name's stand-in pipeline; an unregistered name fails rather than transcribing."""
    by_name: dict[str, _FakePipeline] = {}

    async def dispatch(path: Path, **kwargs) -> ProcessingResult:
        return await by_name[kwargs["source_name"]](path, **kwargs)

    monkeypatch.setattr(jobs, "process_audio", dispatch)
    return by_name


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def gate() -> DictationGate:
    return DictationGate()


@pytest.fixture
def speeds(monkeypatch) -> list[float]:
    known: list[float] = []
    monkeypatch.setattr(history, "processing_speeds", lambda model, limit: list(known))
    return known


@pytest.fixture
async def queue(tmp_path, clock, gate, speeds):
    built = JobQueue(tmp_path / "tmp", gate=gate, clock=clock)
    yield built
    for job in list(built._jobs.values()):
        if job.task is not None and not job.task.done():
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)


def _view(queue: JobQueue, job_id: str):
    return next((v for v in queue.views() if v.id == job_id), None)


def _start(queue: JobQueue, fakes: dict[str, _FakePipeline], pipeline: _FakePipeline) -> str:
    name = f"file-{len(fakes)}.wav"
    fakes[name] = pipeline
    return queue.add_file(b"RIFF....WAVE....", name)


async def _settle(queue: JobQueue, job_id: str) -> None:
    task = queue._jobs[job_id].task
    await asyncio.gather(task, return_exceptions=True)


async def test_a_job_moves_through_its_stages_and_ends_on_its_entry(queue, fakes):
    pipeline = _FakePipeline()
    fakes["talk.wav"] = pipeline
    job_id = queue.add_file(b"RIFF....WAVE....", "talk.wav")
    assert _view(queue, job_id).stage == "queued"
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)
    assert _view(queue, job_id).stage == "transcribing"
    pipeline.release.set()
    await _settle(queue, job_id)

    done = _view(queue, job_id)
    assert (done.stage, done.progress, done.entry_id) == ("done", 1.0, "entry-1")
    call = pipeline.calls[0]
    assert (call["source"], call["source_name"]) == ("file", "talk.wav")
    assert call["copy_to_clipboard"] is False
    assert not call["path"].exists()


async def test_progress_is_estimated_from_past_speeds_and_capped(
    queue, clock, speeds, fakes
):
    speeds.extend([0.01, 0.02, 0.5])
    pipeline = _FakePipeline(duration=60.0)
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)

    clock.now += 1.5
    halfway = _view(queue, job_id)
    assert halfway.progress == pytest.approx(0.5)
    clock.now += 60
    assert _view(queue, job_id).progress == jobs.ESTIMATE_CAP

    pipeline.release.set()
    await _settle(queue, job_id)


@pytest.mark.parametrize(("known", "duration"), [([], 60.0), ([0.02], None)])
async def test_no_percentage_without_past_speeds_or_a_known_length(
    queue, speeds, known, duration, fakes
):
    speeds.extend(known)
    pipeline = _FakePipeline(duration=duration)
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)

    view = _view(queue, job_id)
    assert (view.stage, view.progress) == ("transcribing", None)
    pipeline.release.set()
    await _settle(queue, job_id)


async def _until(condition) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("never happened")


@pytest.mark.parametrize(
    ("known", "first", "second"), [([], 0.0, 0.25), ([0.01, 0.02, 0.5], 0.125, 0.375)]
)
async def test_progress_counts_finished_pieces_plus_the_estimate_inside_one(
    queue, clock, speeds, fakes, known, first, second
):
    speeds.extend(known)
    pipeline = _FakePipeline(duration=60.0, pieces=4)
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)

    clock.now += 0.375
    assert _view(queue, job_id).progress == pytest.approx(first)
    pipeline.next_piece.set()
    await _until(lambda: pipeline.piece == 1)
    clock.now += 0.375
    assert _view(queue, job_id).progress == pytest.approx(second)

    for piece in (2, 3):
        pipeline.next_piece.set()
        await _until(lambda piece=piece: pipeline.piece == piece)
    pipeline.release.set()
    await _settle(queue, job_id)
    assert _view(queue, job_id).progress == 1.0


async def test_a_job_waits_for_a_dictation_before_each_piece(queue, gate, fakes):
    pipeline = _FakePipeline(pieces=2)
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)

    with gate.dictating():
        pipeline.next_piece.set()
        await asyncio.sleep(0.2)
        assert pipeline.piece == 0
    await _until(lambda: pipeline.piece == 1)
    pipeline.release.set()
    await _settle(queue, job_id)
    assert _view(queue, job_id).stage == "done"


async def test_a_paused_job_resumes_after_the_wait_it_was_given(queue, fakes):
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    queue.sleep = sleep
    pipeline = _FakePipeline(pause=12.0)
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)

    assert slept == [12.0]
    assert queue._jobs[job_id].paused is False
    pipeline.release.set()
    await _settle(queue, job_id)
    assert _view(queue, job_id).stage == "done"


async def test_a_job_cancelled_during_a_piece_never_starts_the_pause_it_is_asked_for(
    queue, fakes
):
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    queue.sleep = sleep
    pipeline = _FakePipeline()
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)
    queue.cancel_or_dismiss(job_id)

    with pytest.raises(jobs._JobCancelledError):
        await jobs._JobObserver(queue, queue._jobs[job_id]).pause(300.0)
    assert slept == []
    pipeline.release.set()
    await _settle(queue, job_id)


async def test_a_paused_job_stops_at_once_when_cancelled(queue, fakes):
    never = asyncio.Event()

    async def sleep(seconds: float) -> None:
        await never.wait()

    queue.sleep = sleep
    paused, after = _FakePipeline(pause=3600.0), _FakePipeline()
    job_id = _start(queue, fakes, paused)
    _start(queue, fakes, after)
    await _until(lambda: queue._jobs[job_id].paused)

    assert queue.cancel_or_dismiss(job_id) == "cancelled"
    await asyncio.wait_for(after.transcribing.wait(), 1)
    assert _view(queue, job_id).stage == "cancelled"
    assert not paused.calls[0]["path"].exists()
    after.release.set()


async def test_jobs_transcribe_one_at_a_time_oldest_first(queue, fakes):
    first, second = _FakePipeline(), _FakePipeline()
    first_id = _start(queue, fakes, first)
    second_id = _start(queue, fakes, second)
    await asyncio.wait_for(first.transcribing.wait(), 1)
    await asyncio.sleep(0.02)

    assert _view(queue, second_id).stage == "queued"
    assert second.calls == []
    first.release.set()
    await asyncio.wait_for(second.transcribing.wait(), 1)
    second.release.set()
    await _settle(queue, second_id)
    assert _view(queue, first_id).stage == "done"


async def test_cancelling_a_queued_job_never_transcribes_it(queue, fakes):
    first, second = _FakePipeline(), _FakePipeline()
    _start(queue, fakes, first)
    second_id = _start(queue, fakes, second)
    await asyncio.wait_for(first.transcribing.wait(), 1)
    scratch = queue._jobs[second_id].path

    assert queue.cancel_or_dismiss(second_id) == "cancelled"
    await _settle(queue, second_id)

    assert _view(queue, second_id).stage == "cancelled"
    assert second.calls == []
    assert not scratch.exists()
    first.release.set()


async def test_a_cancelled_transcription_holds_the_turn_until_its_answer_then_saves_nothing(
    queue, fakes
):
    cancelled, after = _FakePipeline(), _FakePipeline()
    job_id = _start(queue, fakes, cancelled)
    _start(queue, fakes, after)
    await asyncio.wait_for(cancelled.transcribing.wait(), 1)

    assert queue.cancel_or_dismiss(job_id) == "cancelled"
    await asyncio.sleep(0.02)
    assert _view(queue, job_id).stage == "cancelled"
    assert after.calls == []

    cancelled.release.set()
    await _settle(queue, job_id)
    view = _view(queue, job_id)
    assert (view.stage, view.entry_id) == ("cancelled", None)
    assert not cancelled.calls[0]["path"].exists()
    await asyncio.wait_for(after.transcribing.wait(), 1)
    after.release.set()


async def test_a_cancelled_job_stays_cancelled_when_its_answer_is_a_failure(queue, fakes):
    pipeline = _FakePipeline(outcome=ResourceUnavailableError("the provider said no"))
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)

    queue.cancel_or_dismiss(job_id)
    pipeline.release.set()
    await _settle(queue, job_id)

    view = _view(queue, job_id)
    assert (view.stage, view.error) == ("cancelled", None)


async def test_a_job_cancelled_while_a_dictation_ran_never_sends_its_audio(queue, gate, fakes):
    pipeline = _FakePipeline()
    with gate.dictating():
        job_id = _start(queue, fakes, pipeline)
        await asyncio.sleep(0.05)
        queue.cancel_or_dismiss(job_id)

    await _settle(queue, job_id)
    assert not pipeline.transcribing.is_set()
    assert _view(queue, job_id).stage == "cancelled"


async def test_a_job_being_saved_cannot_be_cancelled(queue, fakes):
    pipeline = _FakePipeline()
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)
    queue._jobs[job_id].stage = "saving"

    with pytest.raises(NotReadyError):
        queue.cancel_or_dismiss(job_id)
    queue._jobs[job_id].stage = "transcribing"
    pipeline.release.set()
    await _settle(queue, job_id)


async def test_a_finished_job_shows_for_a_minute_and_a_failed_one_until_dismissed(
    queue, clock, fakes
):
    done, silent = _FakePipeline(), _FakePipeline(outcome="silence")
    done_id = _start(queue, fakes, done)
    silent_id = _start(queue, fakes, silent)
    done.release.set()
    silent.release.set()
    await _settle(queue, silent_id)

    clock.now += jobs.FINISHED_SHOWN_SECONDS + 1
    assert [v.id for v in queue.views()] == [silent_id]
    assert done_id not in queue._jobs
    assert queue.cancel_or_dismiss(silent_id) == "dismissed"
    assert queue.views() == []


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("silence", jobs.NO_SPEECH_REASON),
        ("unsaved", jobs.FAILED_REASON),
        (ConfigurationError("Groq API key is missing."), jobs.NO_KEY_REASON),
        (ResourceUnavailableError("Gemini returned no transcription"), jobs.FAILED_REASON),
        (ResourceUnavailableError(UNREADABLE_HERE), UNREADABLE_HERE),
        (ResourceUnavailableError("rate limit", headers={"Retry-After": "30"}), jobs.BUSY_REASON),
        (
            ResourceUnavailableError("daily quota", headers={"Retry-After": "42188"}),
            jobs.LIMIT_REACHED_REASON,
        ),
        (KeyError("provider"), jobs.FAILED_REASON),
    ],
)
async def test_a_failed_job_says_why_in_plain_words(queue, outcome, reason, fakes):
    pipeline = _FakePipeline(outcome=outcome)
    job_id = _start(queue, fakes, pipeline)
    pipeline.release.set()
    await _settle(queue, job_id)

    view = _view(queue, job_id)
    assert (view.stage, view.error) == ("failed", reason)
    assert not pipeline.calls[0]["path"].exists()


async def test_a_job_holds_its_audio_back_while_a_dictation_is_processed(queue, gate, fakes):
    pipeline = _FakePipeline()
    with gate.dictating():
        job_id = _start(queue, fakes, pipeline)
        await asyncio.sleep(0.2)
        assert not pipeline.transcribing.is_set()
        assert queue._jobs[job_id].piece_started_at is None

    await asyncio.wait_for(pipeline.transcribing.wait(), 1)
    pipeline.release.set()
    await _settle(queue, job_id)


def test_leftover_job_files_are_removed_and_nothing_else(tmp_path):
    (tmp_path / "job_abc.wav").write_bytes(b"x")
    (tmp_path / "dictation.wav").write_bytes(b"x")

    remove_leftover_files(tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == ["dictation.wav"]


def test_processing_speeds_read_the_newest_rows_of_one_model(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    history.bootstrap(tmp_path)
    try:
        rows = [(10.0, 1000, "a"), (30.0, 1000, "a"), (None, 500, "a"), (5.0, 50, "b")]
        for audio, work, model in rows:
            history.save_entry(
                text="hi", duration_ms=work, language="en", model_name=model,
                audio_duration_seconds=audio, word_count=1, source="dictation",
            )

        assert history.processing_speeds("a", 50) == pytest.approx([0.03, 0.01])
        assert history.processing_speeds("a", 1) == pytest.approx([0.03])
    finally:
        with history._lock:
            history._close_conn_locked()


def _wav(tmp_path: Path) -> bytes:
    path = tmp_path / "speech.wav"
    sf.write(str(path), np.random.uniform(-0.1, 0.1, 16000).astype(np.float32), 16000)
    return path.read_bytes()


@pytest.fixture
async def client(tmp_path, monkeypatch, queue):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    app.dependency_overrides[get_job_queue] = lambda: queue
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_a_dropped_file_is_saved_as_a_file_entry_and_never_copied(
    client, queue, tmp_path, monkeypatch
):
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    history.bootstrap(tmp_path)
    stt = MagicMock(model_name="mock/provider", is_local=False, longest_piece_seconds=600.0)
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(text="from the file"))
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    monkeypatch.setattr(jobs, "process_audio", service.process_audio)
    try:
        with (
            patch("app.pipeline.service.get_provider", return_value=stt),
            patch("app.pipeline.service.analyze_vad", return_value=None),
            patch("app.pipeline.service.pyperclip.copy") as copy,
        ):
            upload = {"file": ("standup.wav", _wav(tmp_path), "audio/wav")}
            job_id = (await client.post("/jobs/file", files=upload)).json()["id"]
            await _settle(queue, job_id)

        listed = (await client.get("/jobs")).json()
        entry = history.get_page(limit=5).entries[0]
        assert listed[0]["stage"] == "done"
        assert listed[0]["entry_id"] == entry.id
        assert (entry.source, entry.source_name) == ("file", "standup.wav")
        assert entry.text == "from the file"
        assert stt.transcribe.call_args.args[0].suffix == ".flac"
        copy.assert_not_called()
    finally:
        with history._lock:
            history._close_conn_locked()


async def test_a_file_whose_every_piece_is_silent_is_never_sent(
    client, queue, tmp_path, monkeypatch
):
    stt = MagicMock(model_name="mock/provider", is_local=False, longest_piece_seconds=600.0)
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(text="Thank you."))
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    monkeypatch.setattr(service.audio_settings, "silence_vad_enabled", True)
    monkeypatch.setattr(jobs, "process_audio", service.process_audio)

    def vad(path: Path, _settings) -> VadAnalysis:
        return VadAnalysis(1, 10, 0.2, is_silent="piece" in path.name)

    with (
        patch("app.pipeline.service.get_provider", return_value=stt),
        patch("app.pipeline.service.analyze_vad", side_effect=vad),
    ):
        upload = {"file": ("quiet.wav", _wav(tmp_path), "audio/wav")}
        job_id = (await client.post("/jobs/file", files=upload)).json()["id"]
        await _settle(queue, job_id)

    view = _view(queue, job_id)
    assert (view.stage, view.error) == ("failed", jobs.NO_SPEECH_REASON)
    stt.transcribe.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "status"),
    [(b"", 400), (b"MZ" + b"\x00" * 64, 400)],
)
async def test_an_unusable_upload_is_refused_before_a_job_exists(client, queue, payload, status):
    resp = await client.post("/jobs/file", files={"file": ("a.wav", payload, "audio/wav")})

    assert resp.status_code == status
    assert queue.views() == []


async def test_cancel_answers_404_for_an_unknown_job_and_409_while_saving(client, queue, fakes):
    assert (await client.delete("/jobs/nope")).status_code == 404

    pipeline = _FakePipeline()
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)
    queue._jobs[job_id].stage = "saving"
    assert (await client.delete(f"/jobs/{job_id}")).status_code == 409
    queue._jobs[job_id].stage = "transcribing"
    cancelled = await client.delete(f"/jobs/{job_id}")
    assert (cancelled.status_code, cancelled.json()) == (200, {"outcome": "cancelled"})
    pipeline.release.set()
    await _settle(queue, job_id)

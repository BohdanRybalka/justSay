"""Background jobs — lifecycle, estimate, cancel, failure reasons, dictation first, meetings.

Most tests drive `JobQueue` with a stand-in for `process_audio` that makes the observer calls a
real run makes, when the test releases it. One runs the real pipeline over a fake provider to
prove the saved row.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from app.audio import scratch_router
from app.audio.meeting_recorder import leftover_recordings
from app.audio.pauses import SpeechProfile
from app.audio.vad import VadAnalysis
from app.config import settings
from app.core.audio_formats import UNREADABLE_HERE
from app.core.errors import (
    ConfigurationError,
    NotReadyError,
    ResourceUnavailableError,
)
from app.core.types import ProviderMode
from app.main import app
from app.pipeline import chunking, jobs, service
from app.pipeline.jobs import DictationGate, JobQueue, leftover_dictations, remove_leftover_files
from app.pipeline.jobs_router import get_job_queue
from app.pipeline.service import ProcessingResult
from app.stt.base import TranscriptionResult
from app.stt.config import stt_settings
from app.stt.local_whisper_cpp_cmd import MODEL_NOT_INSTALLED, ModelNotInstalledError
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
        sends: bool = False,
    ) -> None:
        self.sends = sends
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
            if self.sends:
                await observer.before_send()
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
            return ProcessingResult("", 5, discarded_reason="silence")
        observer.before_save()
        if self.outcome == "saved":
            observer.saved(f"entry-{len(self.calls)}")
        return ProcessingResult("hello", 5, model_name="mock/model")


@pytest.fixture
def noise_is_voice(monkeypatch):
    monkeypatch.setattr(
        chunking,
        "speech_profile",
        lambda samples, settings: SpeechProfile(
            [(0.0, samples.size / 16000)], samples.size / 16000
        ),
    )


@pytest.fixture(autouse=True)
def _no_indexer(monkeypatch):
    monkeypatch.setattr(vector_store, "run_background_indexer", AsyncMock())


@pytest.fixture(autouse=True)
def fakes(monkeypatch) -> dict[str, _FakePipeline]:
    """Each file's or recording's stand-in pipeline; an unregistered name fails instead."""
    by_name: dict[str, _FakePipeline] = {}

    async def dispatch(path: Path, **kwargs) -> ProcessingResult:
        return await by_name[kwargs["source_name"] or path.name](path, **kwargs)

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


def _recording(tmp_path: Path, fakes: dict[str, _FakePipeline], pipeline: _FakePipeline) -> Path:
    path = tmp_path / f"meeting_{len(fakes):012x}.wav"
    path.write_bytes(b"RIFF....WAVE....")
    fakes[path.name] = pipeline
    return path


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
    assert call["language"] == "auto"
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
        (ModelNotInstalledError(MODEL_NOT_INSTALLED), MODEL_NOT_INSTALLED),
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
    assert pipeline.calls[0]["path"].exists()


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


async def test_a_dropped_file_is_saved_as_a_file_entry(
    client, queue, tmp_path, monkeypatch, noise_is_voice
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


@pytest.mark.parametrize("engine", [ProviderMode.LOCAL, ProviderMode.CLOUD])
async def test_a_file_runs_on_the_chosen_engine_and_its_own_language_not_the_meetings(
    queue, fakes, monkeypatch, engine
):
    monkeypatch.setattr(settings.stt, "mode", engine)
    monkeypatch.setattr(settings.stt, "files_language", "en")
    monkeypatch.setattr(settings.stt, "meetings_language", "pl")
    pipeline = _FakePipeline()

    job_id = _start(queue, fakes, pipeline)
    pipeline.release.set()
    await _settle(queue, job_id)

    call = pipeline.calls[0]
    assert (call["source"], call["mode"], call["language"]) == ("file", engine, "en")


async def test_a_meeting_keeps_its_own_language_beside_a_files_language(
    queue, fakes, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    monkeypatch.setattr(settings.stt, "files_language", "en")
    monkeypatch.setattr(settings.stt, "meetings_language", "pl")
    pipeline = _FakePipeline()
    recording = _recording(tmp_path, fakes, pipeline)

    job_id = queue.add_meeting(recording)
    pipeline.release.set()
    await _settle(queue, job_id)

    call = pipeline.calls[0]
    assert (call["source"], call["mode"], call["language"]) == ("meeting", ProviderMode.CLOUD, "pl")


async def test_a_file_made_in_local_is_never_tried_again_in_the_cloud(
    queue, fakes, monkeypatch
):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
    failing = _FakePipeline(outcome="unsaved")
    job_id = _start(queue, fakes, failing)
    failing.release.set()
    await _settle(queue, job_id)
    working = _FakePipeline()
    fakes[next(name for name, fake in fakes.items() if fake is failing)] = working
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)

    queue.retry(job_id)
    working.release.set()
    await _settle(queue, job_id)

    assert working.calls[0]["mode"] == ProviderMode.LOCAL


@pytest.mark.parametrize("kind", ["file", "meeting"])
async def test_a_cloud_run_sends_nothing_more_once_local_is_chosen(
    queue, fakes, tmp_path, monkeypatch, kind
):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    pipeline = _FakePipeline(pieces=2, sends=True)
    if kind == "file":
        job_id = _start(queue, fakes, pipeline)
    else:
        job_id = queue.add_meeting(_recording(tmp_path, fakes, pipeline))
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)
    assert pipeline.piece == 0

    monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
    pipeline.next_piece.set()
    pipeline.release.set()
    await _settle(queue, job_id)

    assert pipeline.piece == 0
    assert _view(queue, job_id).stage == "failed"


@pytest.mark.parametrize("kind", ["file", "meeting"])
async def test_a_local_run_goes_on_when_cloud_is_chosen(queue, fakes, tmp_path, monkeypatch, kind):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
    pipeline = _FakePipeline(pieces=2, sends=True)
    if kind == "file":
        job_id = _start(queue, fakes, pipeline)
    else:
        job_id = queue.add_meeting(_recording(tmp_path, fakes, pipeline))
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)

    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    pipeline.next_piece.set()
    pipeline.release.set()
    await _settle(queue, job_id)

    assert _view(queue, job_id).stage == "done"


@pytest.mark.parametrize("engine", [ProviderMode.LOCAL, ProviderMode.CLOUD])
async def test_a_stopped_meeting_becomes_a_meeting_entry_on_the_chosen_engine(
    queue, fakes, tmp_path, monkeypatch, engine
):
    monkeypatch.setattr(settings.stt, "mode", engine)
    monkeypatch.setattr(settings.stt, "meetings_language", "pl")
    pipeline = _FakePipeline()
    recording = _recording(tmp_path, fakes, pipeline)

    job_id = queue.add_meeting(recording)
    assert _view(queue, job_id).kind == "meeting"
    assert _view(queue, job_id).name.startswith("Meeting · ")
    pipeline.release.set()
    await _settle(queue, job_id)

    assert _view(queue, job_id).stage == "done"
    call = pipeline.calls[0]
    assert (call["source"], call["source_name"], call["mode"]) == ("meeting", None, engine)
    assert call["language"] == "pl"
    assert not recording.exists()


@pytest.mark.parametrize(
    ("engine", "reason"),
    [
        (ProviderMode.LOCAL, jobs.MEETING_FAILED_HERE_REASON),
        (ProviderMode.CLOUD, jobs.MEETING_FAILED_REASON),
    ],
)
async def test_a_failed_meeting_keeps_its_recording_until_a_retry_saves_it(
    queue, fakes, tmp_path, monkeypatch, engine, reason
):
    monkeypatch.setattr(settings.stt, "mode", engine)
    failing = _FakePipeline(outcome=ResourceUnavailableError("engine down"))
    recording = _recording(tmp_path, fakes, failing)
    job_id = queue.add_meeting(recording)
    failing.release.set()
    await _settle(queue, job_id)

    assert (_view(queue, job_id).stage, _view(queue, job_id).error) == ("failed", reason)
    assert recording.exists()
    assert queue.kept_recording_names() == {recording.name}

    working = _FakePipeline(duration=60.0)
    fakes[recording.name] = working
    assert queue.retry(job_id) is True
    assert (_view(queue, job_id).stage, _view(queue, job_id).error) == ("queued", None)
    working.release.set()
    await _settle(queue, job_id)

    assert _view(queue, job_id).stage == "done"
    assert not recording.exists()
    assert queue.kept_recording_names() == frozenset()


async def test_a_silent_meeting_says_so_and_keeps_its_recording(queue, fakes, tmp_path):
    pipeline = _FakePipeline(outcome="silence")
    recording = _recording(tmp_path, fakes, pipeline)
    job_id = queue.add_meeting(recording)
    pipeline.release.set()
    await _settle(queue, job_id)

    assert _view(queue, job_id).error == jobs.MEETING_NO_SPEECH_REASON
    assert recording.exists()


async def test_only_a_failed_job_can_be_tried_again(queue, fakes, tmp_path):
    assert queue.retry("nope") is False

    running_file = _FakePipeline()
    file_id = _start(queue, fakes, running_file)
    await asyncio.wait_for(running_file.transcribing.wait(), 1)
    with pytest.raises(NotReadyError):
        queue.retry(file_id)
    running_file.release.set()
    await _settle(queue, file_id)

    running = _FakePipeline()
    meeting_id = queue.add_meeting(_recording(tmp_path, fakes, running))
    await asyncio.wait_for(running.transcribing.wait(), 1)
    with pytest.raises(NotReadyError):
        queue.retry(meeting_id)
    running.release.set()
    await _settle(queue, meeting_id)


async def test_a_failed_file_keeps_its_copy_and_a_retry_continues_after_its_finished_pieces(
    queue, fakes
):
    failing = _FakePipeline(outcome="unsaved")
    job_id = _start(queue, fakes, failing)
    failing.release.set()
    await _settle(queue, job_id)
    copy = failing.calls[0]["path"]
    assert copy.exists()

    working = _FakePipeline()
    fakes[next(name for name, fake in fakes.items() if fake is failing)] = working
    assert queue.retry(job_id) is True
    working.release.set()
    await _settle(queue, job_id)

    assert working.calls[0]["finished"] is failing.calls[0]["finished"]
    assert (_view(queue, job_id).stage, copy.exists()) == ("done", False)


async def test_dismissing_a_failed_file_deletes_its_copy(queue, fakes):
    failing = _FakePipeline(outcome="unsaved")
    job_id = _start(queue, fakes, failing)
    failing.release.set()
    await _settle(queue, job_id)

    assert queue.cancel_or_dismiss(job_id) == "dismissed"

    assert not failing.calls[0]["path"].exists()


async def test_removing_a_meeting_card_deletes_its_recording(queue, fakes, tmp_path):
    failing = _FakePipeline(outcome="unsaved")
    failed = _recording(tmp_path, fakes, failing)
    failed_id = queue.add_meeting(failed)
    failing.release.set()
    await _settle(queue, failed_id)
    blocker = _FakePipeline()
    blocker_id = queue.add_meeting(_recording(tmp_path, fakes, blocker))
    await asyncio.wait_for(blocker.transcribing.wait(), 1)
    queued = _recording(tmp_path, fakes, _FakePipeline())
    queued_id = queue.add_meeting(queued)

    assert queue.cancel_or_dismiss(failed_id) == "dismissed"
    assert queue.cancel_or_dismiss(queued_id) == "cancelled"
    await _settle(queue, queued_id)

    assert not failed.exists()
    assert not queued.exists()
    blocker.release.set()
    await _settle(queue, blocker_id)


async def test_leftover_meetings_are_listed_failed_and_can_be_tried_again(
    queue, fakes, tmp_path
):
    temp_dir = tmp_path / "leftovers"
    temp_dir.mkdir()
    older = temp_dir / "meeting_00000000000a.wav"
    newer = temp_dir / "meeting_00000000000b.wav"
    for path, written in ((older, 1_000_000), (newer, 2_000_000)):
        path.write_bytes(b"RIFF....WAVE....")
        os.utime(path, (written, written))
    for name in ("meeting_spill_00000000000a_mic.f32", "meeting_mix_00000000000a.f32", "rec_a.wav"):
        (temp_dir / name).write_bytes(b"x")

    queue.add_leftover_meetings(leftover_recordings(temp_dir))

    views = queue.views()
    assert [(v.kind, v.stage, v.error) for v in views] == [
        ("meeting", "failed", jobs.MEETING_FAILED_REASON)
    ] * 2
    assert queue.kept_recording_names() == {older.name, newer.name}
    pipeline = _FakePipeline()
    fakes[newer.name] = pipeline
    queue.retry(views[0].id)
    pipeline.release.set()
    await _settle(queue, views[0].id)
    assert not newer.exists()


def test_a_meeting_a_past_run_cut_off_says_so_and_can_be_tried_again(queue, tmp_path):
    """A leftover WAV may be a meeting that already failed; only a rebuilt one was cut off."""
    rebuilt = tmp_path / "meeting_00000000000c.wav"
    rebuilt.write_bytes(b"RIFF....WAVE....")

    queue.add_cut_off_meetings([rebuilt])

    [view] = queue.views()
    assert (view.kind, view.stage, view.error) == (
        "meeting",
        "failed",
        jobs.MEETING_INTERRUPTED_REASON,
    )
    assert queue.kept_recording_names() == {rebuilt.name}


async def test_a_meeting_runs_the_real_pipeline_on_the_chosen_engine(
    client, queue, tmp_path, monkeypatch, noise_is_voice
):
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    history.bootstrap(tmp_path)
    stt = MagicMock(model_name="mock/local", is_local=True, longest_piece_seconds=600.0)
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(text="from the call"))
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
    monkeypatch.setattr(jobs, "process_audio", service.process_audio)
    recording = tmp_path / "meeting_0123456789ab.wav"
    recording.write_bytes(_wav(tmp_path))
    try:
        with (
            patch("app.pipeline.service.get_provider", return_value=stt) as route,
            patch("app.pipeline.service.analyze_vad", return_value=None),
            patch("app.stt.local_setup.await_local_ready", new=AsyncMock()) as ready,
        ):
            job_id = queue.add_meeting(recording)
            await _settle(queue, job_id)

        assert route.call_args.args[0] == ProviderMode.LOCAL
        assert ready.await_args.args[0].mode == ProviderMode.LOCAL
        entry = history.get_page(limit=5).entries[0]
        assert (entry.source, entry.source_name, entry.text) == ("meeting", None, "from the call")
        assert (await client.get("/jobs")).json()[0]["entry_id"] == entry.id
    finally:
        with history._lock:
            history._close_conn_locked()


def _dictation_audio(tmp_path: Path) -> Path:
    path = tmp_path / "tmp" / "rec_0123456789ab.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF....WAVE....")
    return path


FRESH_ID = "0123456789abcdef0123456789abcdef"
STALE_ID = "fedcba9876543210fedcba9876543210"


def _kept_dictations(queue: JobQueue) -> list[Path]:
    return sorted(queue._temp_dir.glob("dictation_*.wav"))


def _keep_failed_dictation(
    queue: JobQueue, audio: Path, *, language: str = "uk", mode: ProviderMode = ProviderMode.CLOUD
) -> None:
    queue.keep_failed_dictation(audio, language=language, mode=mode)


def test_a_failed_dictation_keeps_its_audio_as_a_card_until_dismissed(queue, tmp_path):
    audio = _dictation_audio(tmp_path)
    _keep_failed_dictation(queue, audio)

    [card] = queue.views()
    assert (card.kind, card.stage, card.error) == (
        "dictation",
        "failed",
        jobs.DICTATION_FAILED_REASON,
    )
    assert not audio.exists()
    assert len(_kept_dictations(queue)) == 1

    assert queue.cancel_or_dismiss(card.id) == "dismissed"
    assert _kept_dictations(queue) == []


def test_a_failed_dictation_whose_audio_is_gone_makes_no_card(queue, tmp_path):
    _keep_failed_dictation(queue, tmp_path / "gone.wav")

    assert queue.views() == []


async def test_a_kept_dictation_is_sent_again_in_its_own_mode_and_saved_once(
    queue, tmp_path, monkeypatch
):
    sent: list[dict] = []

    async def transcribe_again(path, **kwargs):
        sent.append(kwargs)
        return ProcessingResult("hello", 5, model_name="mock/model", entry_id="entry-9")

    monkeypatch.setattr(jobs, "process_audio", transcribe_again)
    monkeypatch.setattr(stt_settings, "mode", ProviderMode.LOCAL)
    audio = _dictation_audio(tmp_path)
    _keep_failed_dictation(queue, audio, language="de", mode=ProviderMode.LOCAL)
    [card] = queue.views()

    assert queue.retry(card.id) is True
    await _settle(queue, card.id)

    assert (_view(queue, card.id).stage, _view(queue, card.id).entry_id) == ("done", "entry-9")
    assert _kept_dictations(queue) == []
    assert sent == [{"language": "de", "source": "dictation", "mode": ProviderMode.LOCAL}]


async def test_a_retried_dictation_that_holds_no_speech_drops_its_card_and_audio(
    queue, tmp_path, monkeypatch
):
    silent = AsyncMock(return_value=ProcessingResult("", 5, discarded_reason="silence"))
    monkeypatch.setattr(jobs, "process_audio", silent)
    _keep_failed_dictation(queue, _dictation_audio(tmp_path))
    [card] = queue.views()

    queue.retry(card.id)
    await _settle(queue, card.id)

    assert _view(queue, card.id) is None
    assert _kept_dictations(queue) == []


async def test_a_retry_of_a_dictation_does_not_wait_for_a_running_file(
    queue, fakes, tmp_path, monkeypatch
):
    file_pipeline = _FakePipeline()
    file_id = _start(queue, fakes, file_pipeline)
    await asyncio.wait_for(file_pipeline.transcribing.wait(), 1)

    async def transcribe_again(path, **kwargs):
        return ProcessingResult("hello", 5, entry_id="entry-2")

    monkeypatch.setattr(jobs, "process_audio", transcribe_again)
    _keep_failed_dictation(queue, _dictation_audio(tmp_path))
    dictation_id = next(card.id for card in queue.views() if card.kind == "dictation")

    queue.retry(dictation_id)
    await asyncio.wait_for(_settle(queue, dictation_id), 1)

    assert _view(queue, dictation_id).stage == "done"
    assert _view(queue, file_id).stage == "transcribing"
    file_pipeline.release.set()
    await _settle(queue, file_id)


async def test_a_dictation_being_turned_into_text_cannot_be_dismissed(
    queue, tmp_path, monkeypatch
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(path, **kwargs):
        started.set()
        await release.wait()
        return ProcessingResult("hello", 5, entry_id="entry-3")

    monkeypatch.setattr(jobs, "process_audio", slow)
    _keep_failed_dictation(queue, _dictation_audio(tmp_path))
    [card] = queue.views()
    queue.retry(card.id)
    await asyncio.wait_for(started.wait(), 1)

    with pytest.raises(NotReadyError):
        queue.cancel_or_dismiss(card.id)
    release.set()
    await _settle(queue, card.id)
    assert _view(queue, card.id).stage == "done"


def test_a_failed_dictation_is_dropped_with_its_audio_after_fourteen_days(
    queue, tmp_path, clock
):
    _keep_failed_dictation(queue, _dictation_audio(tmp_path))
    [card] = queue.views()

    clock.now += jobs.FAILED_DICTATION_KEEP_SECONDS
    assert _view(queue, card.id) is not None

    clock.now += 1
    assert queue.views() == []
    assert _kept_dictations(queue) == []


def test_clear_temporary_audio_leaves_a_kept_dictation(queue, tmp_path):
    _keep_failed_dictation(queue, _dictation_audio(tmp_path))

    scratch_router._reap_scratch_files(tmp_path / "tmp", frozenset())

    assert len(_kept_dictations(queue)) == 1


def test_leftover_dictations_are_listed_as_failed_unless_kept_past_fourteen_days(
    queue, tmp_path
):
    temp_dir = tmp_path / "leftovers"
    temp_dir.mkdir()
    now = time.time()
    fresh = temp_dir / f"dictation_{FRESH_ID}_local_uk.wav"
    stale = temp_dir / f"dictation_{STALE_ID}_local_uk.wav"
    other = temp_dir / "rec_00000000000c.wav"
    chunk_scratch = temp_dir / f"dictation_{FRESH_ID}_local_uk-16k.wav"
    too_old = jobs.FAILED_DICTATION_KEEP_SECONDS + 3600
    for path, age in ((fresh, 60.0), (stale, too_old), (other, 60.0), (chunk_scratch, 60.0)):
        path.write_bytes(b"RIFF....WAVE....")
        os.utime(path, (now - age, now - age))

    queue.add_leftover_dictations(leftover_dictations(temp_dir))

    assert [(v.kind, v.stage, v.error) for v in queue.views()] == [
        ("dictation", "failed", jobs.DICTATION_FAILED_REASON)
    ]
    assert fresh.exists() and other.exists() and not stale.exists()


async def test_a_kept_dictation_keeps_its_mode_and_language_across_a_restart(
    queue, tmp_path, monkeypatch
):
    queue.keep_failed_dictation(_dictation_audio(tmp_path), language="de", mode=ProviderMode.CLOUD)
    restarted = JobQueue(queue._temp_dir, gate=queue.gate, clock=queue.clock)
    restarted.add_leftover_dictations(jobs.leftover_dictations(queue._temp_dir))
    [card] = restarted.views()

    sent: list[dict] = []

    async def transcribe_again(path, **kwargs):
        sent.append(kwargs)
        return ProcessingResult("hello", 5, entry_id="entry-4")

    monkeypatch.setattr(jobs, "process_audio", transcribe_again)
    monkeypatch.setattr(stt_settings, "mode", ProviderMode.CLOUD)
    restarted.retry(card.id)
    await _settle(restarted, card.id)

    assert sent == [{"language": "de", "source": "dictation", "mode": ProviderMode.CLOUD}]


async def test_a_retry_that_fails_again_keeps_its_audio_as_a_card(queue, tmp_path, monkeypatch):
    monkeypatch.setattr(
        jobs,
        "process_audio",
        AsyncMock(side_effect=ResourceUnavailableError("the provider is down")),
    )
    _keep_failed_dictation(queue, _dictation_audio(tmp_path))
    [card] = queue.views()

    queue.retry(card.id)
    await _settle(queue, card.id)

    assert (_view(queue, card.id).stage, _view(queue, card.id).error) == (
        "failed",
        jobs.DICTATION_FAILED_REASON,
    )
    assert len(_kept_dictations(queue)) == 1


@pytest.mark.parametrize(
    ("made_in", "now", "sent_to"),
    [
        (ProviderMode.CLOUD, ProviderMode.CLOUD, ProviderMode.CLOUD),
        (ProviderMode.CLOUD, ProviderMode.LOCAL, ProviderMode.LOCAL),
        (ProviderMode.LOCAL, ProviderMode.CLOUD, ProviderMode.LOCAL),
        (ProviderMode.LOCAL, ProviderMode.LOCAL, ProviderMode.LOCAL),
    ],
)
async def test_a_retry_goes_to_the_cloud_only_when_cloud_was_made_and_is_on(
    queue, tmp_path, monkeypatch, made_in, now, sent_to
):
    indexer = MagicMock()
    monkeypatch.setattr(JobQueue, "_spawn_indexer", staticmethod(indexer))
    sent: list[dict] = []

    async def transcribe_again(path, **kwargs):
        sent.append(kwargs)
        return ProcessingResult("hello", 5, entry_id="entry-5")

    monkeypatch.setattr(jobs, "process_audio", transcribe_again)
    monkeypatch.setattr(stt_settings, "mode", now)
    _keep_failed_dictation(queue, _dictation_audio(tmp_path), mode=made_in)
    [card] = queue.views()

    queue.retry(card.id)
    await _settle(queue, card.id)

    assert sent[0]["mode"] is sent_to
    assert indexer.called is (sent_to is ProviderMode.CLOUD)


def test_a_kept_dictation_name_carries_a_safe_language(queue, tmp_path):
    _keep_failed_dictation(queue, _dictation_audio(tmp_path), language="../escape")

    [kept] = _kept_dictations(queue)
    assert kept.name.endswith("_cloud_auto.wav")


@pytest.fixture
def move_blocked_by_an_engine(monkeypatch):
    def _held_open(self, target):
        raise PermissionError("the file is in use by another process")

    monkeypatch.setattr(Path, "replace", _held_open)


@pytest.mark.usefixtures("move_blocked_by_an_engine")
def test_a_dictation_an_engine_still_reads_is_copied_when_it_cannot_be_moved(queue, tmp_path):
    audio = _dictation_audio(tmp_path)
    _keep_failed_dictation(queue, audio)

    [card] = queue.views()
    assert (card.kind, card.stage) == ("dictation", "failed")
    [kept] = _kept_dictations(queue)
    assert kept.read_bytes() == audio.read_bytes()
    assert audio.exists()


@pytest.mark.usefixtures("move_blocked_by_an_engine")
def test_a_copy_that_breaks_halfway_leaves_no_card_and_no_partial_file(
    queue, tmp_path, monkeypatch
):
    audio = _dictation_audio(tmp_path)

    def _breaks_halfway(source, target):
        Path(target).write_bytes(b"RIFF")
        raise OSError("the disk is full")

    monkeypatch.setattr(jobs.shutil, "copyfile", _breaks_halfway)
    _keep_failed_dictation(queue, audio)

    assert queue.views() == []
    assert _kept_dictations(queue) == []


def test_a_restart_removes_a_recording_only_when_a_kept_dictation_holds_the_same_audio(tmp_path):
    copied = tmp_path / "rec_0123456789ab.wav"
    only_copy = tmp_path / "rec_ba9876543210.wav"
    kept = tmp_path / f"dictation_{FRESH_ID}_cloud_uk.wav"
    copied.write_bytes(b"RIFF copied")
    kept.write_bytes(b"RIFF copied")
    only_copy.write_bytes(b"RIFF lonely")

    remove_leftover_files(tmp_path)

    assert not copied.exists()
    assert only_copy.exists()
    assert kept.exists()


def test_a_restart_keeps_a_recording_it_cannot_compare_with_a_kept_dictation(tmp_path, monkeypatch):
    recording = tmp_path / "rec_0123456789ab.wav"
    recording.write_bytes(b"RIFF copied")
    (tmp_path / f"dictation_{FRESH_ID}_cloud_uk.wav").write_bytes(b"RIFF copied")

    def _unreadable(first, second, shallow):
        raise OSError("the file is in use by another process")

    monkeypatch.setattr(jobs.filecmp, "cmp", _unreadable)
    remove_leftover_files(tmp_path)

    assert recording.exists()


def test_a_restart_clears_what_a_retry_cut_short_left_behind(queue, tmp_path):
    temp_dir = queue._temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    kept = temp_dir / f"dictation_{FRESH_ID}_cloud_uk.wav"
    left_over = [
        temp_dir / f"dictation_{FRESH_ID}_cloud_uk-16k.wav",
        temp_dir / f"dictation_{FRESH_ID}_cloud_uk-piece0.flac",
    ]
    for path in [kept, *left_over]:
        path.write_bytes(b"RIFF....WAVE....")

    jobs.remove_leftover_files(temp_dir)

    assert kept.exists()
    assert not any(path.exists() for path in left_over)


async def test_a_retry_stopped_by_a_switch_fails_keeping_its_audio_without_a_refusal_warning(
    queue, tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(
        jobs,
        "process_audio",
        AsyncMock(side_effect=chunking.RouteStoppedError("Cloud was switched off")),
    )
    _keep_failed_dictation(queue, _dictation_audio(tmp_path))
    [card] = queue.views()

    queue.retry(card.id)
    await _settle(queue, card.id)

    assert _view(queue, card.id).stage == "failed"
    assert len(_kept_dictations(queue)) == 1
    assert not [record for record in caplog.records if "was refused" in record.getMessage()]


async def test_retry_answers_404_for_an_unknown_job_and_409_for_one_not_failed(
    client, queue, fakes
):
    assert (await client.post("/jobs/nope/retry")).status_code == 404

    pipeline = _FakePipeline()
    job_id = _start(queue, fakes, pipeline)
    await asyncio.wait_for(pipeline.transcribing.wait(), 1)
    assert (await client.post(f"/jobs/{job_id}/retry")).status_code == 409
    pipeline.release.set()
    await _settle(queue, job_id)

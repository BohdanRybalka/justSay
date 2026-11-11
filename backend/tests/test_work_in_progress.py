"""What `GET /pipeline/work-in-progress` reports an update has to wait for, and that the
endpoints doing the work hold it from the moment they start until a job carries it."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.audio.dependencies import get_active_meeting_recorder, get_meeting_recorder, get_recorder
from app.audio.meeting_recorder import MeetingCaptureEmptyError, MeetingRecording
from app.core.errors import NotReadyError
from app.main import app
from app.pipeline.jobs import _Job, dictation_gate
from app.pipeline.router import WorkInProgressView

_REPO = Path(__file__).resolve().parents[2]

RECORDING = SimpleNamespace(is_recording=True, is_busy=True)
OPENING_DEVICES = SimpleNamespace(is_recording=False, is_busy=True)
STOPPING = SimpleNamespace(is_recording=False, is_busy=True)
IDLE_MEETING = SimpleNamespace(is_recording=False, is_busy=False)


@pytest.fixture
async def client():
    app.dependency_overrides[get_recorder] = lambda: SimpleNamespace(is_recording=False)
    app.dependency_overrides[get_active_meeting_recorder] = lambda: IDLE_MEETING
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def _work(client: AsyncClient) -> str | None:
    resp = await client.get("/pipeline/work-in-progress")
    assert resp.status_code == 200
    return resp.json()["work"]


def _dictation_recorder(stop) -> MagicMock:
    return MagicMock(is_recording=True, stop=stop, last_duration_seconds=1.0)


@pytest.mark.parametrize(
    "dictating, meeting, expected",
    [
        (False, None, None),
        (False, IDLE_MEETING, None),
        (False, RECORDING, "meeting"),
        (False, OPENING_DEVICES, "meeting"),
        (True, IDLE_MEETING, "dictation"),
    ],
)
async def test_a_recording_is_reported_by_what_records_it(client, dictating, meeting, expected):
    app.dependency_overrides[get_recorder] = lambda: SimpleNamespace(is_recording=dictating)
    app.dependency_overrides[get_active_meeting_recorder] = lambda: meeting

    assert await _work(client) == expected


async def test_a_meeting_being_recorded_is_named_before_anything_else_running(client):
    app.dependency_overrides[get_recorder] = lambda: SimpleNamespace(is_recording=True)
    app.dependency_overrides[get_active_meeting_recorder] = lambda: RECORDING

    with dictation_gate.dictating(), app.state.jobs.receiving():
        assert await _work(client) == "meeting"
        app.dependency_overrides[get_active_meeting_recorder] = lambda: IDLE_MEETING
        assert await _work(client) == "dictation"


async def test_a_dictation_being_processed_is_reported_as_a_dictation(client):
    with dictation_gate.dictating():
        assert await _work(client) == "dictation"
    assert await _work(client) is None


async def test_a_job_counts_until_its_worker_lets_go(client):
    result = SimpleNamespace(discarded_reason="no speech")
    release = asyncio.Event()

    async def transcribe(*_args, **_kwargs):
        await release.wait()
        return result

    with patch("app.pipeline.jobs.process_audio", transcribe):
        job_id = app.state.jobs.add_file(b"RIFF", "a.wav")
        assert await _work(client) == "transcription"
        assert app.state.jobs.cancel_or_dismiss(job_id) == "cancelled"
        assert await _work(client) == "transcription"
        release.set()
        await asyncio.wait_for(app.state.jobs._jobs[job_id].task, timeout=2)

    assert await _work(client) is None


async def test_a_job_with_no_worker_left_is_not_work_in_progress(client, tmp_path):
    failed = _Job(id="j", kind="file", name="a.wav", path=tmp_path, stage="failed")
    app.state.jobs._jobs["j"] = failed

    assert await _work(client) is None


async def test_a_meeting_still_saving_after_stop_is_a_transcription(client):
    app.dependency_overrides[get_active_meeting_recorder] = lambda: STOPPING

    with app.state.jobs.receiving():
        assert await _work(client) == "transcription"


async def test_a_dictation_counts_as_processed_while_its_recording_is_written(client, tmp_path):
    held_while_stopping = []

    async def stop(_session_id=None) -> Path:
        held_while_stopping.append(dictation_gate.is_processing)
        return tmp_path / "rec.wav"

    app.dependency_overrides[get_recorder] = lambda: _dictation_recorder(stop)
    result = SimpleNamespace(text="hi", duration_ms=1, model_name="m")
    with patch("app.pipeline.router.process_audio", AsyncMock(return_value=result)):
        resp = await client.post("/pipeline/dictate")

    assert resp.status_code == 200
    assert held_while_stopping == [True]
    assert not dictation_gate.is_processing


@pytest.mark.parametrize(
    "failure, status", [(NotReadyError("busy"), 409), (RuntimeError("crash"), 500)]
)
async def test_a_dictation_that_fails_is_no_longer_processed(client, tmp_path, failure, status):
    recording = tmp_path / "rec.wav"
    recording.write_bytes(b"RIFF")

    async def stop(_session_id=None) -> Path:
        return recording

    app.dependency_overrides[get_recorder] = lambda: _dictation_recorder(stop)
    with patch("app.pipeline.router.process_audio", AsyncMock(side_effect=failure)):
        resp = await client.post("/pipeline/dictate")

    assert resp.status_code == status
    assert not dictation_gate.is_processing


async def test_a_stopped_meeting_counts_as_arriving_until_its_job_is_queued(
    client, tmp_path, recording_queue
):
    arriving_while_stopping = []

    class _Stopping:
        is_busy = True

        async def stop(self) -> MeetingRecording:
            arriving_while_stopping.append(recording_queue.arriving)
            return MeetingRecording(path=tmp_path / "m.wav", duration_seconds=1.0, incident=None)

    app.dependency_overrides[get_meeting_recorder] = _Stopping

    resp = await client.post("/audio/meeting/stop")

    assert resp.status_code == 200
    assert arriving_while_stopping == [1]
    assert recording_queue.arriving == 0
    assert recording_queue.added == [tmp_path / "m.wav"]


async def test_a_meeting_stop_that_fails_is_no_longer_arriving(client):
    class _StopsWithNothing:
        is_busy = True

        async def stop(self) -> MeetingRecording:
            raise MeetingCaptureEmptyError("No audio data captured")

    app.dependency_overrides[get_meeting_recorder] = _StopsWithNothing

    resp = await client.post("/audio/meeting/stop")

    assert resp.status_code == 410
    assert not app.state.jobs.is_working


async def test_a_file_still_uploading_is_a_transcription(client):
    upload = httpx.Request(
        "POST", "http://test/jobs/file", files={"file": ("a.wav", b"RIFF" + b"\0" * 4096)}
    )
    first_chunk = {"type": "http.request", "body": upload.read()[:64], "more_body": True}
    chunks = [first_chunk]

    async def receive() -> dict:
        if not chunks:
            await asyncio.Event().wait()
        return chunks.pop()

    async def send(_message: dict) -> None:
        return None

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/jobs/file",
        "raw_path": b"/jobs/file",
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in upload.headers.items()],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }

    async def until_reported() -> None:
        while await _work(client) is None:
            await asyncio.sleep(0.01)

    uploading = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(until_reported(), timeout=2)
        assert app.state.jobs.views() == []
        assert await _work(client) == "transcription"
    finally:
        uploading.cancel()
        await asyncio.gather(uploading, return_exceptions=True)
    assert await _work(client) is None


async def test_asking_for_the_jobs_is_not_itself_a_file_arriving(client, monkeypatch):
    queue = app.state.jobs
    working_while_listed = []

    def views() -> list:
        working_while_listed.append(queue.is_working)
        return []

    monkeypatch.setattr(queue, "views", views)

    assert (await client.get("/jobs")).status_code == 200
    assert working_while_listed == [False]


def test_the_kinds_of_work_are_spelled_alike_by_the_backend_the_shell_and_the_page():
    python = set(get_args(get_args(WorkInProgressView.model_fields["work"].annotation)[0]))
    lib_rs = (_REPO / "src-tauri" / "src" / "lib.rs").read_text(encoding="utf-8")
    rust_enum = re.search(r"enum Work \{([^}]*)\}", lib_rs).group(1)
    rust = {variant.lower() for variant in re.findall(r"\b([A-Z]\w*),", rust_enum)}
    version_row = (_REPO / "src/settings/tabs/version-row.ts").read_text(encoding="utf-8")
    page_type = re.search(r"type Work = ([^;]*);", version_row).group(1)
    page = set(re.findall(r'"(\w+)"', page_type))

    assert python == rust == page == {"meeting", "dictation", "transcription"}

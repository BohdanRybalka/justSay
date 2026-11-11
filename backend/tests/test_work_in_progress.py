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
from app.audio.meeting_recorder import MeetingRecording
from app.main import app
from app.pipeline.jobs import _Job, dictation_gate
from app.pipeline.router import WorkInProgressView

_REPO = Path(__file__).resolve().parents[2]

RECORDING = SimpleNamespace(is_recording=True, is_busy=True)
OPENING_DEVICES = SimpleNamespace(is_recording=False, is_busy=True)
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


async def test_a_dictation_being_processed_is_reported_as_a_dictation(client):
    with dictation_gate.dictating():
        assert await _work(client) == "dictation"
    assert await _work(client) is None


@pytest.mark.parametrize(
    "stage, expected",
    [
        ("queued", "transcription"),
        ("transcribing", "transcription"),
        ("saving", "transcription"),
        ("done", None),
        ("failed", None),
        ("cancelled", None),
    ],
)
async def test_a_job_counts_while_it_waits_runs_or_saves(client, tmp_path, stage, expected):
    app.state.jobs._jobs["j"] = _Job(id="j", kind="file", name="a.wav", path=tmp_path, stage=stage)

    assert await _work(client) == expected


async def test_a_meeting_still_saving_after_stop_is_a_transcription(client):
    app.dependency_overrides[get_active_meeting_recorder] = lambda: OPENING_DEVICES

    with app.state.jobs.receiving():
        assert await _work(client) == "transcription"


async def test_a_dictation_counts_as_processed_while_its_recording_is_written(
    client, tmp_path
):
    recording = tmp_path / "rec.wav"
    held_while_stopping = []

    async def stop(_session_id=None) -> Path:
        held_while_stopping.append(dictation_gate.is_processing)
        return recording

    recorder = MagicMock(is_recording=True, stop=stop, last_duration_seconds=1.0)
    app.dependency_overrides[get_recorder] = lambda: recorder
    result = SimpleNamespace(text="hi", duration_ms=1, model_name="m")
    with patch("app.pipeline.router.process_audio", AsyncMock(return_value=result)):
        resp = await client.post("/pipeline/dictate")

    assert resp.status_code == 200
    assert held_while_stopping == [True]
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


async def test_a_file_still_uploading_is_a_transcription(client):
    upload = httpx.Request(
        "POST", "http://test/jobs/file", files={"file": ("a.wav", b"RIFF" + b"\0" * 4096)}
    )
    body = upload.read()
    rest_sent = asyncio.Event()
    chunks = [
        {"type": "http.request", "body": body[:64], "more_body": True},
        {"type": "http.request", "body": body[64:], "more_body": False},
    ]

    async def receive() -> dict:
        if len(chunks) == 1:
            await rest_sent.wait()
        return chunks.pop(0)

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

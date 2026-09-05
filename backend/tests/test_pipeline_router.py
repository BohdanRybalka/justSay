"""Pipeline router — /pipeline/process-file's ``language`` query-param
default and its upload-content validation.

Dedicated router-test file, split from the service-level `test_pipeline.py`
the same way `test_preferences_router.py` is split from `test_user_settings.py`
(spec 019). `app.pipeline.router.process_audio` is patched so no real STT
call, history write, or clipboard access happens — this file only asserts
what the router forwards to `process_audio`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.audio import get_recorder
from app.main import app


def _wav_bytes(payload_size: int = 1024) -> bytes:
    """Synthesise a minimal RIFF/WAVE header + payload (mirrors
    test_audio_formats.py's helper — validate_audio_upload() requires a
    real-looking WAV container, not arbitrary bytes)."""
    return b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 4) + (b"\x00" * payload_size)


def _fake_result() -> SimpleNamespace:
    """Stand-in for ProcessingResult — process_file() does
    `DictateResponse(**result.__dict__)`, so this needs the same fields."""
    return SimpleNamespace(
        text="hello",
        duration_ms=100,
        copied_to_clipboard=True,
        model_name="mock/provider",
        fallback_reason=None,
    )


@pytest.fixture
async def client(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_process_file_defaults_to_auto_language_when_query_param_omitted(client):
    """The new default (spec 019): dropping a file with no ``language``
    query param must route through as ``language="auto"``."""
    mock_process_audio = AsyncMock(return_value=_fake_result())
    with patch("app.pipeline.router.process_audio", mock_process_audio):
        resp = await client.post(
            "/pipeline/process-file",
            files={"file": ("speech.wav", _wav_bytes(), "audio/wav")},
        )

    assert resp.status_code == 200
    assert mock_process_audio.call_args.kwargs["language"] == "auto"


@pytest.mark.anyio
async def test_process_file_forwards_explicit_language_code_unchanged(client):
    """Regression: an explicit ``language`` query param must still reach
    process_audio() verbatim — the new "auto" default doesn't shadow it."""
    mock_process_audio = AsyncMock(return_value=_fake_result())
    with patch("app.pipeline.router.process_audio", mock_process_audio):
        resp = await client.post(
            "/pipeline/process-file?language=uk",
            files={"file": ("speech.wav", _wav_bytes(), "audio/wav")},
        )

    assert resp.status_code == 200
    assert mock_process_audio.call_args.kwargs["language"] == "uk"




@pytest.mark.anyio
async def test_process_file_rejects_extension_content_mismatch(client):
    """`.wav` filename with non-WAV bytes is rejected at the validator boundary,
    not handed off to the STT provider where it would 500 deep inside soundfile."""
    fake_payload = b"MZ" + (b"\x00" * 64)
    with patch("app.pipeline.router.process_audio", AsyncMock(return_value=_fake_result())):
        resp = await client.post(
            "/pipeline/process-file",
            files={"file": ("evil.wav", fake_payload, "audio/wav")},
        )

    assert resp.status_code == 400
    assert "does not match" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_process_file_rejects_empty_file(client):
    with patch("app.pipeline.router.process_audio", AsyncMock(return_value=_fake_result())):
        resp = await client.post(
            "/pipeline/process-file",
            files={"file": ("speech.wav", b"", "audio/wav")},
        )

    assert resp.status_code == 400
    assert "too small" in resp.json()["detail"].lower()


def _refuse_to_unlink(self, missing_ok: bool = False):
    raise OSError("the file is in use by another process")


@pytest.mark.anyio
async def test_process_file_returns_the_transcription_when_the_scratch_delete_fails(
    client, monkeypatch, caplog
):
    """The scratch delete sits in a ``finally``. Unguarded, an ``OSError`` there
    replaced the response that was about to be returned: a completed
    transcription — already copied to the clipboard and saved to history —
    reached the widget as a bare 500, which `src/api.ts` degrades to "Failed".
    """
    monkeypatch.setattr(Path, "unlink", _refuse_to_unlink)

    with (
        patch("app.pipeline.router.process_audio", AsyncMock(return_value=_fake_result())),
        caplog.at_level(logging.WARNING, logger="app.pipeline.router"),
    ):
        resp = await client.post(
            "/pipeline/process-file",
            files={"file": ("speech.wav", _wav_bytes(), "audio/wav")},
        )

    assert resp.status_code == 200
    assert resp.json()["text"] == "hello"
    assert [r for r in caplog.records if r.name == "app.pipeline.router" and r.exc_info]


@pytest.mark.anyio
async def test_dictate_returns_the_transcription_when_the_recording_delete_fails(
    client, tmp_path, monkeypatch, caplog
):
    """The same ``finally`` on the dictation path, which is the one a user hits
    on every push-to-talk release."""
    recording = tmp_path / "rec.wav"
    recording.write_bytes(_wav_bytes())

    recorder = MagicMock()
    recorder.is_recording = True
    recorder.stop = AsyncMock(return_value=recording)
    recorder.last_duration_seconds = 3.0
    app.dependency_overrides[get_recorder] = lambda: recorder

    monkeypatch.setattr(Path, "unlink", _refuse_to_unlink)

    with (
        patch("app.pipeline.router.process_audio", AsyncMock(return_value=_fake_result())),
        caplog.at_level(logging.WARNING, logger="app.pipeline.router"),
    ):
        resp = await client.post("/pipeline/dictate")

    assert resp.status_code == 200
    assert resp.json()["text"] == "hello"
    assert [r for r in caplog.records if r.name == "app.pipeline.router" and r.exc_info]


@pytest.mark.anyio
async def test_process_file_removes_its_scratch_file_on_the_success_path(client):
    """The guard must not turn the delete into a no-op — the temp directory
    still empties after a successful upload."""
    mock_process_audio = AsyncMock(return_value=_fake_result())

    with patch("app.pipeline.router.process_audio", mock_process_audio):
        resp = await client.post(
            "/pipeline/process-file",
            files={"file": ("speech.wav", _wav_bytes(), "audio/wav")},
        )

    assert resp.status_code == 200
    scratch_path = mock_process_audio.call_args.args[0]
    assert not scratch_path.exists()


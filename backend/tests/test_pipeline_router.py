"""Pipeline router — /pipeline/process-file's ``language`` query-param
default and its upload-content validation.

Dedicated router-test file, split from the service-level `test_pipeline.py`
the same way `test_preferences_router.py` is split from `test_user_settings.py`
(spec 019). `app.pipeline.router.process_audio` is patched so no real STT
call, history write, or clipboard access happens — this file only asserts
what the router forwards to `process_audio`.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.audio.dependencies import get_recorder
from app.audio.session import SessionMismatchError
from app.core.errors import ConfigurationError, NotReadyError, ResourceUnavailableError
from app.main import app
from app.pipeline.router import DictateResponse
from app.pipeline.service import ProcessingResult


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
    reached the widget as a bare 500, which `src/widget/error-label.ts` renders
    as "Failed".
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
_DICTATE_SESSION_ID = "0123456789abcdef0123456789abcdef"


def _stopping_recorder(recording: Path) -> MagicMock:
    recorder = MagicMock()
    recorder.is_recording = True
    recorder.stop = AsyncMock(return_value=recording)
    recorder.last_duration_seconds = 3.0
    return recorder


@pytest.mark.anyio
async def test_dictate_forwards_the_session_id_to_the_stop_it_opens_with(
    client, tmp_path, monkeypatch
):
    """The guard has to reach the recorder to be a guard at all.

    `recorder.stop()` decides ownership inside its own lock, so a router that
    validated the body and then called `stop()` with nothing would answer 200
    for a stranger — the race this spec closes, moved one layer up.
    """
    recording = tmp_path / "rec.wav"
    recording.write_bytes(_wav_bytes())
    recorder = _stopping_recorder(recording)
    app.dependency_overrides[get_recorder] = lambda: recorder
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)

    with patch("app.pipeline.router.process_audio", AsyncMock(return_value=_fake_result())):
        resp = await client.post("/pipeline/dictate", json={"session_id": _DICTATE_SESSION_ID})

    assert resp.status_code == 200
    recorder.stop.assert_awaited_once_with(_DICTATE_SESSION_ID)


@pytest.mark.anyio
async def test_dictate_without_a_session_id_stops_unconditionally(client, tmp_path, monkeypatch):
    """Spec 119 AC 5 on this endpoint: no body means the pre-spec-119 call.

    `smoke_sidecar.py` and any curl caller drive the dictation path with no
    body at all, and a `None` reaching `stop()` is what keeps their capture
    stoppable.
    """
    recording = tmp_path / "rec.wav"
    recording.write_bytes(_wav_bytes())
    recorder = _stopping_recorder(recording)
    app.dependency_overrides[get_recorder] = lambda: recorder
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)

    with patch("app.pipeline.router.process_audio", AsyncMock(return_value=_fake_result())):
        resp = await client.post("/pipeline/dictate")

    assert resp.status_code == 200
    recorder.stop.assert_awaited_once_with(None)


@pytest.mark.anyio
async def test_a_refused_dictate_transcribes_nothing(client, tmp_path):
    """A 403 has to arrive before the pipeline runs, not after it.

    Transcribing somebody else's capture and then refusing to answer would
    still have copied it to the clipboard and written a History row — the
    zero-leak posture broken by an endpoint that was only trying to be safe.
    """
    recorder = _stopping_recorder(tmp_path / "rec.wav")
    recorder.stop = AsyncMock(side_effect=SessionMismatchError("not yours"))
    app.dependency_overrides[get_recorder] = lambda: recorder
    mock_process_audio = AsyncMock(return_value=_fake_result())

    with patch("app.pipeline.router.process_audio", mock_process_audio):
        resp = await client.post("/pipeline/dictate", json={"session_id": _DICTATE_SESSION_ID})

    assert resp.status_code == 403
    mock_process_audio.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("refusal", "expected_status", "expected_code"),
    [
        (ConfigurationError, 400, "configuration_error"),
        (ResourceUnavailableError, 503, "resource_unavailable"),
        (NotReadyError, 409, "not_ready"),
    ],
)
async def test_dictate_lets_a_refusal_answer_with_its_own_status(
    client, tmp_path, monkeypatch, refusal, expected_status, expected_code
):
    """A classified refusal reaches the wire as itself, not as a 500.

    Without the ``except JustSayError: raise`` above ``dictate``'s
    ``except Exception``, every one of these arrives as a 500 whose ``detail``
    reads ``Pipeline failed: <ClassName>: <message>`` — the class-name leak this
    spec exists to remove, and the reason a provider-level migration in another
    package would change nothing a user sees at this endpoint.
    """
    recording = tmp_path / "rec.wav"
    recording.write_bytes(_wav_bytes())
    recorder = _stopping_recorder(recording)
    app.dependency_overrides[get_recorder] = lambda: recorder
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)

    with patch(
        "app.pipeline.router.process_audio",
        AsyncMock(side_effect=refusal("the provider said no")),
    ):
        resp = await client.post("/pipeline/dictate")

    assert resp.status_code == expected_status
    assert resp.json() == {"detail": "the provider said no", "code": expected_code}


@pytest.mark.anyio
async def test_process_file_lets_a_refusal_answer_with_its_own_status(client, monkeypatch):
    """The uploaded-file path carries the same wrapper and needs the same guard."""
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)

    with patch(
        "app.pipeline.router.process_audio",
        AsyncMock(side_effect=ResourceUnavailableError("the local engine is not up")),
    ):
        resp = await client.post(
            "/pipeline/process-file",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        )

    assert resp.status_code == 503
    assert resp.json() == {"detail": "the local engine is not up", "code": "resource_unavailable"}


def test_the_dictate_response_carries_every_processing_result_field() -> None:
    """The wire shape and the domain model must list the same field names.

    ``process_file`` and ``dictate`` both build the response as
    ``DictateResponse(**result.__dict__)``, and ``DictateResponse.model_config``
    is empty, so pydantic 2's default ``extra="ignore"`` applies: a field added
    to ProcessingResult alone is dropped on the wire with no error raised
    anywhere. The dead-code gate does not catch it either -- two of the six
    names are already in vulture's ignore_names.

    Mutation-checked twice, each applied alone: a seventh field added to
    ProcessingResult fails this test naming that field; a seventh field added
    to DictateResponse fails it the same way.
    """
    domain_fields = {field.name for field in dataclasses.fields(ProcessingResult)}
    wire_fields = set(DictateResponse.model_fields)

    assert domain_fields == wire_fields, (
        "ProcessingResult and DictateResponse declare different fields, so a value "
        "would be dropped on the wire: only in ProcessingResult: "
        f"{sorted(domain_fields - wire_fields)}; only in DictateResponse: "
        f"{sorted(wire_fields - domain_fields)}"
    )

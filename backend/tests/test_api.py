from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.service import ProcessingResult


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"]  # non-empty
    assert data["stt_mode"] in ("cloud", "local")
    assert data["llm_mode"] in ("cloud", "local")


@pytest.mark.asyncio
async def test_config(client):
    resp = await client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "stt_mode" in data
    assert "llm_mode" in data
    assert "stt_model" in data
    assert "llm_model" in data


@pytest.mark.asyncio
async def test_switch_stt_mode(client):
    resp = await client.put("/stt/mode", json={"mode": "local"})
    assert resp.status_code == 200
    assert resp.json()["stt_mode"] == "local"

    resp = await client.put("/stt/mode", json={"mode": "cloud"})
    assert resp.status_code == 200
    assert resp.json()["stt_mode"] == "cloud"


@pytest.mark.asyncio
async def test_switch_llm_mode(client):
    resp = await client.put("/llm/mode", json={"mode": "local"})
    assert resp.status_code == 200
    assert resp.json()["llm_mode"] == "local"

    resp = await client.put("/llm/mode", json={"mode": "cloud"})
    assert resp.status_code == 200
    assert resp.json()["llm_mode"] == "cloud"


@pytest.mark.asyncio
async def test_switch_stt_mode_invalid(client):
    resp = await client.put("/stt/mode", json={"mode": "quantum"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_switch_llm_mode_invalid(client):
    resp = await client.put("/llm/mode", json={"mode": "quantum"})
    assert resp.status_code == 422


def _make_recorder_mock(duration: float, audio_path: Path) -> MagicMock:
    recorder = MagicMock()
    recorder.is_recording = True
    recorder.stop = AsyncMock(return_value=audio_path)
    recorder.last_duration_seconds = duration
    return recorder


def _make_pipeline_result() -> ProcessingResult:
    return ProcessingResult(
        raw_text="ok", cleaned_text="ok", duration_ms=100, copied_to_clipboard=True
    )


@pytest.mark.asyncio
async def test_dictate_zero_duration_passes_none_to_pipeline(client, tmp_path):
    """0.0s captured_duration must not be forwarded — pipeline should re-detect."""
    audio_file = tmp_path / "rec.wav"
    audio_file.write_bytes(b"")

    recorder = _make_recorder_mock(duration=0.0, audio_path=audio_file)

    with patch("app.pipeline.router.get_recorder", return_value=recorder), patch(
        "app.pipeline.router.process_audio", new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = _make_pipeline_result()
        await client.post("/pipeline/dictate")

    assert mock_process.call_args.kwargs["audio_duration"] is None


@pytest.mark.asyncio
async def test_dictate_positive_duration_forwarded_to_pipeline(client, tmp_path):
    """Positive captured_duration must be forwarded as-is to avoid re-detection."""
    audio_file = tmp_path / "rec.wav"
    audio_file.write_bytes(b"")

    recorder = _make_recorder_mock(duration=7.5, audio_path=audio_file)

    with patch("app.pipeline.router.get_recorder", return_value=recorder), patch(
        "app.pipeline.router.process_audio", new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = _make_pipeline_result()
        await client.post("/pipeline/dictate")

    assert mock_process.call_args.kwargs["audio_duration"] == 7.5

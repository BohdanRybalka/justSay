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
async def test_set_stt_mode_accepts_json_object(client):
    """Wire format ``{"mode": "..."}`` must keep working after ProviderModeUpdate removal."""
    resp = await client.put("/stt/mode", json={"mode": "local"})
    assert resp.status_code == 200
    assert resp.json()["stt_mode"] == "local"

    resp = await client.put("/stt/mode", json={"mode": "cloud"})
    assert resp.status_code == 200
    assert resp.json()["stt_mode"] == "cloud"


@pytest.mark.asyncio
async def test_set_llm_mode_accepts_json_object(client):
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


# --- /settings/storage path masking (Tech Debt batch v0.8.2) -------

def test_mask_home_masks_paths_under_home(tmp_path, monkeypatch):
    """Unit test for the `_mask_home` function — paths under `Path.home()`
    come back as `~/...`."""
    from pathlib import Path

    from app.core.settings_router import _mask_home

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    masked = _mask_home(fake_home / ".justsay" / "history.db")
    assert masked.startswith("~/"), masked
    assert "history.db" in masked


def test_mask_home_passes_through_paths_outside_home(tmp_path, monkeypatch):
    """Network drives / external SSDs are returned as-is.

    Documents the privacy hygiene's boundary — see release notes v0.8.2.
    If users point `output_dir` at a Dropbox share or NAS, we'd rather show
    the real path than lie with `~/...`.
    """
    from pathlib import Path

    from app.core.settings_router import _mask_home

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    external = tmp_path / "external" / "data"
    external.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    assert _mask_home(external) == str(external)


def test_mask_home_degrades_gracefully_when_home_unavailable(tmp_path, monkeypatch):
    """`Path.home()` can raise `RuntimeError` in sandboxes / containers
    without `HOME`/`USERPROFILE` — endpoint must not 500."""
    from pathlib import Path

    from app.core.settings_router import _mask_home

    def _raise(cls):
        raise RuntimeError("no home")

    monkeypatch.setattr(Path, "home", classmethod(_raise))
    assert _mask_home(tmp_path / "foo") == str(tmp_path / "foo")


@pytest.mark.asyncio
async def test_storage_info_routes_output_dir_through_mask(client, tmp_path, monkeypatch):
    """The router actually calls `_mask_home` for `output_dir`.

    Sets `output_dir` under a fake home so we can assert the round-trip
    masking without depending on the dev machine's real `~/.justsay`
    contents. The unit tests above cover the masking logic itself.
    """
    from pathlib import Path

    from app.core import user_settings as us

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_output = fake_home / "JustSayData"
    fake_output.mkdir()
    cached = us.UserSettings(output_dir=str(fake_output))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(us, "_settings", cached)

    resp = await client.get("/settings/storage")
    assert resp.status_code == 200
    data = resp.json()
    # The headline assertion: output_dir is masked → renderer never sees
    # the absolute home path on this code path.
    assert data["output_dir"].startswith("~"), data["output_dir"]
    assert "JustSayData" in data["output_dir"]
    # Raw home prefix MUST NOT appear in the output_dir string.
    assert str(fake_home).replace("\\", "/") not in data["output_dir"]


# --- /stt/transcribe content validation -----------------------------

@pytest.mark.asyncio
async def test_transcribe_rejects_extension_content_mismatch(client):
    """`.wav` filename with non-WAV bytes is rejected at the validator boundary,
    not handed off to the STT provider where it would 500 deep inside soundfile."""
    fake_payload = b"MZ" + (b"\x00" * 64)  # DOS / PE prefix
    resp = await client.post(
        "/stt/transcribe",
        files={"file": ("evil.wav", fake_payload, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "does not match" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_file(client):
    resp = await client.post(
        "/stt/transcribe",
        files={"file": ("speech.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400
    assert "too small" in resp.json()["detail"].lower()


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
        text="ok", duration_ms=100, copied_to_clipboard=True
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

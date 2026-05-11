"""Settings router — API key masking and cloud-status endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import user_settings
from app.core.config import settings as runtime_settings
from app.main import app


@pytest.fixture
async def client(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    settings_dir = home / ".justsay"
    settings_dir.mkdir()

    from pathlib import Path
    from app.core import history

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(user_settings, "SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(user_settings, "SETTINGS_PATH", settings_dir / "settings.json")
    monkeypatch.setattr(user_settings, "_settings", None)
    monkeypatch.setattr(history, "_output_dir", settings_dir)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    from app.core.history import _lock, _close_conn_locked
    with _lock:
        _close_conn_locked()


# --- GET /settings masks keys -----------------------------------------------

@pytest.mark.anyio
async def test_get_settings_masks_stored_key(client, monkeypatch):
    monkeypatch.setattr(user_settings, "_settings", None)
    user_settings.update_user_settings({"gemini_api_key": "AIza-real-key"})

    resp = await client.get("/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["gemini_api_key"] == "***"
    assert "AIza-real-key" not in str(data)


@pytest.mark.anyio
async def test_get_settings_returns_empty_string_when_no_key(client):
    resp = await client.get("/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["gemini_api_key"] == ""
    assert data["groq_api_key"] == ""


# --- PUT /settings masks response and guards placeholder --------------------

@pytest.mark.anyio
async def test_put_settings_response_masks_key(client):
    resp = await client.put("/settings", json={"gemini_api_key": "AIza-new"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["settings"]["gemini_api_key"] == "***"
    assert "AIza-new" not in str(data)


@pytest.mark.anyio
async def test_put_settings_placeholder_does_not_overwrite(client, monkeypatch):
    monkeypatch.setattr(user_settings, "_settings", None)
    user_settings.update_user_settings({"gemini_api_key": "AIza-original"})

    resp = await client.put("/settings", json={"gemini_api_key": "***"})
    assert resp.status_code == 200

    stored = user_settings.get_user_settings()
    assert stored.gemini_api_key == "AIza-original"


# --- GET /settings/cloud-status all four combinations -----------------------

@pytest.fixture(autouse=True)
def _reset_runtime_keys():
    """Ensure runtime key fields are blank before each test to prevent cross-test pollution."""
    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.llm.groq_api_key = ""
    yield
    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.llm.groq_api_key = ""


@pytest.mark.anyio
async def test_cloud_status_both_empty(client):
    resp = await client.get("/settings/cloud-status")
    assert resp.status_code == 200
    assert resp.json() == {"gemini_key_set": False, "groq_key_set": False}


@pytest.mark.anyio
async def test_cloud_status_both_set(client):
    runtime_settings.stt.gemini_api_key = "AIza-x"
    runtime_settings.stt.groq_api_key = "gsk-x"

    resp = await client.get("/settings/cloud-status")
    assert resp.status_code == 200
    assert resp.json() == {"gemini_key_set": True, "groq_key_set": True}


@pytest.mark.anyio
async def test_cloud_status_gemini_only(client):
    runtime_settings.stt.gemini_api_key = "AIza-x"

    resp = await client.get("/settings/cloud-status")
    assert resp.status_code == 200
    assert resp.json() == {"gemini_key_set": True, "groq_key_set": False}


@pytest.mark.anyio
async def test_cloud_status_groq_only(client):
    runtime_settings.stt.groq_api_key = "gsk-x"

    resp = await client.get("/settings/cloud-status")
    assert resp.status_code == 200
    assert resp.json() == {"gemini_key_set": False, "groq_key_set": True}

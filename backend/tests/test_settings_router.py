"""Settings router — API key masking and cloud-status endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings as runtime_settings
from app.main import app
from app.preferences import user_settings


@pytest.fixture
async def client(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    settings_dir = home / ".justsay"
    settings_dir.mkdir()

    from app.transcripts import history

    monkeypatch.setenv("JUSTSAY_DATA_DIR", str(settings_dir))
    monkeypatch.setattr(user_settings, "_settings", None)
    monkeypatch.setattr(history, "_output_dir", settings_dir)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    from app.transcripts.history import _close_conn_locked, _lock
    with _lock:
        _close_conn_locked()



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




def _counting_spy(counter: dict):
    """monkeypatch-able stand-in for maybe_prewarm_local that just counts calls."""
    def _spy(stt_settings):
        counter["n"] += 1
    return _spy


@pytest.mark.prewarm
@pytest.mark.anyio
async def test_put_settings_triggers_prewarm_when_switching_to_local(client, monkeypatch):
    """`put_settings()` calls `maybe_prewarm_local(runtime_settings.stt)`
    after `sync_to_runtime(...)`. Marked `@pytest.mark.prewarm` so
    conftest's autouse no-op fixture doesn't mask this — `maybe_prewarm_local`
    is patched to a counting spy so the real pip-install/model-load path
    never runs."""
    import app.stt.local_setup as local_setup_module

    call_count = {"n": 0}
    monkeypatch.setattr(local_setup_module, "maybe_prewarm_local", _counting_spy(call_count))

    resp = await client.put("/settings", json={"stt_mode": "local"})
    assert resp.status_code == 200
    assert call_count["n"] == 1


@pytest.mark.prewarm
@pytest.mark.anyio
async def test_put_settings_triggers_prewarm_on_incidental_cache_clear(client, monkeypatch):
    """An unrelated settings change (e.g. a glossary edit) that incidentally
    clears the STT cache via `sync_to_runtime`'s `changed_stt` check must
    still re-trigger `maybe_prewarm_local` while Local stays the active mode
    — otherwise that edit would silently reintroduce a cold lazy-load."""
    import app.stt.local_setup as local_setup_module

    monkeypatch.setattr(runtime_settings.stt, "initial_prompt", runtime_settings.stt.initial_prompt)

    call_count = {"n": 0}
    monkeypatch.setattr(local_setup_module, "maybe_prewarm_local", _counting_spy(call_count))

    resp = await client.put("/settings", json={"stt_mode": "local"})
    assert resp.status_code == 200
    assert call_count["n"] == 1

    resp = await client.put("/settings", json={"initial_prompt": "Tauri FastAPI Pydantic"})
    assert resp.status_code == 200
    assert call_count["n"] == 2


@pytest.mark.prewarm
@pytest.mark.anyio
async def test_put_settings_does_not_prewarm_on_non_stt_field_change(client, monkeypatch):
    """A settings edit that has nothing to do with STT (e.g. `shortcut`) must
    not call `maybe_prewarm_local` at all while Local is active -- spec 024's
    fix for the previously-unconditional call gating it on
    `sync_to_runtime`'s own `changed_stt` return value instead."""
    import app.stt.local_setup as local_setup_module

    monkeypatch.setattr(runtime_settings.stt, "initial_prompt", runtime_settings.stt.initial_prompt)

    call_count = {"n": 0}
    monkeypatch.setattr(local_setup_module, "maybe_prewarm_local", _counting_spy(call_count))

    resp = await client.put("/settings", json={"stt_mode": "local"})
    assert resp.status_code == 200
    assert call_count["n"] == 1

    resp = await client.put("/settings", json={"shortcut": "Ctrl+Alt+KeyB"})
    assert resp.status_code == 200
    assert call_count["n"] == 1

    resp = await client.put("/settings", json={"initial_prompt": "Tauri FastAPI Pydantic"})
    assert resp.status_code == 200
    assert call_count["n"] == 2




@pytest.fixture
def isolated_temp_dir(tmp_path, monkeypatch):
    temp_dir = tmp_path / "audio-tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(runtime_settings.audio, "temp_dir", temp_dir)
    return temp_dir


@pytest.mark.anyio
async def test_storage_reports_size_of_configured_temp_dir(client, isolated_temp_dir):
    (isolated_temp_dir / "rec_abc123.wav").write_bytes(b"x" * 1234)

    resp = await client.get("/settings/storage")
    assert resp.status_code == 200
    assert resp.json()["temp_size_bytes"] == 1234


@pytest.mark.anyio
async def test_cleanup_removes_files_from_configured_temp_dir(client, isolated_temp_dir):
    (isolated_temp_dir / "rec_abc123.wav").write_bytes(b"x" * 1234)
    (isolated_temp_dir / "pipeline_def456.m4a").write_bytes(b"y" * 766)

    resp = await client.post("/settings/cleanup")
    assert resp.status_code == 200
    assert resp.json()["freed_bytes"] == 2000

    assert isolated_temp_dir.exists()
    assert list(isolated_temp_dir.iterdir()) == []


@pytest.mark.anyio
async def test_cleanup_never_deletes_a_history_database_it_finds(client, isolated_temp_dir):
    """The defect this endpoint shipped with: `output_dir` pointed at the
    scratch directory, so `shutil.rmtree` took 89 real transcripts with it.
    Deletion is now scoped by ownership, so anything the app did not write
    survives by definition rather than by being on an exception list.
    """
    history_db = isolated_temp_dir / "history.db"
    history_db.write_bytes(b"SQLite format 3\x00" + b"z" * 500)
    unrelated = isolated_temp_dir / "notes.txt"
    unrelated.write_bytes(b"keep me")
    (isolated_temp_dir / "rec_abc123.wav").write_bytes(b"x" * 1234)

    resp = await client.post("/settings/cleanup")

    assert resp.status_code == 200
    assert resp.json()["freed_bytes"] == 1234
    assert history_db.exists()
    assert unrelated.exists()
    assert not (isolated_temp_dir / "rec_abc123.wav").exists()


@pytest.mark.anyio
async def test_a_meeting_wav_is_counted_and_cleaned_like_any_other_scratch_file(
    client, isolated_temp_dir
):
    """Spec 066 added a third producer into the scratch directory, and ADR 033
    scopes deletion by ownership — so the meeting recorder registers its own
    prefix rather than being caught by a location rule. A file matching no
    prefix still survives both.
    """
    (isolated_temp_dir / "meeting_abc123.wav").write_bytes(b"m" * 4321)
    history_db = isolated_temp_dir / "history.db"
    history_db.write_bytes(b"SQLite format 3\x00" + b"z" * 500)

    assert (await client.get("/settings/storage")).json()["temp_size_bytes"] == 4321

    resp = await client.post("/settings/cleanup")

    assert resp.status_code == 200
    assert resp.json()["freed_bytes"] == 4321
    assert not (isolated_temp_dir / "meeting_abc123.wav").exists()
    assert history_db.exists()


@pytest.mark.anyio
async def test_reported_size_equals_bytes_cleanup_frees(client, isolated_temp_dir):
    """Shown and freed come from one helper, so a foreign file cannot inflate
    the number the user is asked to act on."""
    (isolated_temp_dir / "rec_abc123.wav").write_bytes(b"x" * 1234)
    (isolated_temp_dir / "history.db").write_bytes(b"z" * 99999)

    reported = (await client.get("/settings/storage")).json()["temp_size_bytes"]
    freed = (await client.post("/settings/cleanup")).json()["freed_bytes"]

    assert reported == freed == 1234




@pytest.mark.anyio
@pytest.mark.parametrize(
    "bad_value",
    ["../evil", "..\\evil", "foo/bar", "foo\\bar", "..", "a b", "large-v3;rm", "large-v3\n"],
)
async def test_put_settings_rejects_unsafe_whisper_model_size(client, bad_value):
    resp = await client.put("/settings", json={"whisper_model_size": bad_value})
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_put_settings_accepts_valid_whisper_model_size(client):
    resp = await client.put("/settings", json={"whisper_model_size": "large-v3-turbo"})
    assert resp.status_code == 200
    assert user_settings.get_user_settings().whisper_model_size == "large-v3-turbo"

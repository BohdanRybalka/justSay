"""Scratch-directory endpoints — the size reported and the bytes reaped.

Deletion is scoped by ownership rather than by location (ADR 033), so a file
the app did not write is counted by neither and survives both.
"""

from __future__ import annotations

import pytest

from app.audio.config import audio_settings


@pytest.fixture
def isolated_temp_dir(tmp_path, monkeypatch):
    temp_dir = tmp_path / "audio-tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(audio_settings, "temp_dir", temp_dir)
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

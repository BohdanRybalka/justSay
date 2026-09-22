"""Scratch-directory endpoints — the size reported and the bytes reaped.

Deletion is scoped by ownership rather than by location (ADR 033), so a file
the app did not write is counted by neither and survives both.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.audio import scratch_router
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


@pytest.mark.anyio
async def test_both_endpoints_walk_the_directory_off_the_event_loop_thread(
    client, isolated_temp_dir, monkeypatch
):
    """A scratch directory is walked on a worker thread, not the loop's own.

    Each helper records the thread it ran on. The recorded id differing from the
    loop's is what fails when a handler stops awaiting `asyncio.to_thread` and
    calls straight through, which would stall every other request for the walk.
    """
    (isolated_temp_dir / "rec_abc123.wav").write_bytes(b"x" * 16)
    ran_on: dict[str, int] = {}

    for name in ("_scratch_size", "_reap_scratch_files"):
        original = getattr(scratch_router, name)

        def _record(tmp_dir, _name=name, _original=original):
            ran_on[_name] = threading.get_ident()
            return _original(tmp_dir)

        monkeypatch.setattr(scratch_router, name, _record)

    loop_thread = threading.get_ident()
    assert (await client.get("/settings/storage")).status_code == 200
    assert (await client.post("/settings/cleanup")).status_code == 200

    assert set(ran_on) == {"_scratch_size", "_reap_scratch_files"}, (
        f"a handler never reached its helper: {sorted(ran_on)}"
    )
    assert threading.get_ident() == loop_thread
    for name, thread_id in ran_on.items():
        assert thread_id != loop_thread, (
            f"{name} ran on the event loop's own thread, so the directory walk "
            "blocks every other request for its duration"
        )


@pytest.mark.anyio
async def test_asyncio_is_what_moves_the_walk_rather_than_the_helpers_themselves():
    """The handlers own the thread hop; the helpers stay plain and synchronous.

    Pins the seam so a later edit cannot satisfy the test above by making a
    helper spawn its own thread while the handler still blocks.
    """
    assert not asyncio.iscoroutinefunction(scratch_router._scratch_size)
    assert not asyncio.iscoroutinefunction(scratch_router._reap_scratch_files)
    assert asyncio.iscoroutinefunction(scratch_router.get_storage_info)
    assert asyncio.iscoroutinefunction(scratch_router.cleanup_temp)

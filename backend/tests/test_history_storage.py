"""Round-trip + migration + relocate + concurrency + validation tests.

Adapted from the JSONL-era v0.5.2 tests: every assertion is preserved per the
QA contract; ``.jsonl`` path checks translate to ``.db`` checks; round-trip
checks gain a SQL ``SELECT COUNT(*)`` parallel; relocate-failure tests mock
the SQLite copy-verify failure path instead of the JSONL one.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import history, user_settings


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Pin every storage location into a tmp tree so we never touch ~/.justsay."""
    home = tmp_path / "home"
    home.mkdir()

    legacy_dir = home / ".justsay"
    legacy_dir.mkdir()
    legacy_path = legacy_dir / "history.jsonl"

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(history, "LEGACY_DIR", legacy_dir)
    monkeypatch.setattr(history, "LEGACY_PATH", legacy_path)
    monkeypatch.setattr(history, "_output_dir", legacy_dir)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)
    monkeypatch.setattr(user_settings, "SETTINGS_DIR", legacy_dir)
    monkeypatch.setattr(user_settings, "SETTINGS_PATH", legacy_dir / "settings.json")
    monkeypatch.setattr(user_settings, "_settings", None)

    yield {"home": home, "legacy_dir": legacy_dir, "legacy_path": legacy_path}

    # Close any open connection so the tmp tree can be deleted on Windows.
    with history._lock:
        history._close_conn_locked()


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    finally:
        conn.close()


def test_round_trip_default_path(isolated_storage):
    history.save_entry(raw_text="hi", cleaned_text="hi", duration_ms=10)
    assert history.history_path() == isolated_storage["legacy_dir"] / "history.db"
    assert history.history_path().exists()
    assert len(history.get_entries()) == 1
    # SQL parallel: assert the row count via a fresh connection too.
    assert _row_count(history.history_path()) == 1


def test_round_trip_custom_path(isolated_storage, tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    history.init_output_dir(custom)
    history.save_entry(raw_text="hi", cleaned_text="hi", duration_ms=10)
    assert (custom / "history.db").exists()
    assert _row_count(custom / "history.db") == 1
    # legacy untouched
    assert not isolated_storage["legacy_path"].exists()


def test_migration_moves_legacy(isolated_storage, tmp_path):
    legacy = isolated_storage["legacy_path"]
    legacy.write_text(
        '{"id":"a","timestamp":"2026-01-01T00:00:00+00:00","language":"uk","style":"normal","raw_text":"x","cleaned_text":"x","duration_ms":1}\n',
        encoding="utf-8",
    )

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    moved = history.migrate_legacy_if_needed(target_dir)

    assert moved is True
    assert (target_dir / "history.jsonl").exists()
    assert not legacy.exists()
    assert legacy.with_suffix(".jsonl.bak").exists()


def test_migration_collision_keeps_legacy(isolated_storage, tmp_path):
    legacy = isolated_storage["legacy_path"]
    legacy.write_text("legacy-line\n", encoding="utf-8")

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target_file = target_dir / "history.jsonl"
    target_file.write_text("target-line\n", encoding="utf-8")

    moved = history.migrate_legacy_if_needed(target_dir)
    assert moved is False
    assert legacy.read_text(encoding="utf-8") == "legacy-line\n"
    assert target_file.read_text(encoding="utf-8") == "target-line\n"


def test_migration_noop_when_no_legacy(isolated_storage, tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    moved = history.migrate_legacy_if_needed(target_dir)
    assert moved is False


def test_relocate_moves_file_and_updates_path(isolated_storage, tmp_path):
    history.save_entry(raw_text="hi", cleaned_text="hi", duration_ms=10)
    old_path = history.history_path()
    assert old_path.exists()

    new_dir = tmp_path / "new"
    result, reason = history.relocate(new_dir)

    assert result == history.RelocateResult.MOVED
    assert reason is None
    assert history.history_path() == new_dir / "history.db"
    assert (new_dir / "history.db").exists()
    assert not old_path.exists()
    assert len(history.get_entries()) == 1


def test_relocate_collision_keeps_target(isolated_storage, tmp_path):
    history.save_entry(raw_text="hi", cleaned_text="hi", duration_ms=10)
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    target = new_dir / "history.db"
    # Pre-populate with a real DB so SQLite can open it after the relocate.
    pre_conn = history._connect(target)
    history._init_schema(pre_conn)
    pre_conn.close()

    result, _ = history.relocate(new_dir)
    assert result == history.RelocateResult.NEW_ALREADY_HAS_FILE
    # Target still exists and is a valid DB
    assert target.exists()
    assert _row_count(target) == 0


def test_relocate_failure_rolls_back(isolated_storage, tmp_path, monkeypatch):
    history.save_entry(raw_text="hi", cleaned_text="hi", duration_ms=10)
    old_path = history.history_path()
    new_dir = tmp_path / "new"

    def fail_verify(_src, _dst):
        return False
    monkeypatch.setattr(history, "_verify_db_row_count", fail_verify)

    result, reason = history.relocate(new_dir)
    assert result == history.RelocateResult.FAILED
    assert "Verification" in (reason or "")
    assert old_path.exists()  # still there
    assert not (new_dir / "history.db").exists()


def test_validation_rejects_relative(isolated_storage):
    with pytest.raises(ValueError):
        user_settings._validate_output_dir("relative/path")


def test_validation_rejects_empty(isolated_storage):
    with pytest.raises(ValueError):
        user_settings._validate_output_dir("")
    with pytest.raises(ValueError):
        user_settings._validate_output_dir("   ")


def test_validation_rejects_non_string(isolated_storage):
    with pytest.raises(ValueError):
        user_settings._validate_output_dir(None)
    with pytest.raises(ValueError):
        user_settings._validate_output_dir(123)


def test_validation_accepts_valid_writable(isolated_storage, tmp_path):
    candidate = tmp_path / "valid"
    candidate.mkdir()
    result = user_settings._validate_output_dir(str(candidate))
    assert result == candidate.resolve()
    # No probe leak
    assert not any(p.name.startswith(".justsay-write-probe") for p in candidate.iterdir())


def test_concurrent_saves_serialise(isolated_storage):
    """10 concurrent save_entry calls must all land in the DB with no loss."""
    errors = []

    def worker(i):
        try:
            history.save_entry(raw_text=f"e{i}", cleaned_text=f"e{i}", duration_ms=i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert errors == []
    entries = history.get_entries(limit=100)
    assert len(entries) == 10
    # SQL parallel
    assert _row_count(history.history_path()) == 10


def test_storage_info_masks_home(isolated_storage):
    from app.main import app

    history.save_entry(raw_text="hi", cleaned_text="hi", duration_ms=1)
    with TestClient(app) as client:
        resp = client.get("/settings/storage")
        assert resp.status_code == 200
        data = resp.json()
        # Path should be masked with ~ rather than the absolute home prefix
        assert data["history_path"].startswith("~")
        assert str(isolated_storage["home"]) not in data["history_path"]
        assert data["history_entries"] >= 1


def test_put_settings_returns_warning_on_collision(isolated_storage, tmp_path):
    from app.main import app

    history.save_entry(raw_text="hi", cleaned_text="hi", duration_ms=1)

    new_dir = tmp_path / "new_with_existing"
    new_dir.mkdir()
    # Pre-populate with a valid SQLite file.
    pre_conn = history._connect(new_dir / "history.db")
    history._init_schema(pre_conn)
    pre_conn.close()

    with TestClient(app) as client:
        resp = client.put("/settings", json={"output_dir": str(new_dir)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["warning"] is not None
        assert "preserved" in body["warning"].lower()


def test_put_settings_400_on_bad_path(isolated_storage):
    from app.main import app
    with TestClient(app) as client:
        resp = client.put("/settings", json={"output_dir": "relative/bad"})
        assert resp.status_code == 400


def test_put_settings_does_not_persist_on_failure(isolated_storage, tmp_path, monkeypatch):
    """If relocate FAILS, settings.json must not be written with the new value."""
    from app.main import app

    history.save_entry(raw_text="hi", cleaned_text="hi", duration_ms=1)

    # Force settings.json to exist with a known value first.
    user_settings._save(user_settings.get_user_settings())
    original_dir = user_settings.get_user_settings().output_dir

    new_dir = tmp_path / "doomed"
    new_dir.mkdir()

    def fail_verify(_src, _dst):
        return False
    monkeypatch.setattr(history, "_verify_db_row_count", fail_verify)

    with TestClient(app) as client:
        resp = client.put("/settings", json={"output_dir": str(new_dir)})
        assert resp.status_code == 500

    # Settings on disk + in memory unchanged.
    user_settings._settings = None  # force re-read from disk
    assert user_settings.get_user_settings().output_dir == original_dir
    on_disk = json.loads(user_settings.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert on_disk["output_dir"] == original_dir

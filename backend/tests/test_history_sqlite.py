"""SQLite-specific tests for the v1 history store.

Covers schema/PRAGMA, stats cache TTL+invalidation, explicit transactions,
ISO ↔ epoch ms round-trip, ``OperationalError`` → 503 mapping at the router,
relocate branches, concurrent saves.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core import history


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    yield {"home": home, "tmp_path": tmp_path}

    with history._lock:
        history._close_conn_locked()


# --- Schema / PRAGMA -----------------------------------------------------

def test_user_version_set(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    db = target / "history.db"
    conn = sqlite3.connect(db)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert version == history.SCHEMA_VERSION


def test_pragmas_set_in_factory(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert jm.lower() == "delete"
        # synchronous=FULL = 2
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_check_constraint_rejects_invalid_style(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    with pytest.raises(sqlite3.IntegrityError):
        history.save_entry(text="x", duration_ms=1, style="bogus")


def test_check_constraint_rejects_negative_duration(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    with pytest.raises(sqlite3.IntegrityError):
        history.save_entry(text="x", duration_ms=-1)


# --- CRUD round-trip ----------------------------------------------------

def test_save_then_get_entries(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    e = history.save_entry(text="hello world", duration_ms=100, language="uk", word_count=2)
    [out] = history.get_entries(limit=10)
    assert out.id == e.id
    assert out.text == "hello world"
    assert out.word_count == 2


def test_delete_entry_removes_row(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    e = history.save_entry(text="x", duration_ms=1)
    assert history.delete_entry(e.id) is True
    assert history.get_count() == 0


def test_delete_nonexistent_id_returns_false(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1, word_count=5)
    # Trigger stats cache so we can detect (un)invalidation.
    history.compute_stats()
    cache_before = history._stats_cache

    assert history.delete_entry("does-not-exist") is False

    # Cache must NOT have been invalidated by a no-op delete.
    assert history._stats_cache is cache_before


def test_clear_all_returns_count_and_empties(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    for _ in range(3):
        history.save_entry(text="x", duration_ms=1)
    assert history.clear_all() == 3
    assert history.get_count() == 0


# --- Stats cache TTL + invalidation -------------------------------------

def test_stats_cache_invalidated_on_save(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    assert history.compute_stats().total_entries == 0
    history.save_entry(text="x", duration_ms=1, word_count=5)
    assert history.compute_stats().total_entries == 1
    assert history.compute_stats().total_words == 5


def test_stats_cache_invalidated_on_delete(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    e = history.save_entry(text="x", duration_ms=1, word_count=5)
    assert history.compute_stats().total_entries == 1
    history.delete_entry(e.id)
    assert history.compute_stats().total_entries == 0


def test_stats_cache_invalidated_on_clear(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1, word_count=5)
    assert history.compute_stats().total_entries == 1
    history.clear_all()
    assert history.compute_stats().total_entries == 0


def test_stats_cache_invalidated_on_relocate(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1, word_count=5)
    s1 = history.compute_stats()
    assert s1.total_entries == 1

    new_dir = tmp_path / "new"
    history.relocate(new_dir)
    s2 = history.compute_stats()
    assert s2.total_entries == 1  # data moved with it


def test_stats_cache_ttl_returns_cached_value(isolated_storage, tmp_path):
    """Within 5 s of a non-mutating second call, the cached value is returned."""
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1, word_count=5)

    s1 = history.compute_stats()
    # Mutate state at the SQL layer WITHOUT going through save_entry → cache stays fresh
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms, word_count) "
            "VALUES ('zzz', 0, 'uk', 'normal', '', '', 0, 99)"
        )
    s2 = history.compute_stats()
    assert s2.total_entries == s1.total_entries  # cache served the old value


def test_compute_stats_empty_db_zero_counts(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    s = history.compute_stats()
    assert s.total_entries == 0
    assert s.total_words == 0
    assert s.total_audio_seconds == 0.0
    assert s.today_words == 0
    assert s.week_words == 0
    assert s.by_language == {}
    assert s.by_model == {}


def test_compute_stats_excludes_null_model_name(isolated_storage, tmp_path):
    """Entries with NULL model_name must NOT appear as a ``None`` key in by_model."""
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1, word_count=10, model_name=None)
    history.save_entry(text="y", duration_ms=1, word_count=20, model_name="gemini/flash")
    s = history.compute_stats()
    assert None not in s.by_model
    assert s.by_model == {"gemini/flash": 20}
    assert s.total_words == 30  # both entries counted in totals


# --- Explicit transactions ----------------------------------------------

def test_save_entry_issues_begin_and_commit(isolated_storage, tmp_path):
    """save_entry MUST issue an explicit BEGIN before INSERT and COMMIT after."""
    target = tmp_path / "target"
    history.bootstrap(target)
    statements: list[str] = []

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.isolation_level is None  # manual txn control
        conn.set_trace_callback(lambda s: statements.append(s.strip().upper()))

    history.save_entry(text="x", duration_ms=1)

    with history._lock:
        history._ensure_conn_locked().set_trace_callback(None)

    begin_idx = next(i for i, s in enumerate(statements) if s.startswith("BEGIN"))
    insert_idx = next(i for i, s in enumerate(statements) if s.startswith("INSERT"))
    commit_idx = next(i for i, s in enumerate(statements) if s.startswith("COMMIT"))
    assert begin_idx < insert_idx < commit_idx


# --- ISO ↔ epoch ms round-trip ------------------------------------------

def test_iso_to_epoch_ms_with_offset():
    iso = "2026-01-01T12:00:00+02:00"
    ms = history._iso_to_epoch_ms(iso)
    expected = int(datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert ms == expected


def test_iso_to_epoch_ms_with_z_suffix():
    """Defensive Z→+00:00 shim for Python 3.10 fromisoformat."""
    iso = "2026-01-01T10:00:00Z"
    ms = history._iso_to_epoch_ms(iso)
    expected = int(datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert ms == expected


def test_round_trip_through_db_preserves_iso(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    e_in = history.save_entry(text="x", duration_ms=1)
    [e_out] = history.get_entries(limit=10)
    t_in = datetime.fromisoformat(e_in.timestamp.replace("Z", "+00:00"))
    t_out = datetime.fromisoformat(e_out.timestamp.replace("Z", "+00:00"))
    assert abs((t_out - t_in).total_seconds()) < 0.001


# --- Relocate branches --------------------------------------------------

def test_relocate_moved_branch(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1)

    new_dir = tmp_path / "new"
    res, reason = history.relocate(new_dir)
    assert res == history.RelocateResult.MOVED
    assert reason is None
    assert (new_dir / "history.db").exists()
    assert not (target / "history.db").exists()


def test_relocate_no_old_file_branch(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    # Don't save any entries; close the conn so the file doesn't get auto-created.
    with history._lock:
        history._close_conn_locked()
    (target / "history.db").unlink(missing_ok=True)

    new_dir = tmp_path / "new"
    res, _ = history.relocate(new_dir)
    assert res == history.RelocateResult.NO_OLD_FILE


def test_relocate_new_already_has_file_branch(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="old", duration_ms=1)

    # Pre-populate the new directory with its own history.db
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    history.bootstrap(new_dir)
    history.save_entry(text="new", duration_ms=1)
    history.bootstrap(target)  # switch back to old

    res, _ = history.relocate(new_dir)
    assert res == history.RelocateResult.NEW_ALREADY_HAS_FILE
    # Existing file at new location preserved.
    assert (new_dir / "history.db").exists()


def test_relocate_failed_on_copy_oserror(isolated_storage, tmp_path, monkeypatch):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1)

    new_dir = tmp_path / "new"

    def boom(*_a, **_kw):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(history.shutil, "copy2", boom)

    res, reason = history.relocate(new_dir)
    assert res == history.RelocateResult.FAILED
    assert reason and "simulated copy failure" in reason
    # Old file still present.
    assert (target / "history.db").exists()


# --- Concurrent saves ---------------------------------------------------

def test_concurrent_saves_no_loss(isolated_storage, tmp_path):
    """10 threads × 5 saves each = 50 distinct rows, all unique IDs."""
    target = tmp_path / "target"
    history.bootstrap(target)

    errors: list[Exception] = []

    def worker(worker_id: int):
        try:
            for j in range(5):
                history.save_entry(text=f"w{worker_id}-{j}", duration_ms=1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert history.get_count() == 50
    entries = history.get_entries(limit=100)
    ids = {e.id for e in entries}
    assert len(ids) == 50  # all unique


# --- 503 mapping at router ----------------------------------------------

def test_operational_error_mapped_to_503(isolated_storage, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app

    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1)

    with TestClient(app) as client:
        with patch("app.core.history_router.compute_stats", side_effect=sqlite3.OperationalError("database is locked")):
            resp = client.get("/history/stats")
            assert resp.status_code == 503
            assert resp.headers.get("Retry-After") == "1"

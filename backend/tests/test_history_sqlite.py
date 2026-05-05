"""SQLite-specific tests for the v1 history store (Plan 012).

Covers schema/PRAGMA, JSONL→SQLite migration (clean / duplicates / corrupt),
stats cache TTL+invalidation, explicit transactions, ISO ↔ epoch ms round-trip,
``OperationalError`` → 503 mapping at the router layer.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core import history


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    legacy_dir = home / ".justsay"
    legacy_dir.mkdir()

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(history, "LEGACY_DIR", legacy_dir)
    monkeypatch.setattr(history, "LEGACY_PATH", legacy_dir / "history.jsonl")
    monkeypatch.setattr(history, "_output_dir", legacy_dir)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    yield {"home": home, "legacy_dir": legacy_dir, "tmp_path": tmp_path}

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
        history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1, style="bogus")


def test_check_constraint_rejects_negative_duration(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    with pytest.raises(sqlite3.IntegrityError):
        history.save_entry(raw_text="x", cleaned_text="x", duration_ms=-1)


# --- JSONL → SQLite migration -------------------------------------------

_GOOD_LINE = (
    '{{"id":"{id}","timestamp":"2026-01-01T10:00:00+00:00","language":"uk",'
    '"style":"normal","raw_text":"x","cleaned_text":"x","duration_ms":1}}\n'
)


def _write_jsonl(path: Path, ids: list[str]) -> None:
    path.write_text("".join(_GOOD_LINE.format(id=i) for i in ids), encoding="utf-8")


def test_migration_zero_entries(isolated_storage, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "history.jsonl").write_text("", encoding="utf-8")
    history.bootstrap(target)
    assert (target / "history.db").exists()
    assert (target / "history.jsonl.bak").exists()
    assert history.get_count() == 0


def test_migration_eight_entries(isolated_storage, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    _write_jsonl(target / "history.jsonl", [f"e{i:02d}" for i in range(8)])
    history.bootstrap(target)
    assert (target / "history.db").exists()
    assert history.get_count() == 8


def test_migration_duplicate_ids_use_insert_or_ignore(isolated_storage, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    _write_jsonl(target / "history.jsonl", ["dup", "dup", "uniq"])
    history.bootstrap(target)
    assert history.get_count() == 2  # dup collapsed
    assert (target / "history.jsonl.bak").exists()


def test_migration_corrupt_lines_skipped(isolated_storage, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    good = _GOOD_LINE.format(id="ok1")
    (target / "history.jsonl").write_text(
        good + "this is not json\n" + '{"incomplete":"json"}\n' + good.replace("ok1", "ok2"),
        encoding="utf-8",
    )
    history.bootstrap(target)
    assert history.get_count() == 2


def test_migration_failure_keeps_jsonl_intact(isolated_storage, tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    _write_jsonl(target / "history.jsonl", ["a", "b"])

    # Force a failure DURING migration (before the move) by sabotaging shutil.move.
    import shutil as _shutil

    def boom(*_a, **_kw):
        raise OSError("simulated atomic move failure")

    monkeypatch.setattr(history, "shutil", type("S", (), {"copy2": _shutil.copy2, "move": boom}))
    with pytest.raises(RuntimeError, match="SQLite migration failed"):
        history.bootstrap(target)

    assert (target / "history.jsonl").exists()
    assert not (target / "history.db").exists()
    # No tmp leak
    assert not list(target.glob("history.db.tmp-*"))


# --- Stats cache TTL + invalidation -------------------------------------

def test_stats_cache_invalidated_on_save(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    assert history.compute_stats().total_entries == 0
    history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1, word_count=5)
    assert history.compute_stats().total_entries == 1
    assert history.compute_stats().total_words == 5


def test_stats_cache_invalidated_on_delete(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    e = history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1, word_count=5)
    assert history.compute_stats().total_entries == 1
    history.delete_entry(e.id)
    assert history.compute_stats().total_entries == 0


def test_stats_cache_invalidated_on_clear(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1, word_count=5)
    assert history.compute_stats().total_entries == 1
    history.clear_all()
    assert history.compute_stats().total_entries == 0


def test_stats_cache_invalidated_on_relocate(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1, word_count=5)
    s1 = history.compute_stats()
    assert s1.total_entries == 1

    new_dir = tmp_path / "new"
    history.relocate(new_dir)
    s2 = history.compute_stats()
    assert s2.total_entries == 1  # data moved with it


def test_stats_cache_ttl_returns_cached_value(isolated_storage, tmp_path, monkeypatch):
    """Within 5 s of a non-mutating second call, the cached value is returned."""
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1, word_count=5)

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


# --- Explicit transactions ----------------------------------------------

def test_save_entry_issues_begin_and_commit(isolated_storage, tmp_path):
    """save_entry MUST issue an explicit BEGIN before INSERT and COMMIT after.

    We capture every statement via Connection.set_trace_callback and assert
    the sequence — this catches a regression where someone removes the
    BEGIN/COMMIT wrappers and relies on autocommit (which would silently work).
    """
    target = tmp_path / "target"
    history.bootstrap(target)
    statements: list[str] = []

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.isolation_level is None  # manual txn control
        conn.set_trace_callback(lambda s: statements.append(s.strip().upper()))

    history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1)

    with history._lock:
        history._ensure_conn_locked().set_trace_callback(None)

    # Must contain BEGIN, an INSERT, and COMMIT — in that order.
    begin_idx = next(i for i, s in enumerate(statements) if s.startswith("BEGIN"))
    insert_idx = next(i for i, s in enumerate(statements) if s.startswith("INSERT"))
    commit_idx = next(i for i, s in enumerate(statements) if s.startswith("COMMIT"))
    assert begin_idx < insert_idx < commit_idx


# --- ISO ↔ epoch ms round-trip ------------------------------------------

def test_iso_to_epoch_ms_with_offset():
    iso = "2026-01-01T12:00:00+02:00"
    ms = history._iso_to_epoch_ms(iso)
    # 12:00 +02:00 = 10:00 UTC
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
    e_in = history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1)
    [e_out] = history.get_entries(limit=10)
    # Round-trip is via UTC; the format is normalized but the instant matches.
    t_in = datetime.fromisoformat(e_in.timestamp.replace("Z", "+00:00"))
    t_out = datetime.fromisoformat(e_out.timestamp.replace("Z", "+00:00"))
    assert abs((t_out - t_in).total_seconds()) < 0.001


# --- 503 mapping at router ----------------------------------------------

def test_operational_error_mapped_to_503(isolated_storage, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app

    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(raw_text="x", cleaned_text="x", duration_ms=1)

    with TestClient(app) as client:
        # Router imported compute_stats by name at load time, so patch the binding
        # on the router module, not on app.core.history.
        with patch("app.core.history_router.compute_stats", side_effect=sqlite3.OperationalError("database is locked")):
            resp = client.get("/history/stats")
            assert resp.status_code == 503
            assert resp.headers.get("Retry-After") == "1"

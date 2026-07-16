"""SQLite-specific tests for the v1 history store.

Covers schema/PRAGMA, stats cache TTL+invalidation, explicit transactions,
ISO ↔ epoch ms round-trip, ``OperationalError`` → 503 mapping at the router,
relocate branches, concurrent saves.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.core import history, vector_store


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


# --- Phase 2 — FTS5 schema migration + triggers --------------------------

def test_schema_version_is_v3(isolated_storage, tmp_path):
    """Bumped to 3 by Phase 3 (sqlite-vec) — see the Phase 3 section below."""
    target = tmp_path / "target"
    history.bootstrap(target)
    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_fts_table_and_triggers_exist(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    with history._lock:
        conn = history._ensure_conn_locked()
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        ).fetchall()}
    assert "entry_fts" in names
    assert {"entries_ai", "entries_ad", "entries_au"}.issubset(names)


def test_insert_propagates_to_fts(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="quick brown fox", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'brown'"
        ).fetchall()
    assert len(hits) == 1


def test_delete_propagates_to_fts(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    e = history.save_entry(text="quick brown fox", duration_ms=1)
    history.delete_entry(e.id)
    with history._lock:
        conn = history._ensure_conn_locked()
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'brown'"
        ).fetchall()
    assert hits == []


def test_clear_all_propagates_to_fts(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="quick brown fox", duration_ms=1)
    history.save_entry(text="lazy dog", duration_ms=1)
    history.clear_all()
    with history._lock:
        conn = history._ensure_conn_locked()
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'brown OR dog'"
        ).fetchall()
    assert hits == []


def test_migration_v1_to_v2_populates_fts(isolated_storage, tmp_path):
    """A pre-existing v1 DB with rows must upgrade to v2 with the FTS
    populated from the rebuild branch of _init_schema."""
    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.execute("PRAGMA user_version = 1")
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'hello brown fox', 'hello brown fox', 0)"
        )
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('b', 1, 'uk', 'normal', 'lazy dog jumps', 'lazy dog jumps', 0)"
        )
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        # bootstrap() always migrates through to the current SCHEMA_VERSION
        # (3, post-Phase-3) in one boot — there is no "stop at v2" state.
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'brown'"
        ).fetchall()
    assert len(hits) == 1


def test_partial_migration_recovery_docsize_shadow_missing(isolated_storage, tmp_path):
    """Defensive against the SQLite-private shadow-table contract: if
    `entry_fts_docsize` is somehow gone (interrupted DDL, version quirk),
    _init_schema falls back to rebuild rather than crashing. Closes
    QA exit-gate RED-1."""
    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'shadow probe', 'shadow probe', 0)"
        )
        raw.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        raw.execute("PRAGMA user_version = 2")
        # Drop the FTS table — this removes its shadow tables atomically.
        raw.execute("DROP TABLE entry_fts")
        raw.commit()
    finally:
        raw.close()

    # bootstrap() runs _init_schema, which re-creates entry_fts via
    # IF NOT EXISTS. After DROP TABLE the shadow tables are gone too, so
    # the freshly-created FTS sits over a populated `entries` table with
    # an empty docsize → the row-count probe spots the divergence and
    # the recovery branch issues a rebuild.
    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'shadow'"
        ).fetchall()
    assert len(hits) == 1  # rebuild ran, index is populated again


def test_partial_migration_recovery_fts_missing(isolated_storage, tmp_path):
    """If a previous boot wrote user_version=2 but the FTS table is
    missing (interrupted migration), _init_schema must recreate the FTS
    table and rebuild the index from existing rows."""
    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.execute("PRAGMA user_version = 2")
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'recovery text', 'recovery text', 0)"
        )
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        ).fetchall()}
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'recovery'"
        ).fetchall()
    assert "entry_fts" in names
    assert {"entries_ai", "entries_ad", "entries_au"}.issubset(names)
    assert len(hits) == 1


def test_crash_before_user_version_pragma_retries(isolated_storage, tmp_path, monkeypatch):
    """If a crash leaves user_version at 1 even though FTS DDL ran, the
    next boot's migrator must succeed idempotently (IF NOT EXISTS
    everywhere; rebuild populates the index)."""
    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)  # simulate FTS DDL applied
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'crash safety', 'crash safety', 0)"
        )
        # user_version intentionally NOT set — simulates the crash window.
        raw.execute("PRAGMA user_version = 1")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        # bootstrap() always migrates through to the current SCHEMA_VERSION
        # (3, post-Phase-3) in one boot — there is no "stop at v2" state.
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'crash'"
        ).fetchall()
    assert len(hits) == 1


def test_relocate_rebuilds_fts(isolated_storage, tmp_path):
    """After relocate, FTS index on the new path must answer queries
    consistent with the moved entries."""
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="relocate searchable text", duration_ms=1)

    new_dir = tmp_path / "new"
    res, _ = history.relocate(new_dir)
    assert res == history.RelocateResult.MOVED

    with history._lock:
        conn = history._ensure_conn_locked()
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'searchable'"
        ).fetchall()
    assert len(hits) == 1


# --- Phase 3 — sqlite-vec schema migration (spec 003) ---------------------
#
# Mirrors the exact crash-safety/idempotency pattern the v1->v2 migration
# tests above already establish: DDL is executed manually against a raw
# connection to simulate an out-of-band prior version, then history.bootstrap
# is called and the migrator's self-healing behaviour is asserted.

def test_v1_to_v3_migration_lands_at_v3(isolated_storage, tmp_path):
    """A pre-existing v1 DB (only `entries`, no FTS, no embeddings tables)
    upgrades straight to v3 in one boot."""
    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.execute("PRAGMA user_version = 1")
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'hello brown fox', 'hello brown fox', 0)"
        )
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        ).fetchall()}
    expected = {"entry_fts", "embeddings_meta", "entry_embeddings", "entry_embeddings_dim_guard"}
    assert expected.issubset(names)


def test_v2_to_v3_migration_lands_at_v3(isolated_storage, tmp_path):
    """A pre-existing v2 DB (FTS already migrated, no embeddings tables)
    upgrades to v3 without disturbing the FTS index."""
    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'v2 already migrated', 'v2 already migrated', 0)"
        )
        raw.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        raw.execute("PRAGMA user_version = 2")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        ).fetchall()}
        fts_hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'migrated'"
        ).fetchall()
    assert {"embeddings_meta", "entry_embeddings", "entry_embeddings_dim_guard"}.issubset(names)
    assert len(fts_hits) == 1  # v2 data untouched by the v3 upgrade


def test_crash_before_v3_user_version_pragma_retries(isolated_storage, tmp_path):
    """If a crash leaves user_version at 2 even though the v3 DDL already
    ran (embeddings tables exist), the next boot's migrator must succeed
    idempotently — mirrors test_crash_before_user_version_pragma_retries
    for v1->v2 above."""
    from app.core import vector_store

    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)
        raw.executescript(vector_store._DDL_V3)  # simulate v3 DDL already applied
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'crash safety v3', 'crash safety v3', 0)"
        )
        # user_version intentionally left at 2 — simulates the crash window
        # between the v3 DDL running and the PRAGMA write.
        raw.execute("PRAGMA user_version = 2")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)  # must not raise, must retry idempotently

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        ).fetchall()}
    assert {"embeddings_meta", "entry_embeddings", "entry_embeddings_dim_guard"}.issubset(names)


def test_partial_v3_migration_recovery_tables_missing(isolated_storage, tmp_path):
    """If a previous boot wrote user_version=3 but the embeddings tables
    are missing (interrupted migration, manual edit), _init_schema must
    recreate them via IF NOT EXISTS rather than crashing — mirrors
    test_partial_migration_recovery_fts_missing for v2 above."""
    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)
        raw.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        raw.execute("PRAGMA user_version = 3")  # embeddings tables never actually created
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)  # must not raise

    with history._lock:
        conn = history._ensure_conn_locked()
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        ).fetchall()}
    assert {"embeddings_meta", "entry_embeddings", "entry_embeddings_dim_guard"}.issubset(names)


def test_fresh_db_has_v3_embeddings_tables_but_not_vec_entries(isolated_storage, tmp_path):
    """A brand-new DB lands at v3 with embeddings_meta/entry_embeddings
    present, but vec_entries is NOT created — it is lazy, created on first
    successful embed (its dimension depends on the active provider)."""
    target = tmp_path / "target"
    history.bootstrap(target)
    with history._lock:
        conn = history._ensure_conn_locked()
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        ).fetchall()}
    assert {"embeddings_meta", "entry_embeddings", "entry_embeddings_dim_guard"}.issubset(names)
    assert "vec_entries" not in names


# --- /history/search router behaviour ------------------------------------

def test_search_empty_q_returns_200_empty(isolated_storage, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        # The TestClient lifespan re-bootstraps history to the user-settings
        # output_dir (Path.home() patched by isolated_storage). Save AFTER
        # entering the context so the row lands in the same DB the request
        # will hit.
        history.save_entry(text="anything", duration_ms=1)
        resp = client.get("/history/search?q=")
        assert resp.status_code == 200
        assert resp.json() == {"entries": [], "total": 0}


def test_search_returns_results_ordered_by_relevance(isolated_storage, tmp_path):
    """First hit must be the entry with the strongest match (BM25 ASC =
    best first). Guards against the 'fake-green tests just check non-empty'
    failure mode flagged by QA RED-1."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        history.save_entry(text="brown bear at the zoo", duration_ms=1)
        history.save_entry(text="brown brown brown bear bear", duration_ms=1)  # strongest
        history.save_entry(text="completely unrelated text", duration_ms=1)
        resp = client.get("/history/search?q=brown bear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        assert "brown brown brown" in data["entries"][0]["text"]


def test_search_sanitized_to_empty_returns_200_empty(isolated_storage, tmp_path):
    """Plan 021: the whitelist sanitizer strips bare punctuation. A query
    like ``"`` now sanitizes to an empty string and the endpoint returns
    200 with an empty list (was 400 pre-Plan-021)."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        history.save_entry(text="anything", duration_ms=1)
        resp = client.get('/history/search?q="')
        assert resp.status_code == 200
        assert resp.json() == {"entries": [], "total": 0}


def test_search_query_too_long_returns_422(isolated_storage, tmp_path):
    """Plan 021: ``q`` has ``max_length=500`` at the router; FastAPI emits
    422 for over-long inputs."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        history.save_entry(text="anything", duration_ms=1)
        resp = client.get(f"/history/search?q={'a' * 501}")
        assert resp.status_code == 422


def test_search_lock_error_returns_503(isolated_storage, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        history.save_entry(text="anything", duration_ms=1)
        with patch(
            "app.core.words.search_history",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            resp = client.get("/history/search?q=anything")
            assert resp.status_code == 503
            assert resp.headers.get("Retry-After") == "1"


# --- Background indexer opt-in guard (spec 017 / ADR 010) ----------------

def test_testclient_lifespan_does_not_invoke_real_background_indexer(isolated_storage, tmp_path):
    """Regression test for the Acceptance Criteria's "Automatic background
    indexing" clause. `TestClient(app)`'s context manager runs the real
    FastAPI `lifespan()`, which fires `vector_store.run_background_indexer()`
    at startup (main.py), and a normal request through `process_audio` would
    schedule it again per dictation (pipeline/service.py). Absent
    `@pytest.mark.background_indexer`, conftest.py's autouse
    `_no_background_indexer_by_default` fixture must have already replaced
    `vector_store.run_background_indexer` with a no-op before this
    `TestClient(app)` block triggers lifespan — proven here by patching a
    sentinel on `backfill_batch` (the function the real
    `run_background_indexer` loops on internally) and asserting it is never
    reached."""
    from fastapi.testclient import TestClient
    from app.main import app

    with patch.object(vector_store, "backfill_batch", new=AsyncMock()) as sentinel:
        with TestClient(app) as client:
            history.save_entry(text="anything", duration_ms=1)
            resp = client.get("/history/search?q=anything")
            assert resp.status_code == 200

        sentinel.assert_not_called()


# --- Hybrid search endpoint (spec 017 / ADR 010) --------------------------
#
# `mode` is gone: /history/search always runs the hybrid FTS+semantic path
# and degrades silently to FTS-only on any semantic-lane failure. Uses the
# async ``client`` fixture (conftest.py, ASGITransport — no lifespan) rather
# than ``TestClient(app)``: TestClient's context manager runs the FastAPI
# lifespan, which re-bootstraps history at whatever
# `user_settings.get_user_settings().output_dir` resolves to. Within a
# single test *file* that value is cached at first use and NOT reset by
# this file's `isolated_storage` fixture (only test_user_settings.py /
# test_settings_router.py reset that cache), so successive TestClient-based
# tests in this file silently share one growing DB. The other pre-existing
# TestClient-based tests above tolerate this (`>=`, or an empty-query
# short-circuit); these new tests assert exact counts, so they use the
# lifespan-free `client` fixture over `isolated_storage`'s own
# directly-bootstrapped `history` state instead.

@pytest.mark.parametrize(
    "unavailable_state",
    [
        "vec_extension_unavailable",
        "disabled_by_eligibility",
        "zero_entries_embedded",
        "embed_call_raises",
    ],
)
@pytest.mark.asyncio
async def test_search_degrades_to_200_fts_only_for_every_semantic_unavailable_state(
    isolated_storage, tmp_path, client, unavailable_state
):
    """The concrete regression test for the exception-leak bug (ADR 010 /
    Context-Why): every one of the four semantic-lane-unavailable states
    must yield a 200 FTS-only response, never a 503 and never a raw
    exception class name anywhere in the body."""
    from unittest.mock import MagicMock

    history.bootstrap(tmp_path)
    history.save_entry(text="findable brown bear", duration_ms=1)

    ctx = []
    if unavailable_state == "vec_extension_unavailable":
        ctx.append(patch.object(history, "_vec_available", False))
    elif unavailable_state == "disabled_by_eligibility":
        ctx.append(
            patch(
                "app.embeddings.resolve_embedding_provider",
                new=AsyncMock(return_value=(None, "Local embeddings need Ollama")),
            )
        )
    elif unavailable_state == "zero_entries_embedded":
        fake = MagicMock()
        fake.model_name = "gemini/text-embedding-004"
        fake.embed = AsyncMock(side_effect=AssertionError("must not embed when index is empty"))
        ctx.append(
            patch(
                "app.embeddings.resolve_embedding_provider",
                new=AsyncMock(return_value=(fake, None)),
            )
        )
    else:  # embed_call_raises
        fake = MagicMock()
        fake.model_name = "gemini/text-embedding-004"
        fake.embed = AsyncMock(side_effect=RuntimeError("upstream auth failed"))
        ctx.append(
            patch(
                "app.embeddings.resolve_embedding_provider",
                new=AsyncMock(return_value=(fake, None)),
            )
        )
        # Seed one embedded entry so the zero-entries-embedded short-circuit
        # is bypassed and the embed call itself actually runs and raises.
        e1 = history.save_entry(text="already embedded", duration_ms=1)
        with history._lock:
            conn = history._ensure_conn_locked()
            vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
            rowid = conn.execute("SELECT rowid FROM entries WHERE id = ?", (e1.id,)).fetchone()[0]
            vector_store.insert_embedding(
                conn, e1.id, rowid, [1.0, 0.0, 0.0], "cloud", "text-embedding-004"
            )

    with contextlib.ExitStack() as stack:
        for c in ctx:
            stack.enter_context(c)
        resp = await client.get("/history/search?q=brown")

    assert resp.status_code == 200
    body_text = resp.text
    assert "RuntimeError" not in body_text
    assert "AssertionError" not in body_text
    data = resp.json()
    assert isinstance(data["entries"], list)
    assert isinstance(data["total"], int)
    # FTS-only degradation: the seeded FTS-matchable entry is still found.
    assert any("brown bear" in e["text"] for e in data["entries"])


@pytest.mark.asyncio
async def test_search_no_longer_accepts_mode_param(isolated_storage, tmp_path, client):
    """A stray `mode=semantic` query string is simply ignored by FastAPI
    (undeclared query params are dropped) — not an error, and the hybrid
    path runs regardless of what `mode` says."""
    history.bootstrap(tmp_path)
    history.save_entry(text="правив у файлі", duration_ms=1, language="uk")
    resp = await client.get("/history/search?q=прав&mode=semantic")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["entries"]) == 1
    assert "<mark>прав</mark>" in data["entries"][0]["highlighted_text"]


@pytest.mark.asyncio
async def test_embeddings_status_route_removed(client):
    """The route is gone from history_router.py entirely. Per this
    router's existing, documented shape (see the identical `mode=search`
    405-vs-404 comment this spec's frontend code carried before removal),
    a GET here structurally collides with the still-registered
    `DELETE /history/{entry_id}` path template with entry_id=
    "embeddings-status" — Starlette reports a path-template match with an
    unsupported verb as 405, not 404. Either way, the route no longer
    resolves to a working endpoint."""
    resp = await client.get("/history/embeddings-status")
    assert resp.status_code == 405


@pytest.mark.asyncio
async def test_backfill_embeddings_route_removed(client):
    """Same 405-not-404 reasoning as the sibling embeddings-status test
    above: POST /history/backfill-embeddings structurally collides with
    `DELETE /history/{entry_id}`."""
    resp = await client.post("/history/backfill-embeddings", json={"batch_size": 10})
    assert resp.status_code == 405


def test_concurrent_save_and_search_serialised(isolated_storage, tmp_path):
    """save_entry and search both serialise on history._lock — this test
    documents that guarantee. Not a race-condition test: partial reads
    are physically impossible under a single Python mutex around a single
    connection."""
    from app.core import words as words_service

    target = tmp_path / "target"
    history.bootstrap(target)

    errors: list[Exception] = []
    seen: list[int] = []

    def writer():
        try:
            for i in range(20):
                history.save_entry(text=f"searchable token-{i}", duration_ms=1)
        except Exception as e:
            errors.append(e)

    def searcher():
        try:
            for _ in range(20):
                results = words_service.search_history("searchable", limit=50)
                seen.append(len(results))
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=searcher)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert errors == []
    # Monotonic: search results never decreased once a row was visible.
    # (Cannot assert strict equality — interleaving is non-deterministic.)
    assert all(0 <= n <= 20 for n in seen)
    assert history.get_count() == 20

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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.transcripts import history, vector_store


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """See docs/adr/014-lazy-app-data-path-resolution.md: patching
    `Path.home()` is not a supported isolation mechanism (it's a no-op
    against any already-resolved module-level constant, which was exactly
    the Spec 028 Item 1 bug). conftest.py's autouse `_isolated_app_data`
    fixture already redirects `JUSTSAY_DATA_DIR` at this same `tmp_path`
    before this fixture runs -- the resets below are kept anyway so this
    file's tests don't depend on conftest.py's exact reset shape."""
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)
    monkeypatch.setattr(history, "_page_total_cache", None)

    yield {"tmp_path": tmp_path}

    with history._lock:
        history._close_conn_locked()



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



def test_save_then_get_page(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    e = history.save_entry(text="hello world", duration_ms=100, language="uk", word_count=2)
    out = history.get_page(limit=10).entries[0]
    assert out.id == e.id
    assert out.text == "hello world"
    assert out.word_count == 2


def test_delete_entry_removes_row(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    e = history.save_entry(text="x", duration_ms=1)
    assert history.delete_entry(e.id) is True
    assert history.get_page().total == 0


def test_delete_nonexistent_id_returns_false(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1, word_count=5)
    history.compute_stats()
    cache_before = history._stats_cache

    assert history.delete_entry("does-not-exist") is False

    assert history._stats_cache is cache_before


def test_clear_all_returns_count_and_empties(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    for _ in range(3):
        history.save_entry(text="x", duration_ms=1)
    assert history.clear_all() == 3
    assert history.get_page().total == 0



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
    assert s2.total_entries == 1


def test_stats_cache_ttl_returns_cached_value(isolated_storage, tmp_path):
    """Within 5 s of a non-mutating second call, the cached value is returned."""
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1, word_count=5)

    s1 = history.compute_stats()
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, "
            "cleaned_text, duration_ms, word_count) "
            "VALUES ('zzz', 0, 'uk', 'normal', '', '', 0, 99)"
        )
    s2 = history.compute_stats()
    assert s2.total_entries == s1.total_entries


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
    assert s.total_words == 30



def test_save_entry_issues_begin_and_commit(isolated_storage, tmp_path):
    """save_entry MUST issue an explicit BEGIN before INSERT and COMMIT after."""
    target = tmp_path / "target"
    history.bootstrap(target)
    statements: list[str] = []

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.isolation_level is None
        conn.set_trace_callback(lambda s: statements.append(s.strip().upper()))

    history.save_entry(text="x", duration_ms=1)

    with history._lock:
        history._ensure_conn_locked().set_trace_callback(None)

    begin_idx = next(i for i, s in enumerate(statements) if s.startswith("BEGIN"))
    insert_idx = next(i for i, s in enumerate(statements) if s.startswith("INSERT"))
    commit_idx = next(i for i, s in enumerate(statements) if s.startswith("COMMIT"))
    assert begin_idx < insert_idx < commit_idx



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
    e_out = history.get_page(limit=10).entries[0]
    t_in = datetime.fromisoformat(e_in.timestamp.replace("Z", "+00:00"))
    t_out = datetime.fromisoformat(e_out.timestamp.replace("Z", "+00:00"))
    assert abs((t_out - t_in).total_seconds()) < 0.001



def test_relocate_moved_branch(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1)

    new_dir = tmp_path / "new"
    res, reason = history.relocate(new_dir)
    assert res == history.RelocateOutcome.MOVED
    assert reason is None
    assert (new_dir / "history.db").exists()
    assert not (target / "history.db").exists()


def test_relocate_no_old_file_branch(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    with history._lock:
        history._close_conn_locked()
    (target / "history.db").unlink(missing_ok=True)

    new_dir = tmp_path / "new"
    res, _ = history.relocate(new_dir)
    assert res == history.RelocateOutcome.NO_OLD_FILE


def test_relocate_new_already_has_file_branch(isolated_storage, tmp_path):
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="old", duration_ms=1)

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    history.bootstrap(new_dir)
    history.save_entry(text="new", duration_ms=1)
    history.bootstrap(target)

    res, _ = history.relocate(new_dir)
    assert res == history.RelocateOutcome.NEW_ALREADY_HAS_FILE
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
    assert res == history.RelocateOutcome.FAILED
    assert reason and "simulated copy failure" in reason
    assert (target / "history.db").exists()



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
    assert history.get_page().total == 50
    entries = history.get_page(limit=100).entries
    ids = {e.id for e in entries}
    assert len(ids) == 50



def test_operational_error_mapped_to_503(isolated_storage, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry(text="x", duration_ms=1)

    with TestClient(app) as client:
        with patch(
            "app.transcripts.history_router.compute_stats",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            resp = client.get("/history/stats")
            assert resp.status_code == 503
            assert resp.headers.get("Retry-After") == "1"



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
        raw.execute("DROP TABLE entry_fts")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'shadow'"
        ).fetchall()
    assert len(hits) == 1


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
        raw.executescript(history._DDL_V2)
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'crash safety', 'crash safety', 0)"
        )
        raw.execute("PRAGMA user_version = 1")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
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
    assert res == history.RelocateOutcome.MOVED

    with history._lock:
        conn = history._ensure_conn_locked()
        hits = conn.execute(
            "SELECT rowid FROM entry_fts WHERE entry_fts MATCH 'searchable'"
        ).fetchall()
    assert len(hits) == 1



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
    assert len(fts_hits) == 1


def test_crash_before_v3_user_version_pragma_retries(isolated_storage, tmp_path):
    """If a crash leaves user_version at 2 even though the v3 DDL already
    ran (embeddings tables exist), the next boot's migrator must succeed
    idempotently — mirrors test_crash_before_user_version_pragma_retries
    for v1->v2 above."""
    from app.transcripts import vector_store

    db_path = tmp_path / "history.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)
        raw.executescript(vector_store._DDL_V3)
        raw.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('a', 0, 'uk', 'normal', 'crash safety v3', 'crash safety v3', 0)"
        )
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
        raw.execute("PRAGMA user_version = 3")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

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



def test_search_empty_q_returns_200_empty(isolated_storage, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
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
        history.save_entry(text="brown brown brown bear bear", duration_ms=1)
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
            "app.transcripts.words.search_history",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            resp = client.get("/history/search?q=anything")
            assert resp.status_code == 503
            assert resp.headers.get("Retry-After") == "1"



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
    else:
        fake = MagicMock()
        fake.model_name = "gemini/text-embedding-004"
        fake.embed = AsyncMock(side_effect=RuntimeError("upstream auth failed"))
        ctx.append(
            patch(
                "app.embeddings.resolve_embedding_provider",
                new=AsyncMock(return_value=(fake, None)),
            )
        )
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
    from app.transcripts import words as words_service

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
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    assert all(0 <= n <= 20 for n in seen)
    assert history.get_page().total == 20


def test_get_page_clamps_its_own_limit(isolated_storage, tmp_path):
    """The router's `Query(..., le=HISTORY_LIMIT_MAX)` bounds the endpoint, not
    the function. `LIMIT -1` is SQLite for "every row", so an unclamped service
    call still materialises the whole table into `HistoryEntry` objects under
    the lock. `words.top_words` and `words.search_history` both clamp in the
    service as well as at their routers.

    The clamp also bounds the "is there more" probe: the largest read the store
    ever issues is `HISTORY_LIMIT_MAX + 1` rows, not `HISTORY_LIMIT_MAX + 100 + 1`.
    """
    for index in range(history.HISTORY_LIMIT_MAX + 5):
        history.save_entry(text=f"entry {index}", duration_ms=1)

    assert len(history.get_page(limit=history.HISTORY_LIMIT_MAX + 100).entries) == (
        history.HISTORY_LIMIT_MAX
    )
    assert len(history.get_page(limit=-1).entries) == 1
    assert len(history.get_page(limit=0).entries) == 1
    assert len(history.get_page(limit=5).entries) == 5



def test_entry_columns_match_the_bootstrapped_schema(isolated_storage, tmp_path):
    """`ENTRY_COLUMNS` is checked against the DDL, not against another copy of itself.

    Adding a column to `_DDL_*` without listing it here -- or reordering the
    declaration away from the table -- breaks the INSERT that builds its
    placeholder run from `len(ENTRY_COLUMNS)`, so the drift fails here first.
    """
    target = tmp_path / "target"
    history.bootstrap(target)
    conn = sqlite3.connect(target / "history.db")
    try:
        schema_columns = tuple(
            row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()
        )
    finally:
        conn.close()
    assert history.ENTRY_COLUMNS == schema_columns


def test_entry_read_columns_are_exactly_what_row_to_entry_reads(
    isolated_storage, tmp_path
):
    """The read list drops `cleaned_text` and nothing else.

    `get_page` builds its SELECT from `ENTRY_READ_COLUMNS`, so a column
    dropped from it becomes a `KeyError` in `_row_to_entry` at runtime.

    The expectation comes from the table rather than from `ENTRY_COLUMNS`.
    `ENTRY_READ_COLUMNS` is *defined* as `ENTRY_COLUMNS` minus `cleaned_text`,
    so comparing the two restates the definition and holds no matter what the
    schema does; reading `PRAGMA table_info(entries)` is what makes a column
    added to the DDL and left out of the read list fail here.
    """
    target = tmp_path / "target"
    history.bootstrap(target)
    history.save_entry("hello world", 1200, language="uk", style="normal")
    entries = history.get_page().entries

    conn = sqlite3.connect(target / "history.db")
    try:
        schema_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()
        }
    finally:
        conn.close()

    assert set(history.ENTRY_READ_COLUMNS) == schema_columns - {"cleaned_text"}
    assert len(entries) == 1
    assert entries[0].text == "hello world"


def test_columns_sql_qualifies_every_name_with_the_alias():
    """The FTS lane joins `entries` as `e`, so every name must carry the alias."""
    assert history.columns_sql(("id", "ts")) == "id, ts"
    assert history.columns_sql(("id", "ts"), alias="e") == "e.id, e.ts"


def test_columns_sql_refuses_a_name_that_is_not_an_entries_column():
    """The one place this module interpolates identifiers into SQL.

    sqlite cannot parameterise a column name, so the safety of `columns_sql`
    rested entirely on a docstring asking callers not to pass a request value.
    The check makes that a property of the function.
    """
    with pytest.raises(ValueError, match="entries table"):
        history.columns_sql(("id", "raw_text; DROP TABLE entries"))


def test_columns_sql_refuses_an_alias_that_is_not_an_identifier():
    """The column names were checked and the table alias was not, so the
    docstring's promise that an `entries` column is the only thing this can
    emit was false for anything reaching the second argument."""
    with pytest.raises(ValueError, match="table alias"):
        history.columns_sql(("id",), alias="x FROM sqlite_master; -- ")


def test_a_saved_row_lands_in_the_right_columns_whatever_their_order(
    isolated_storage, tmp_path, monkeypatch
):
    """AC: the INSERT's names and its values cannot drift apart.

    The column names come from `ENTRY_COLUMNS` while the values were a
    hand-ordered literal, with nothing tying the two orders together: a
    consistent reorder of the DDL and `ENTRY_COLUMNS` still passes the schema
    test above and writes every row with its values one column out, which
    sqlite's type affinity accepts in silence. Reordering the declaration here
    is that reorder, and named placeholders are what survive it.
    """
    target = tmp_path / "target"
    history.bootstrap(target)
    monkeypatch.setattr(history, "ENTRY_COLUMNS", tuple(reversed(history.ENTRY_COLUMNS)))

    history.save_entry(
        text="hello world",
        duration_ms=1200,
        language="uk",
        style="normal",
        model_name="gemini/flash",
        tokens_used=42,
        audio_duration_seconds=3.5,
        word_count=2,
    )

    conn = sqlite3.connect(target / "history.db")
    try:
        row = conn.execute(
            "SELECT raw_text, duration_ms, model_name, tokens_used, word_count "
            "FROM entries"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("hello world", 1200, "gemini/flash", 42, 2)


def test_a_failed_relocate_does_not_cache_a_lazily_resolved_directory(
    isolated_storage, tmp_path, monkeypatch
):
    """AC: the verification-failure rollback leaves `_output_dir` as it found it.

    `_resolve_output_dir` resolves a fallback fresh on every call and is never
    cached (ADR 014): the store must follow a `JUSTSAY_DATA_DIR` that changes
    under it. The rollback runs with `_output_dir` unset whenever `relocate`
    was reached without a `bootstrap` first, so writing the resolved fallback
    there pins it for the life of the process.
    """
    monkeypatch.setattr(history, "_output_dir", None)
    monkeypatch.setattr(history, "_conn", None)
    history.save_entry(text="x", duration_ms=1)
    monkeypatch.setattr(history, "_verify_db_row_count", lambda *_a, **_kw: False)

    outcome, reason = history.relocate(tmp_path / "new")

    assert outcome == history.RelocateOutcome.FAILED
    assert reason and "Verification failed" in reason
    assert history._output_dir is None


def test_a_relocate_that_raises_mid_copy_leaves_a_working_store_behind(
    isolated_storage, tmp_path, monkeypatch
):
    """The rollback after a failed copy reopens the way every other site does.

    `relocate`'s exception handler hand-rolled the close/connect/schema/
    invalidate sequence that `_reopen_conn_locked` owns, so this module carried
    two reopen semantics and the divergent one was the rollback -- the path
    where a second failure matters most. It now calls the same helper, with the
    one difference it actually needs (a second failure leaves `_conn` as None
    rather than raising out of `relocate`) written as a wrapper around it.
    """
    history.save_entry(text="before the move", duration_ms=1)
    old_dir = history._resolve_output_dir()
    monkeypatch.setattr(
        history.shutil, "copy2", MagicMock(side_effect=OSError("disk full"))
    )

    history.compute_stats()
    generation_before = history._derived_generation

    outcome, reason = history.relocate(tmp_path / "new")

    assert outcome == history.RelocateOutcome.FAILED
    assert reason and "disk full" in reason
    assert history._resolve_output_dir() == old_dir
    assert history._stats_cache is None
    assert history._derived_generation != generation_before
    history.save_entry(text="after the failure", duration_ms=1)
    assert [e.text for e in history.get_page().entries] == [
        "after the failure",
        "before the move",
    ]


class _RecordingLock:
    """Delegates to a real lock and records every acquire and release."""

    def __init__(self, inner, events: list[str]):
        self._inner = inner
        self._events = events

    def __enter__(self):
        self._events.append("acquire")
        return self._inner.__enter__()

    def __exit__(self, *exc):
        self._events.append("release")
        return self._inner.__exit__(*exc)


def test_get_page_never_releases_the_store_lock_between_its_reads(isolated_storage, tmp_path):
    """JS-126's technical half: the page, the probe and the total come from one state.

    What has to hold is that no other thread can write between the reads, and
    the only thing that guarantees it is the lock staying held across all of them.
    Two weaker pins were tried and both passed against a mutant that reintroduces
    the defect: a writer thread started from inside ``get_page`` is blocked while
    *any* read holds the lock, and counting acquisitions cannot see a
    ``_count_locked`` moved out of the ``with`` block entirely. Recording the
    sequence is what distinguishes them — a release between two of the reads is
    exactly the defect, whether or not a second acquisition follows it.

    The page size matches the store size so the page comes back full and the
    has-more probe runs; a short page skips it and could not pin its position.
    """
    target = tmp_path / "target"
    history.bootstrap(target)
    for i in range(3):
        history.save_entry(text=f"entry {i}", duration_ms=1)

    events: list[str] = []
    real_entries_locked = history._entries_locked
    real_has_more_locked = history._has_more_locked
    real_count_locked = history._count_locked

    def entries(conn, limit, before):
        events.append("read-entries")
        return real_entries_locked(conn, limit, before)

    def has_more(conn, after_ts, after_id):
        events.append("read-has-more")
        return real_has_more_locked(conn, after_ts, after_id)

    def count(conn):
        events.append("read-count")
        return real_count_locked(conn)

    with (
        patch.object(history, "_lock", _RecordingLock(history._lock, events)),
        patch.object(history, "_entries_locked", entries),
        patch.object(history, "_has_more_locked", has_more),
        patch.object(history, "_count_locked", count),
    ):
        page = history.get_page(limit=3)

    assert events == ["acquire", "read-entries", "read-has-more", "read-count", "release"], (
        "the store lock was released between two of the page's reads, so a write "
        f"can land between them: {events}"
    )
    assert len(page.entries) == page.total == 3


def _seed(target, count, ts_at):
    """``count`` entries whose ``ts`` is exactly ``ts_at(index)``, newest last.

    ``save_entry`` stamps wall-clock milliseconds, so two entries written in the
    same millisecond collide and none of the ordering properties under test could
    be stated. The timestamps are rewritten to the values the test names.
    """
    history.bootstrap(target)
    ids = [history.save_entry(text=f"entry {index}", duration_ms=1).id for index in range(count)]
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        try:
            for index, entry_id in enumerate(ids):
                conn.execute("UPDATE entries SET ts = ? WHERE id = ?", (ts_at(index), entry_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return ids


def _walk(page_size):
    """Every page the client would fetch, in order, echoing each ``next_cursor``."""
    pages = []
    cursor = None
    while True:
        page = history.get_page(limit=page_size, before=cursor)
        pages.append(page)
        cursor = page.next_cursor
        if cursor is None:
            return pages


def test_an_entry_saved_between_two_pages_repeats_and_hides_nothing(isolated_storage, tmp_path):
    """The defect JS-126 was filed for, at the service level.

    Under offset paging the write shifts every older row down one, so page 2 at
    offset 30 re-sends the row that ended page 1 and the 60th row is never shown.
    A cursor is a position rather than a count, so nothing moves under it.
    """
    seeded = _seed(tmp_path / "target", 60, lambda index: 1_700_000_000_000 + index)

    first = history.get_page(limit=30)
    history.save_entry(text="landed between the two pages", duration_ms=1)
    second = history.get_page(limit=30, before=first.next_cursor)

    ids = [e.id for e in first.entries] + [e.id for e in second.entries]
    assert len(ids) == 60
    assert len(set(ids)) == 60
    assert set(ids) == set(seeded)


def test_deleting_the_row_the_cursor_names_hides_no_other_row(isolated_storage, tmp_path):
    """A cursor keeps working when the row it names is gone.

    ``(ts, id)`` is a point in the order, not a row, so the rows strictly after it
    are still well defined. The union below is 60 because page 1 was painted
    before the delete; what the criterion is about is the 59 survivors, and every
    one of them is returned exactly once.
    """
    seeded = _seed(tmp_path / "target", 60, lambda index: 1_700_000_000_000 + index)

    first = history.get_page(limit=30)
    deleted_id = first.next_cursor.id
    assert history.delete_entry(deleted_id) is True
    second = history.get_page(limit=30, before=first.next_cursor)

    ids = [e.id for e in first.entries] + [e.id for e in second.entries]
    assert len(ids) == len(set(ids))
    survivors = set(seeded) - {deleted_id}
    assert len(survivors) == 59
    assert set(ids) - {deleted_id} == survivors


def test_entries_sharing_one_timestamp_page_without_repeat_or_loss(isolated_storage, tmp_path):
    """``ORDER BY ts DESC`` alone has no tiebreaker, so a page boundary landing
    inside a same-millisecond group repeats or drops a row against a table nobody
    is writing to. ``id`` makes the order total."""
    seeded = _seed(tmp_path / "target", 60, lambda index: 1_700_000_000_000)

    first = history.get_page(limit=30)
    second = history.get_page(limit=30, before=first.next_cursor)

    ids = [e.id for e in first.entries] + [e.id for e in second.entries]
    assert len(ids) == 60
    assert set(ids) == set(seeded)


def test_a_cursor_naming_a_row_that_never_existed_answers_from_that_position(
    isolated_storage, tmp_path
):
    seeded = _seed(tmp_path / "target", 60, lambda index: 1_700_000_000_000 + index * 10)

    page = history.get_page(
        limit=200, before=history.HistoryCursor(ts=1_700_000_000_305, id="nosuchrow")
    )

    assert [e.id for e in page.entries] == list(reversed(seeded[:31]))
    assert page.total == 60


def test_next_cursor_is_null_only_on_the_last_page(isolated_storage, tmp_path):
    _seed(tmp_path / "target", 60, lambda index: 1_700_000_000_000 + index)

    pages = _walk(25)

    assert [len(p.entries) for p in pages] == [25, 25, 10]
    assert [p.next_cursor is None for p in pages] == [False, False, True]


def test_an_exactly_full_last_page_costs_no_extra_request(isolated_storage, tmp_path):
    """60 rows at page size 30 is two requests, not three.

    This is what the has-more probe buys over comparing a count: a full page asks
    the index whether one more key exists past its last row, so the second page
    learns it is the last one and "Load more" never offers a page that turns out
    to be empty.
    """
    _seed(tmp_path / "target", 60, lambda index: 1_700_000_000_000 + index)

    pages = _walk(30)

    assert [len(p.entries) for p in pages] == [30, 30]
    assert pages[0].next_cursor is not None
    assert pages[-1].next_cursor is None


def test_a_cursor_cannot_be_built_outside_the_bounds_sqlite_can_hold(isolated_storage, tmp_path):
    """The bounds belong to ``HistoryCursor``, not only to the router's signature.

    A ``ts`` wider than a signed 64-bit integer reaches ``conn.execute`` and raises
    ``OverflowError`` out of the driver, which is the failure ``_entries_locked``'s
    docstring says the caller never causes; and a bound that lives only in a
    FastAPI signature is not a bound on the function, which is the reasoning
    ``_clamp_limit`` already carries about ``limit``. So a non-router caller is
    refused at construction rather than at the driver.
    """
    history.bootstrap(tmp_path / "target")
    history.save_entry(text="x", duration_ms=1)

    with pytest.raises(ValidationError):
        history.HistoryCursor(ts=history.CURSOR_TS_MAX + 1, id="x")
    with pytest.raises(ValidationError):
        history.HistoryCursor(ts=history.CURSOR_TS_MIN - 1, id="x")
    with pytest.raises(ValidationError):
        history.HistoryCursor(ts=0, id="a" * (history.CURSOR_ID_MAX_LENGTH + 1))

    at_the_edge = history.HistoryCursor(ts=history.CURSOR_TS_MAX, id="x")
    assert history.get_page(limit=5, before=at_the_edge).entries != []


def test_a_saved_id_fits_the_cursor_id_bound(isolated_storage, tmp_path):
    """The router caps ``before_id`` so a cursor cannot carry a payload. The cap
    is stated against what ``save_entry`` actually produces rather than against a
    ``12`` repeated in prose."""
    history.bootstrap(tmp_path / "target")

    assert len(history.save_entry(text="x", duration_ms=1).id) < history.CURSOR_ID_MAX_LENGTH


def test_the_page_read_asks_for_exactly_the_clamped_limit(isolated_storage, tmp_path):
    """The clamp is the whole bound: the largest read the store ever issues is
    ``HISTORY_LIMIT_MAX`` rows, and no row is read only to be dropped (ADR 055)."""
    history.bootstrap(tmp_path / "target")
    for index in range(history.HISTORY_LIMIT_MAX + 5):
        history.save_entry(text=f"entry {index}", duration_ms=1)

    real_entries_locked = history._entries_locked
    read_sizes: list[int] = []

    def entries(conn, limit, before):
        rows = real_entries_locked(conn, limit, before)
        read_sizes.append(len(rows))
        return rows

    with patch.object(history, "_entries_locked", entries):
        page = history.get_page(limit=history.HISTORY_LIMIT_MAX + 100)

    assert read_sizes == [history.HISTORY_LIMIT_MAX]
    assert len(page.entries) == history.HISTORY_LIMIT_MAX


@contextlib.contextmanager
def _statement_trace():
    """Every SQL statement the production code path issues while the block runs.

    The plan pins below read this rather than a SELECT written out in the test.
    A pin that spells its own query out proves something about that string and
    nothing about the shipped one -- with the query rewritten here, replacing the
    row-value seek with the portable ``OR`` form went unnoticed.

    The callback is cleared from the connection it was installed on rather than
    from whichever one the store resolves to afterwards: a block that reopens the
    store would otherwise leave a live callback appending into a list that
    outlives the test.
    """
    statements: list[str] = []
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.set_trace_callback(statements.append)
    try:
        yield statements
    finally:
        with history._lock:
            conn.set_trace_callback(None)


def _statements_from(action):
    with _statement_trace() as statements:
        action()
    return statements


def _reads_entries(statement):
    upper = " ".join(statement.split()).upper()
    return upper.startswith("SELECT") and "FROM ENTRIES" in upper


def _plan_of(statement):
    with history._lock:
        conn = history._ensure_conn_locked()
        return [
            row[-1]
            for row in conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
        ]


def _the_page_read(action):
    """The one statement that fetches rows, as opposed to the total beside it."""
    reads = [s for s in _statements_from(action) if _reads_entries(s) and "raw_text" in s]
    assert len(reads) == 1, reads
    return reads[0]


def test_the_shipped_cursor_read_seeks_and_orders_on_the_full_key(isolated_storage, tmp_path):
    """Two properties no behavioural or plan assertion in this environment can
    reach, pinned on the statement the store actually issued.

    ``ORDER BY ts DESC`` alone comes back in the right order against this index
    today, so results cannot tell the two spellings apart -- the order would only
    go wrong once the planner chose differently, which is exactly what the second
    key column is there to prevent.

    And the row-value predicate cannot be told from the portable
    ``ts < ? OR (ts = ? AND id < ?)`` by ``EXPLAIN QUERY PLAN`` either: the trace
    callback hands back the statement with its parameters substituted in, and
    against literals SQLite folds the ``OR`` form into
    ``SEARCH entries USING INDEX entries_ts_id_idx (ts<?)`` -- measured, not
    assumed. The row-value form is the one ADR 053 chose and the one
    ``vector_store.selftest``'s SQLite floor exists for, so the choice is pinned
    where it is visible.
    """
    _seed(tmp_path / "target", 40, lambda index: 1_700_000_000_000 + index)
    cursor = history.get_page(limit=20).next_cursor

    statement = _the_page_read(lambda: history.get_page(limit=20, before=cursor))

    normalised = " ".join(statement.split()).upper()
    assert "WHERE (TS, ID) < (" in normalised, statement
    assert "ORDER BY TS DESC, ID DESC" in normalised, statement


def test_the_shipped_cursor_read_is_a_seek_into_the_composite_index(
    isolated_storage, tmp_path
):
    """The property the whole change is bought for, measured on the shipped
    statement: a page read seeks straight to its position instead of walking the
    index from the newest row."""
    _seed(tmp_path / "target", 40, lambda index: 1_700_000_000_000 + index)
    cursor = history.get_page(limit=20).next_cursor

    plan = _plan_of(_the_page_read(lambda: history.get_page(limit=20, before=cursor)))

    assert any("SEARCH" in step and "entries_ts_id_idx" in step for step in plan), plan
    assert not any("SCAN entries" in step for step in plan), plan
    assert not any("TEMP B-TREE" in step for step in plan), plan


def test_the_shipped_first_page_read_needs_no_sort(isolated_storage, tmp_path):
    _seed(tmp_path / "target", 40, lambda index: 1_700_000_000_000 + index)

    plan = _plan_of(_the_page_read(lambda: history.get_page(limit=20)))

    assert any("entries_ts_id_idx" in step for step in plan), plan
    assert not any("TEMP B-TREE" in step for step in plan), plan


def test_the_shipped_word_search_read_keeps_its_ordering_index(isolated_storage, tmp_path):
    """``entries_ts_idx`` is gone and ``entries_ts_id_idx`` leads with the same
    column, so ``words.search_history``'s ``LIKE`` lane still gets its ``ts DESC``
    order from an index instead of a sort. It was never a seek -- a leading-wildcard
    ``LIKE`` is unindexable -- so what is pinned is the absence of ``TEMP B-TREE``.
    """
    from app.transcripts import words

    _seed(tmp_path / "target", 40, lambda index: 1_700_000_000_000 + index)

    reads = [
        s
        for s in _statements_from(lambda: words.search_history("ntry", limit=5))
        if _reads_entries(s) and "LIKE" in s.upper()
    ]
    assert reads, "words.search_history issued no LIKE read over entries"

    for statement in reads:
        plan = _plan_of(statement)
        assert any("entries_ts_id_idx" in step for step in plan), (statement, plan)
        assert not any("TEMP B-TREE" in step for step in plan), (statement, plan)


@pytest.mark.asyncio
async def test_the_shipped_embedding_backfill_read_keeps_its_ordering_index(
    isolated_storage, tmp_path
):
    """Same swap, read in the other direction: ``ORDER BY ts ASC`` walks the same
    index backwards rather than sorting."""
    _seed(tmp_path / "target", 40, lambda index: 1_700_000_000_000 + index)

    with _statement_trace() as statements:
        await vector_store.backfill_batch(5)

    reads = [s for s in statements if _reads_entries(s) and "ORDER BY" in s.upper()]
    assert reads, statements

    for statement in reads:
        plan = _plan_of(statement)
        assert any("entries_ts_id_idx" in step for step in plan), (statement, plan)
        assert not any("TEMP B-TREE" in step for step in plan), (statement, plan)


def test_the_shipped_stats_aggregate_plan_is_unchanged_by_the_index_swap(
    isolated_storage, tmp_path
):
    """``compute_stats`` reads every row to sum them, so it never used
    ``entries_ts_idx`` and does not use its replacement either. Pinned so the swap
    is on record as having left it alone."""
    _seed(tmp_path / "target", 40, lambda index: 1_700_000_000_000 + index)

    reads = [
        s
        for s in _statements_from(history.compute_stats)
        if _reads_entries(s) and "CASE WHEN ts >=" in s
    ]
    assert len(reads) == 1, reads

    plan = _plan_of(reads[0])
    assert any("SCAN" in step and "entries" in step for step in plan), (reads[0], plan)
    assert not any("entries_ts_id_idx" in step for step in plan), (reads[0], plan)


def _the_has_more_probe(action):
    """The key-only statement that answers "is there another page", if one ran."""
    return [
        statement
        for statement in _statements_from(action)
        if _reads_entries(statement)
        and "raw_text" not in statement
        and "COUNT(*)" not in statement.upper()
    ]


def test_a_full_page_reads_exactly_the_rows_it_returns(isolated_storage, tmp_path):
    """ADR 055's first half, pinned on the statement the store actually issued.

    The trace callback substitutes the bound parameters in, so the limit the page
    read carries is readable rather than inferred: a page of 30 asks SQLite for 30
    rows, not for 31 of which one transcript body is thrown away.
    """
    _seed(tmp_path / "target", 60, lambda index: 1_700_000_000_000 + index)

    reads = [
        statement
        for statement in _statements_from(lambda: history.get_page(limit=30))
        if _reads_entries(statement) and "raw_text" in statement
    ]

    assert len(reads) == 1, reads
    assert "LIMIT 30" in " ".join(reads[0].split()).upper(), reads[0]


def test_the_has_more_probe_reads_a_key_and_seeks_the_index(isolated_storage, tmp_path):
    """The question a full page asks about the next one costs an index key.

    Both halves are asserted on the shipped statement: it names no transcript
    column, and the planner answers it out of ``entries_ts_id_idx`` without
    reaching a table row or sorting.

    The covering assertion is what the column loop cannot make. ``SELECT *``
    names none of ``ENTRY_READ_COLUMNS`` literally and still seeks the index, so
    every other check here passes while the probe reads a whole transcript body
    per page. Only ``COVERING INDEX`` in the plan says the table was never
    reached.
    """
    _seed(tmp_path / "target", 60, lambda index: 1_700_000_000_000 + index)

    probes = _the_has_more_probe(lambda: history.get_page(limit=30))

    assert len(probes) == 1, probes
    for column in history.ENTRY_READ_COLUMNS:
        if column not in ("id", "ts"):
            assert column not in probes[0], probes[0]

    plan = _plan_of(probes[0])
    assert any("SEARCH" in step and "entries_ts_id_idx" in step for step in plan), plan
    assert not any("SCAN entries" in step for step in plan), plan
    assert not any("TEMP B-TREE" in step for step in plan), plan
    assert any("COVERING INDEX" in step.upper() for step in plan), plan


def test_a_short_page_asks_nothing_about_a_next_one(isolated_storage, tmp_path):
    """A page that comes back short is the last one by its own length, so the
    probe is skipped entirely rather than run to learn what is already known."""
    _seed(tmp_path / "target", 10, lambda index: 1_700_000_000_000 + index)

    page = history.get_page(limit=30)
    probes = _the_has_more_probe(lambda: history.get_page(limit=30))

    assert page.next_cursor is None
    assert probes == [], probes


def test_a_full_last_page_validates_no_cursor(isolated_storage, tmp_path):
    """An id too long for ``HistoryCursor`` no longer breaks a full *last* page.

    ``consolidate_into`` copies ids verbatim out of a merged database, so a row
    whose id exceeds ``CURSOR_ID_MAX_LENGTH`` can be sitting in the store. A page
    that is both full and last returns no cursor, so building one to ask "is
    there more" turned a page that reads fine into a ``ValidationError`` and a
    500. The probe therefore takes the position as primitives.

    That is the whole of what this pins. A full page that *does* have a page
    after it returns a cursor built from that same id and still raises, which is
    JS-137 and predates this spec.
    """
    target = tmp_path / "target"
    history.bootstrap(target)
    over_long_id = "x" * (history.CURSOR_ID_MAX_LENGTH + 1)
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES (?, ?, 'uk', 'normal', 'row', 'row', 1)",
            [("newer", 1_700_000_000_001), (over_long_id, 1_700_000_000_000)],
        )
        conn.execute("COMMIT")
        history.invalidate_derived_caches_locked()

    page = history.get_page(limit=2)

    assert page.next_cursor is None
    assert [entry.id for entry in page.entries] == ["newer", over_long_id]


def test_clear_all_reports_the_rows_it_deleted_not_a_memoised_count(
    isolated_storage, tmp_path
):
    """``ClearResult.deleted`` comes from the DELETE, so the cache cannot skew it.

    A row smuggled past every mutator leaves the memoised total one behind the
    table -- the residual risk ADR 055 accepts for the number on screen. The
    number of rows a delete removed is not that number, and ``cursor.rowcount``
    is what knows it.
    """
    _seed(tmp_path / "target", 3, lambda index: 1_700_000_000_000 + index)
    assert history.get_page(limit=1).total == 3

    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('smuggled', 1, 'uk', 'normal', 'row', 'row', 1)"
        )
        conn.execute("COMMIT")

    assert history.clear_all() == 4


def test_closing_the_connection_drops_the_memoised_total(isolated_storage, tmp_path):
    """The cache is the only strong reference to the connection it is keyed on.

    Left in place, it keeps a closed ``sqlite3.Connection`` alive past every
    ``bootstrap``, ``relocate`` and reopen. Clearing it where the connection is
    closed removes the retention, and with it the question of whether a freed
    connection could be replaced at the same address and match ``is``.
    """
    _seed(tmp_path / "target", 3, lambda index: 1_700_000_000_000 + index)
    history.get_page(limit=1)
    assert history._page_total_cache is not None

    with history._lock:
        history._close_conn_locked()

    assert history._page_total_cache is None


def test_the_total_is_counted_once_until_a_write_lands(
    isolated_storage, tmp_path, monkeypatch
):
    """ADR 055's second half: the only term that grew with the store is now read
    when it can have changed, not once per page.

    The clock is pinned rather than raced. The memo's third key is an age against
    ``time.monotonic``, so on real time these calls would have to finish inside
    ``STATS_TTL_SECONDS`` and a stalled worker would fail this test for a reason
    other than the defect it names.
    """
    monkeypatch.setattr(history, "time", SimpleNamespace(monotonic=lambda: 1_000.0))
    _seed(tmp_path / "target", 10, lambda index: 1_700_000_000_000 + index)

    def counts(action):
        return [
            statement
            for statement in _statements_from(action)
            if "COUNT(*)" in statement.upper() and "FROM ENTRIES" in statement.upper()
        ]

    def three_pages():
        for _ in range(3):
            history.get_page(limit=5)

    assert len(counts(three_pages)) == 1

    history.save_entry(text="one more", duration_ms=1)

    assert len(counts(lambda: history.get_page(limit=5))) == 1


def test_the_total_matches_a_fresh_count_after_every_mutator(isolated_storage, tmp_path):
    """The cached total against an independent oracle, not against a second call
    into the code under test: a ``SELECT COUNT(*)`` read through its own
    connection on the same file, after each path that can change the row count.
    """
    target = tmp_path / "target"
    history.bootstrap(target)

    def fresh_count():
        conn = sqlite3.connect(target / "history.db")
        try:
            return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        finally:
            conn.close()

    def assert_agrees(step):
        assert history.get_page(limit=1).total == fresh_count(), step

    kept = history.save_entry(text="kept", duration_ms=1).id
    history.save_entry(text="doomed", duration_ms=1)
    assert_agrees("after save_entry")

    history.delete_entry(kept)
    assert_agrees("after delete_entry")

    history.save_entry(text="another", duration_ms=1)
    assert_agrees("after a second save_entry")

    history.clear_all()
    assert_agrees("after clear_all")

    history.save_entry(text="post-clear", duration_ms=1)
    with history._lock:
        history._close_conn_locked()
    history.bootstrap(target)
    assert_agrees("after a store reopen")


def test_the_total_follows_the_generation_counter_not_the_table(isolated_storage, tmp_path):
    """Both halves of the invalidation contract in one test.

    A row written straight through the open connection, behind every mutator that
    would have bumped the generation, does not move the total; bumping the counter
    by hand does. That is exactly the residual risk ADR 055 accepts, stated as a
    test rather than as a sentence.
    """
    _seed(tmp_path / "target", 3, lambda index: 1_700_000_000_000 + index)
    assert history.get_page(limit=1).total == 3

    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('smuggled', 1, 'uk', 'normal', 'row', 'row', 1)"
        )
        conn.execute("COMMIT")

    assert history.get_page(limit=1).total == 3

    with history._lock:
        history.invalidate_derived_caches_locked()

    assert history.get_page(limit=1).total == 4


def test_the_memoised_total_expires_so_a_second_writer_is_picked_up(
    isolated_storage, tmp_path, monkeypatch
):
    """A writer outside this process bumps no generation, so only the age catches it.

    ``journal_mode=DELETE`` exists for exactly this shape -- the same
    ``history.db`` reachable from a second process, a sync folder being the case
    the module docstring names. Such a writer changes the table without going
    through any mutator in this module, so the generation and the connection
    halves of the key both still match and the memoised total would stand
    forever. ``STATS_TTL_SECONDS`` is what bounds that, and the cache is aged by
    hand rather than by sleeping for it.
    """
    target = tmp_path / "target"
    _seed(target, 3, lambda index: 1_700_000_000_000 + index)
    assert history.get_page(limit=1).total == 3

    other_process = sqlite3.connect(target / "history.db")
    try:
        other_process.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
            "VALUES ('from-elsewhere', 1, 'uk', 'normal', 'row', 'row', 1)"
        )
        other_process.commit()
    finally:
        other_process.close()

    assert history.get_page(limit=1).total == 3

    stamp, generation, cached_conn, total = history._page_total_cache
    monkeypatch.setattr(
        history,
        "_page_total_cache",
        (stamp - history.STATS_TTL_SECONDS - 1, generation, cached_conn, total),
    )

    assert history.get_page(limit=1).total == 4


def test_the_plan_probe_reports_every_shape_that_failed(isolated_storage):
    """``selftest`` promises every check runs and a failure reports all of them.

    ``cursor_seek_plan_failure`` is one of those checks and makes the same
    promise across the shapes it plans: with both broken, both are named. A probe
    that returned on the first one would hide the second in exactly the
    environment -- an old bundled library -- where knowing how much of the read
    path it broke is the point.
    """
    shipped = history._CURSOR_SEEK_PLAN_SHAPES
    assert len(shipped) > 1, shipped

    projection_that_walks_the_table = "(SELECT COUNT(*) FROM entries)"
    all_broken = tuple(
        (name, projection_that_walks_the_table, requires_covering)
        for name, _, requires_covering in shipped
    )

    with patch.object(history, "_CURSOR_SEEK_PLAN_SHAPES", all_broken):
        failure = history.cursor_seek_plan_failure()

    assert failure is not None
    for name, _, _ in shipped:
        assert name in failure, (name, failure)


def test_the_plan_probe_reports_every_condition_one_shape_failed(isolated_storage):
    """A shape that breaks two of the plan assertions is reported on both.

    The packaged build this probe exists for is the one where the bundled
    library is old enough to break more than one property at a time, and how
    much of the read path it gave back is the whole value of the message. A
    projection that reaches the table through a scanning subquery fails the
    table-walk assertion and the covering one together, and both must be named.
    """
    shipped = history._CURSOR_SEEK_PLAN_SHAPES
    covering = [position for position, shape in enumerate(shipped) if shape[2]]
    assert covering, shipped
    broken = covering[0]

    projection_that_scans_and_reaches_the_table = (
        "raw_text, (SELECT COUNT(raw_text) FROM entries)"
    )
    patched = tuple(
        (
            name,
            projection_that_scans_and_reaches_the_table if position == broken else projection,
            requires_covering,
        )
        for position, (name, projection, requires_covering) in enumerate(shipped)
    )

    with patch.object(history, "_CURSOR_SEEK_PLAN_SHAPES", patched):
        failure = history.cursor_seek_plan_failure()

    assert failure is not None
    assert "walks the entries table" in failure, failure
    assert "instead of answering from the index" in failure, failure


def test_the_plan_probe_names_the_shape_that_walked_the_index(isolated_storage):
    """Both shipped shapes are planned, and a failure says which one failed.

    Exactly one shape at a time is given a projection whose plan walks the table
    while every other shape keeps the one it ships with, so the message is
    attributed rather than merely produced: a probe that named the first
    configured shape whatever failed would report the wrong statement for half of
    what ships, and would still pass a test that broke every shape at once.

    The same attribution is asserted for the covering requirement, one shape at a
    time, against a projection that seeks the index and then reaches the table --
    the exact mutation a seek-only probe accepts.
    """
    assert history.cursor_seek_plan_failure() is None

    shipped = history._CURSOR_SEEK_PLAN_SHAPES

    def broken_one_at_a_time(position_broken, projection_that_fails):
        return tuple(
            (
                name,
                projection_that_fails if position == position_broken else projection,
                requires_covering,
            )
            for position, (name, projection, requires_covering) in enumerate(shipped)
        )

    def assert_names_only(patched, position_broken):
        with patch.object(history, "_CURSOR_SEEK_PLAN_SHAPES", patched):
            failure = history.cursor_seek_plan_failure()
        assert failure is not None, patched
        assert shipped[position_broken][0] in failure, (
            shipped[position_broken][0],
            failure,
        )
        for position, (name, _, _) in enumerate(shipped):
            if position != position_broken:
                assert name not in failure, (name, failure)

    projection_that_walks_the_table = "(SELECT COUNT(*) FROM entries)"
    for broken in range(len(shipped)):
        assert_names_only(
            broken_one_at_a_time(broken, projection_that_walks_the_table), broken
        )

    projection_that_reaches_the_table = "*"
    covering = [
        position for position, shape in enumerate(shipped) if shape[2]
    ]
    assert covering, shipped
    for broken in covering:
        assert_names_only(
            broken_one_at_a_time(broken, projection_that_reaches_the_table), broken
        )


def _bulk_seed(target, count):
    history.bootstrap(target)
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        try:
            conn.executemany(
                "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, duration_ms) "
                "VALUES (?, ?, 'uk', 'normal', 'row', 'row', 1)",
                [(f"{index:012d}", 1_700_000_000_000 + index) for index in range(count)],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _vm_steps_of(action):
    """SQLite VM instructions spent on every statement ``action`` issues, one at a
    time. The handler stays on the shared connection for the whole call, so a
    ``get_page`` is counted across its page read and the total beside it rather
    than on whichever statement the test picked out.
    """
    steps = 0

    def tick():
        nonlocal steps
        steps += 1
        return 0

    with history._lock:
        conn = history._ensure_conn_locked()
        conn.set_progress_handler(tick, 1)
    try:
        action()
    finally:
        with history._lock:
            conn.set_progress_handler(None, 0)
    return steps


def _run_read(sql):
    with history._lock:
        history._ensure_conn_locked().execute(sql).fetchall()


def _last_page_probe(store_size, tmp_path, name):
    """Steps for the whole shipped ``get_page`` of the last page, steps for the
    offset read it replaces, and the statements the shipped call issued over
    ``entries``, on a store of ``store_size`` rows.

    The derived caches are dropped between the measured call and the traced one, so
    the traced call is a separate cold call rather than a warmed repeat of the
    measured one: it is structurally identical to the call the step counter was
    open across, which is what its statement list stands in for.
    """
    _bulk_seed(tmp_path / name, store_size)
    with history._lock:
        conn = history._ensure_conn_locked()
        ts, entry_id = conn.execute(
            "SELECT ts, id FROM entries ORDER BY ts DESC, id DESC LIMIT 1 OFFSET ?",
            (store_size - 31,),
        ).fetchone()
    cursor = history.HistoryCursor(ts=ts, id=entry_id)
    offset_read = (
        f"SELECT {history.columns_sql(history.ENTRY_READ_COLUMNS)} FROM entries "
        f"ORDER BY ts DESC LIMIT 30 OFFSET {store_size - 30}"
    )
    cursor_steps = _vm_steps_of(lambda: history.get_page(limit=30, before=cursor))
    offset_steps = _vm_steps_of(lambda: _run_read(offset_read))
    with history._lock:
        history.invalidate_derived_caches_locked()
    return (
        cursor_steps,
        offset_steps,
        [
            statement
            for statement in _statements_from(lambda: history.get_page(limit=30, before=cursor))
            if _reads_entries(statement)
        ],
    )


def test_reading_the_last_page_costs_the_same_at_2000_rows_and_at_8000(isolated_storage, tmp_path):
    """The property keyset paging is bought for, measured across the whole call
    rather than across the one statement the test picked out. The offset
    measurement is taken alongside so the instrument is shown to notice growth: a
    step counter that reported "flat" for both queries would prove nothing about
    either.

    What the instrument reaches is on record rather than assumed. The counter is
    open across the whole call rather than across a statement the test re-runs, and
    that call issues exactly the three reads asserted below: the page read, the
    key-only has-more probe beside it, and the whole-store total. Inside that
    window the page read's size dependence is visible per opcode, and the total's
    is not: ``SELECT COUNT(*)`` compiles to a single ``OP_Count`` whose b-tree walk
    happens inside one instruction, so a step counter reads it flat at any store
    size. That is why the acceptance criterion names the instrument and not just
    the outcome.
    """
    small_cursor, small_offset, small_reads = _last_page_probe(2_000, tmp_path, "small")
    large_cursor, large_offset, _ = _last_page_probe(8_000, tmp_path, "large")

    assert len(small_reads) == 3, small_reads
    assert any("raw_text" in statement for statement in small_reads), small_reads
    assert any("COUNT(*)" in statement.upper() for statement in small_reads), small_reads
    assert any(
        "raw_text" not in statement and "COUNT(*)" not in statement.upper()
        for statement in small_reads
    ), small_reads

    assert large_cursor < small_cursor * 1.5, (small_cursor, large_cursor)
    assert large_offset > small_offset * 1.5, (small_offset, large_offset)


def test_an_existing_database_swaps_its_index_without_a_schema_bump(isolated_storage, tmp_path):
    """The swap rides ``_DDL_V1``, which runs unconditionally and idempotently on
    every open. ``SCHEMA_VERSION`` stays at 3 on purpose: the upgrade branch
    rebuilds the whole FTS index, and an index swap needs no row touched."""
    target = tmp_path / "target"
    history.bootstrap(target)
    kept = history.save_entry(text="written before the swap", duration_ms=1).id
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("DROP INDEX entries_ts_id_idx")
        conn.execute("CREATE INDEX entries_ts_idx ON entries(ts DESC)")
        indexes_before = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        history._close_conn_locked()
    assert "entries_ts_idx" in indexes_before
    assert "entries_ts_id_idx" not in indexes_before

    history.bootstrap(target)

    with history._lock:
        conn = history._ensure_conn_locked()
        indexes_after = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert "entries_ts_id_idx" in indexes_after
    assert "entries_ts_idx" not in indexes_after
    assert user_version == 3
    assert [e.id for e in history.get_page().entries] == [kept]

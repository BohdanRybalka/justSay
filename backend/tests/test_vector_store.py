"""Vector store — v3 DDL/migration, dim-guard trigger, model-switch wipe,
cascade-delete, relocate integrity, backfill resumability + clamping,
selftest (spec 003)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.core import history, vector_store
from app.core import words as words_module
from app.core.config import settings
from app.core.types import ProviderMode


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)
    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()


class _FakeProvider:
    def __init__(self, model_name: str, vector: list[float] | None = None, fail: bool = False):
        self.model_name = model_name
        self._vector = vector or [0.1, 0.2, 0.3]
        self._fail = fail

    async def embed(self, text: str) -> list[float]:
        if self._fail:
            raise RuntimeError("provider outage")
        return self._vector


def _rowid(conn: sqlite3.Connection, entry_id: str) -> int:
    return conn.execute("SELECT rowid FROM entries WHERE id = ?", (entry_id,)).fetchone()[0]


# --- DDL / migration -------------------------------------------------------

def test_schema_version_is_v3():
    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_v3_tables_and_trigger_exist_vec_entries_lazy():
    with history._lock:
        conn = history._ensure_conn_locked()
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            ).fetchall()
        }
    assert "embeddings_meta" in names
    assert "entry_embeddings" in names
    assert "entry_embeddings_dim_guard" in names
    # vec_entries is created lazily on first successful embed, not at bootstrap.
    assert "vec_entries" not in names


def test_migration_v1_to_v3(tmp_path):
    db_path = tmp_path / "history.db"
    with history._lock:
        history._close_conn_locked()
    db_path.unlink(missing_ok=True)
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.execute("PRAGMA user_version = 1")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            ).fetchall()
        }
    assert "embeddings_meta" in names
    assert "entry_embeddings" in names


def test_migration_v2_to_v3(tmp_path):
    db_path = tmp_path / "history.db"
    with history._lock:
        history._close_conn_locked()
    db_path.unlink(missing_ok=True)
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)
        raw.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        raw.execute("PRAGMA user_version = 2")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            ).fetchall()
        }
    assert "embeddings_meta" in names
    assert "entry_embeddings" in names


def test_crash_before_v3_pragma_retries(tmp_path):
    """A crash that ran the v3 DDL but never wrote user_version=3 must
    self-heal idempotently on the next boot (mirrors the existing v1->v2
    crash-safety test in test_history_sqlite.py)."""
    db_path = tmp_path / "history.db"
    with history._lock:
        history._close_conn_locked()
    db_path.unlink(missing_ok=True)
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)
        raw.executescript(vector_store._DDL_V3)  # simulate v3 DDL applied
        raw.execute("PRAGMA user_version = 2")  # crash window: version NOT bumped
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)  # must not raise

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_partial_migration_recovery_v3_tables_missing(tmp_path):
    """If a previous boot wrote user_version=3 but the v3 tables are
    missing (interrupted migration), _init_schema must recreate them via
    IF NOT EXISTS."""
    db_path = tmp_path / "history.db"
    with history._lock:
        history._close_conn_locked()
    db_path.unlink(missing_ok=True)
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(history._DDL_V1)
        raw.executescript(history._DDL_V2)
        raw.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        raw.execute("PRAGMA user_version = 3")  # v3 tables never actually created
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            ).fetchall()
        }
    assert "embeddings_meta" in names
    assert "entry_embeddings" in names
    assert "entry_embeddings_dim_guard" in names


# --- dim-guard trigger -----------------------------------------------------

def test_dim_guard_trigger_rejects_mismatched_dim():
    e = history.save_entry(text="hello", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO entry_embeddings(entry_id, model, dim, created_ts) "
                "VALUES (?, ?, ?, ?)",
                (e.id, "text-embedding-004", 5, 1),
            )


def test_dim_guard_trigger_allows_matching_dim():
    e = history.save_entry(text="hello", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        conn.execute(
            "INSERT INTO entry_embeddings(entry_id, model, dim, created_ts) VALUES (?, ?, ?, ?)",
            (e.id, "text-embedding-004", 3, 1),
        )


# --- ensure_vec_table_locked: model-switch wipe ----------------------------

def test_ensure_vec_table_idempotent_when_unchanged():
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        # must not raise
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        meta = conn.execute(
            "SELECT provider, model, dim FROM embeddings_meta WHERE id=1"
        ).fetchone()
    assert tuple(meta) == ("cloud", "text-embedding-004", 3)


def test_ensure_vec_table_wipes_on_model_switch():
    e1 = history.save_entry(text="alpha searchable text", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e1.id, _rowid(conn, e1.id), [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )
        count_before = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert count_before == 1

    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "local", "nomic-embed-text", 5)
        count_after = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
        meta = conn.execute(
            "SELECT provider, model, dim FROM embeddings_meta WHERE id=1"
        ).fetchone()
    assert count_after == 0
    assert tuple(meta) == ("local", "nomic-embed-text", 5)

    # Existing FTS5 keyword search is provably unaffected by the vector wipe.
    fts_hits = words_module.search_history("alpha", limit=5)
    assert len(fts_hits) == 1


@pytest.mark.asyncio
async def test_provider_switch_via_embed_background_wipes_old_embeddings():
    """Seeds embeddings under one model, flips the resolved provider,
    triggers a background embed, and asserts the old rows are gone and
    embeddings_meta reflects the new model. mode=fts still returns the
    pre-wipe entries."""
    e = history.save_entry(text="alpha searchable text", duration_ms=1)

    provider_a = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])
    with (
        patch.object(settings.stt, "mode", ProviderMode.CLOUD),
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(return_value=(provider_a, None)),
        ),
    ):
        await vector_store.embed_entry_background(e.id, "alpha searchable text")

    with history._lock:
        conn = history._ensure_conn_locked()
        meta_before = conn.execute(
            "SELECT provider, model, dim FROM embeddings_meta WHERE id=1"
        ).fetchone()
    assert tuple(meta_before) == ("cloud", "gemini/text-embedding-004", 3)

    provider_b = _FakeProvider("ollama/nomic-embed-text", vector=[1.0, 2.0, 3.0, 4.0, 5.0])
    with (
        patch.object(settings.stt, "mode", ProviderMode.LOCAL),
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(return_value=(provider_b, None)),
        ),
    ):
        await vector_store.embed_entry_background(e.id, "alpha searchable text")

    with history._lock:
        conn = history._ensure_conn_locked()
        count_after = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
        meta_after = conn.execute(
            "SELECT provider, model, dim FROM embeddings_meta WHERE id=1"
        ).fetchone()
    # Wiped-then-reinserted: exactly the current entry, embedded under the new model.
    assert count_after == 1
    assert tuple(meta_after) == ("local", "ollama/nomic-embed-text", 5)

    fts_hits = words_module.search_history("alpha", limit=5)
    assert len(fts_hits) == 1


# --- cascade delete: no new code in history.py's mutators ------------------

def test_delete_entry_cascades_to_entry_embeddings():
    e = history.save_entry(text="cascade me", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e.id, _rowid(conn, e.id), [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )

    history.delete_entry(e.id)  # calls ONLY history.delete_entry — no vector_store call

    with history._lock:
        conn = history._ensure_conn_locked()
        row = conn.execute("SELECT 1 FROM entry_embeddings WHERE entry_id = ?", (e.id,)).fetchone()
    assert row is None


def test_clear_all_cascades_to_entry_embeddings():
    e = history.save_entry(text="cascade me too", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e.id, _rowid(conn, e.id), [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )

    history.clear_all()  # calls ONLY history.clear_all — no vector_store call

    with history._lock:
        conn = history._ensure_conn_locked()
        count = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert count == 0


def test_delete_entry_removes_vec_entries_row_via_trigger():
    e = history.save_entry(text="vec cascade", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        rowid = _rowid(conn, e.id)
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e.id, rowid, [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )

    history.delete_entry(e.id)

    with history._lock:
        conn = history._ensure_conn_locked()
        row = conn.execute("SELECT 1 FROM vec_entries WHERE rowid = ?", (rowid,)).fetchone()
    assert row is None


# --- relocate integrity -----------------------------------------------------

def test_relocate_preserves_embeddings_and_semantic_search(tmp_path):
    e1 = history.save_entry(text="close match alpha", duration_ms=1)
    e2 = history.save_entry(text="totally different beta", duration_ms=1)

    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e1.id, _rowid(conn, e1.id), [1.0, 0.0, 0.0], "cloud", "text-embedding-004"
        )
        vector_store.insert_embedding(
            conn, e2.id, _rowid(conn, e2.id), [0.0, 1.0, 0.0], "cloud", "text-embedding-004"
        )

    new_dir = tmp_path / "new"
    res, _ = history.relocate(new_dir)
    assert res == history.RelocateResult.MOVED

    with history._lock:
        conn = history._ensure_conn_locked()
        rows = vector_store.query_similar(conn, [1.0, 0.0, 0.0], 1)
    assert len(rows) == 1
    assert rows[0]["id"] == e1.id


# --- backfill resumability + clamping ---------------------------------------

@pytest.mark.asyncio
async def test_backfill_resumable_across_two_calls():
    for i in range(5):
        history.save_entry(text=f"resume entry {i}", duration_ms=1)
    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        first = await vector_store.backfill_batch(2)
        assert first.processed == 2
        assert first.remaining == 3

        # Simulate the process being killed between calls: close + reopen
        # the connection. "already has an entry_embeddings row" IS the
        # resume cursor — no separate progress table.
        with history._lock:
            history._close_conn_locked()

        second = await vector_store.backfill_batch(2)
        assert second.processed == 2
        assert second.remaining == 1

        third = await vector_store.backfill_batch(2)
        assert third.processed == 1
        assert third.remaining == 0

    with history._lock:
        conn = history._ensure_conn_locked()
        total_embedded = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert total_embedded == 5


@pytest.mark.asyncio
async def test_backfill_batch_size_clamped_to_200():
    for i in range(205):
        history.save_entry(text=f"entry-{i}", duration_ms=1)
    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(99999)

    assert result.processed == 200
    assert result.remaining == 5
    assert vector_store.BACKFILL_BATCH_MAX == 200


@pytest.mark.asyncio
async def test_backfill_noop_when_disabled():
    history.save_entry(text="x", duration_ms=1)
    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(None, "disabled"))
    ):
        result = await vector_store.backfill_batch(50)
    assert result.processed == 0
    assert result.remaining == 1


# --- embeddings_status -------------------------------------------------------

@pytest.mark.asyncio
async def test_embeddings_status_reports_counts_and_provider():
    history.save_entry(text="a", duration_ms=1)
    e2 = history.save_entry(text="b", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e2.id, _rowid(conn, e2.id), [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )

    fake = _FakeProvider("gemini/text-embedding-004")
    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        status = await vector_store.embeddings_status()

    assert status.available is True
    assert status.indexed == 1
    assert status.total == 2
    assert status.provider == "gemini/text-embedding-004"


@pytest.mark.asyncio
async def test_embeddings_status_unavailable_when_vec_not_loaded():
    with patch.object(history, "_vec_available", False):
        status = await vector_store.embeddings_status()
    assert status.available is False
    assert status.reason == vector_store.VEC_EXTENSION_UNAVAILABLE_DETAIL


# --- embed_entry_background: best-effort contract ---------------------------

@pytest.mark.asyncio
async def test_embed_entry_background_never_raises_on_provider_failure():
    e = history.save_entry(text="will fail", duration_ms=1)
    failing = _FakeProvider("gemini/text-embedding-004", fail=True)
    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(failing, None))
    ):
        await vector_store.embed_entry_background(e.id, "will fail")  # must not raise

    with history._lock:
        conn = history._ensure_conn_locked()
        count = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert count == 0


@pytest.mark.asyncio
async def test_embed_entry_background_skips_deleted_entry():
    """If the entry was deleted before the background task ran, insert is
    skipped instead of raising (rowid lookup returns None)."""
    fake = _FakeProvider("gemini/text-embedding-004")
    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        await vector_store.embed_entry_background("does-not-exist", "text")  # must not raise


# --- selftest / --selftest-sqlite-vec ---------------------------------------

def test_selftest_ok():
    ok, msg = vector_store.selftest()
    assert ok is True
    assert msg == "ok"


def test_selftest_never_raises_on_broken_extension(monkeypatch):
    def boom(_conn):
        raise RuntimeError("extension load disabled on this platform")

    monkeypatch.setattr(vector_store.sqlite_vec, "load", boom)
    ok, msg = vector_store.selftest()
    assert ok is False
    assert "extension load disabled" in msg

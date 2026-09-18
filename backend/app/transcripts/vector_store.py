"""SQLite-side embeddings storage — ``vec0`` virtual table + ``entry_embeddings``.

Lives next to ``history.py`` rather than in ``app/embeddings/`` because it
shares that module's ``_lock`` and connection, which it imports at module level.
``app.transcripts.schema`` imports this module back, but only from inside a
function body, and ``app.embeddings`` is always lazy-imported here: both keep
the package importable in either order.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time

import sqlite_vec
from pydantic import BaseModel

from app.core.errors import ResourceUnavailableError
from app.transcripts import history

log = logging.getLogger(__name__)

BACKFILL_BATCH_MAX = 200

_DDL_V3 = """
CREATE TABLE IF NOT EXISTS embeddings_meta (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL CHECK (dim > 0)
);
CREATE TABLE IF NOT EXISTS entry_embeddings (
  entry_id TEXT PRIMARY KEY REFERENCES entries(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL CHECK (dim > 0),
  created_ts INTEGER NOT NULL
);
CREATE TRIGGER IF NOT EXISTS entry_embeddings_dim_guard
BEFORE INSERT ON entry_embeddings
WHEN NEW.dim != (SELECT dim FROM embeddings_meta WHERE id = 1)
BEGIN
  SELECT RAISE(ABORT, 'entry_embeddings.dim mismatch — provider/model changed without a wipe');
END;
"""

VEC_EXTENSION_UNAVAILABLE_DETAIL = (
    "sqlite-vec extension failed to load on this platform — semantic search unavailable"
)
NO_ENTRIES_EMBEDDED_DETAIL = (
    "No entries have been embedded yet — background indexing has not caught up"
)


class SemanticSearchUnavailableError(ResourceUnavailableError):
    """Raised by ``search.search_history_semantic``, silenced by its caller (ADR 010).

    It declares no ``status_code`` or ``code`` of its own, so it answers
    ``resource_unavailable`` with 503 the day something does route it to one.
    """


class BackfillResult(BaseModel):
    processed: int
    remaining: int


def recreate_delete_trigger_locked(conn: sqlite3.Connection) -> None:
    """Caller MUST hold ``history._lock``. Re-declares ``entries_ad_vec``.

    A trigger ON ``entries``, so dropping that table takes it with it and a deleted
    transcript would leave its vector behind for a later entry to inherit, SQLite
    reusing the rowid. A no-op until ``vec_entries`` exists.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_entries'"
    ).fetchone()
    if not exists:
        return
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS entries_ad_vec AFTER DELETE ON entries BEGIN "
        "DELETE FROM vec_entries WHERE rowid = old.rowid; END"
    )


def ensure_vec_table_locked(conn: sqlite3.Connection, provider: str, model: str, dim: int) -> None:
    """Caller MUST hold ``history._lock``. Prepares ``vec_entries`` for a model.

    A no-op if ``(provider, model)`` already matches ``embeddings_meta``; otherwise
    wipes and recreates at the new dimension, writing ``embeddings_meta`` LAST.
    """
    row = conn.execute(
        "SELECT provider, model, dim FROM embeddings_meta WHERE id = 1"
    ).fetchone()
    if row is not None and (row["provider"], row["model"]) == (provider, model):
        return

    if row is not None:
        conn.execute("DROP TABLE IF EXISTS vec_entries")
        conn.execute("DELETE FROM entry_embeddings")

    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_entries USING vec0(embedding float[{dim}])"
    )
    recreate_delete_trigger_locked(conn)
    conn.execute(
        "INSERT INTO embeddings_meta(id, provider, model, dim) VALUES (1, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "provider=excluded.provider, model=excluded.model, dim=excluded.dim",
        (provider, model, dim),
    )


def insert_embedding(
    conn: sqlite3.Connection,
    entry_id: str,
    rowid: int,
    vector: list[float],
    provider: str,
    model: str,
) -> None:
    """Caller MUST hold ``history._lock``. Writes one vector and its metadata.

    ``ensure_vec_table_locked`` must already have run for ``(provider, model)``.
    ``vec0`` rejects ``INSERT OR REPLACE``, so the vector is DELETEd then INSERTed.
    """
    dim = len(vector)
    packed = sqlite_vec.serialize_float32(vector)
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM vec_entries WHERE rowid = ?", (rowid,))
        conn.execute("INSERT INTO vec_entries(rowid, embedding) VALUES (?, ?)", (rowid, packed))
        conn.execute(
            "INSERT OR REPLACE INTO entry_embeddings(entry_id, model, dim, created_ts) "
            "VALUES (?, ?, ?, ?)",
            (entry_id, model, dim, int(time.time() * 1000)),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def query_similar(
    conn: sqlite3.Connection, query_vector: list[float], limit: int
) -> list[sqlite3.Row]:
    """Caller MUST hold ``history._lock``. Assumes ``vec_entries`` exists."""
    packed = sqlite_vec.serialize_float32(query_vector)
    return conn.execute(
        "SELECT e.*, distance FROM vec_entries "
        "JOIN entries e ON e.rowid = vec_entries.rowid "
        "WHERE embedding MATCH ? AND k = ? "
        "ORDER BY distance LIMIT ?",
        (packed, limit, limit),
    ).fetchall()


async def embed_entry_background(entry_id: str, text: str) -> None:
    """``BackgroundTasks`` entrypoint — runs after the response is sent.

    Best-effort by contract and never raises: embeddings disabled returns at
    ``debug``, and every other failure is caught and logged at ``warning``.
    """
    if not history._vec_available:
        log.debug("sqlite-vec unavailable — skipping background embed for %s", entry_id)
        return

    try:
        from app.core.config import settings
        from app.embeddings import resolve_embedding_provider

        provider, _reason = await resolve_embedding_provider(
            settings.stt, settings.embeddings
        )
        if provider is None:
            log.debug("Embeddings disabled — skipping background embed for %s", entry_id)
            return

        provider_id = settings.stt.mode.value
        vector = await provider.embed(text)
        with history._lock:
            conn = history._ensure_conn_locked()
            ensure_vec_table_locked(conn, provider_id, provider.model_name, len(vector))
            row = conn.execute("SELECT rowid FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                return
            insert_embedding(conn, entry_id, row[0], vector, provider_id, provider.model_name)
    except Exception:
        log.warning("Background embedding failed for entry %s", entry_id, exc_info=True)


async def backfill_batch(batch_size: int) -> BackfillResult:
    """Embed up to ``batch_size`` (clamped to ``[1, BACKFILL_BATCH_MAX]``)
    not-yet-embedded entries, oldest first. Resumable: "already has an
    ``entry_embeddings`` row" IS the resume cursor — no separate progress
    table.
    """
    clamped = max(1, min(int(batch_size), BACKFILL_BATCH_MAX))

    with history._lock:
        conn = history._ensure_conn_locked()
        rows = conn.execute(
            "SELECT rowid, id, cleaned_text FROM entries "
            "WHERE id NOT IN (SELECT entry_id FROM entry_embeddings) "
            "ORDER BY ts ASC LIMIT ?",
            (clamped,),
        ).fetchall()

    processed = 0
    if rows and history._vec_available:
        from app.core.config import settings
        from app.embeddings import resolve_embedding_provider

        provider, _reason = await resolve_embedding_provider(
            settings.stt, settings.embeddings
        )
        if provider is not None:
            provider_id = settings.stt.mode.value
            for row in rows:
                try:
                    vector = await provider.embed(row["cleaned_text"])
                    with history._lock:
                        conn = history._ensure_conn_locked()
                        ensure_vec_table_locked(conn, provider_id, provider.model_name, len(vector))
                        insert_embedding(
                            conn, row["id"], row["rowid"], vector, provider_id, provider.model_name
                        )
                    processed += 1
                except Exception:
                    log.warning("Backfill embed failed for entry %s", row["id"], exc_info=True)

    with history._lock:
        conn = history._ensure_conn_locked()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE id NOT IN (SELECT entry_id FROM entry_embeddings)"
        ).fetchone()[0]

    return BackfillResult(processed=processed, remaining=remaining)



_INDEXER_BATCH_SIZE = 20
_INDEXER_PACING_SECONDS = 1.5
_indexer_lock = asyncio.Lock()


async def run_background_indexer() -> None:
    """Silently drains the not-yet-embedded backlog. Never raises, never awaited.

    Nudged at startup and after a dictation (ADR 010), and serialized on
    ``_indexer_lock``, so overlapping nudges collapse into one active sweep.
    """
    if not history._vec_available:
        return
    try:
        async with _indexer_lock:
            while True:
                result = await backfill_batch(_INDEXER_BATCH_SIZE)
                if result.remaining == 0:
                    return
                if result.processed == 0:
                    log.debug(
                        "Background indexer stalled: %d entries remain unembedded, "
                        "provider unavailable or erroring", result.remaining,
                    )
                    return
                await asyncio.sleep(_INDEXER_PACING_SECONDS)
    except Exception:
        log.warning("Background indexer sweep failed unexpectedly", exc_info=True)


def _row_value_version_failure() -> str | None:
    """The bundled library is new enough to parse ``(ts, id) < (?, ?)`` at all.

    Row values arrived in SQLite 3.15.0; below that the history page read is a
    syntax error. Checked here rather than at startup, which it would take down.
    """
    if sqlite3.sqlite_version_info >= history.ROW_VALUE_MIN_SQLITE_VERSION:
        return None
    wanted = ".".join(str(part) for part in history.ROW_VALUE_MIN_SQLITE_VERSION)
    return (
        f"SQLite {sqlite3.sqlite_version} predates row-value support "
        f"(needs >= {wanted}), so history paging cannot run"
    )


def _vec_extension_failure() -> str | None:
    """sqlite-vec loads and answers a KNN query inside this build.

    Opens its own in-memory connection, so it works before history has ever
    bootstrapped, and queries an inserted vector back out of a 3-dim ``vec0``.
    """
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("CREATE VIRTUAL TABLE vt USING vec0(embedding float[3])")
            conn.execute(
                "INSERT INTO vt(rowid, embedding) VALUES (1, ?)",
                (sqlite_vec.serialize_float32([1.0, 2.0, 3.0]),),
            )
            row = conn.execute(
                "SELECT rowid FROM vt WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32([1.0, 2.0, 3.0]), 1),
            ).fetchone()
        finally:
            conn.close()
        if row is None or row[0] != 1:
            return "KNN query did not return the inserted row"
        return None
    except Exception as e:
        return str(e)


def _cursor_seek_plan_failure() -> str | None:
    """The bundled planner answers the cursor predicate with a seek, not a walk."""
    try:
        return history.cursor_seek_plan_failure()
    except Exception as e:
        return f"cursor seek probe failed: {e}"


def selftest() -> tuple[bool, str]:
    """``--selftest-sqlite-vec`` backend. Never raises.

    Assertions only a packaged build can answer: the library parses row values, its
    planner seeks ``entries_ts_id_idx``, sqlite-vec loads. **Every check runs.**
    """
    checks = (
        _row_value_version_failure,
        _cursor_seek_plan_failure,
        _vec_extension_failure,
    )
    try:
        outcomes = [check() for check in checks]
    except Exception as e:
        return False, f"a selftest check raised: {e}"
    failures = [outcome for outcome in outcomes if outcome is not None]
    if failures:
        return False, "; ".join(failures)
    return True, "ok"


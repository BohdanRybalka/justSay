"""SQLite-side embeddings storage — ``vec0`` virtual table + ``entry_embeddings``.

Lives next to ``history.py`` (not ``app/embeddings/``) because it shares its
``_lock`` and connection — mirrors how ``words.py`` keeps its SQL narrowly
scoped and delegates provider selection to ``app.llm.get_llm_provider``.

Import direction: this module imports ``app.core.history`` at module level
(needs ``_lock``/``_ensure_conn_locked``). ``history.py`` imports THIS
module back, but only via a lazy import inside ``_init_schema`` (function
body, not module top) — that keeps both modules importable in either order
without a circular-import crash at load time. ``app.embeddings`` is also
always lazy-imported here, same discipline ``words.py`` uses for
``app.llm.get_llm_provider``.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time

import sqlite_vec
from pydantic import BaseModel

from app.core import history

log = logging.getLogger(__name__)

BACKFILL_BATCH_MAX = 200

# Static v3 DDL — registered inside history.py's _init_schema. `vec_entries`
# itself is NOT here: it is created lazily on first successful embed (its
# dimension depends on the active provider, unknown until then). The
# `entries_ad_vec` delete-propagation trigger similarly lives in
# `ensure_vec_table_locked`, not here, since it only makes sense once
# `vec_entries` exists.
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

# Detail strings for SemanticSearchUnavailableError, consumed internally by
# words._semantic_lane (spec 017 / ADR 010) to decide how to log a degraded
# lane -- never surfaced to an HTTP client, since /history/search always
# returns 200 now. Kept as module constants so tests reference the exact
# same text.
VEC_EXTENSION_UNAVAILABLE_DETAIL = (
    "sqlite-vec extension failed to load on this platform — semantic search unavailable"
)
NO_ENTRIES_EMBEDDED_DETAIL = (
    "No entries have been embedded yet — background indexing has not caught up"
)


class SemanticSearchUnavailableError(Exception):
    """Raised by ``words.search_history_semantic``. Caught and silenced by
    ``words._semantic_lane`` (spec 017 / ADR 010) -- never reaches the
    router or an HTTP response; only logged at ``debug`` level."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class BackfillResult(BaseModel):
    processed: int
    remaining: int


def ensure_vec_table_locked(conn: sqlite3.Connection, provider: str, model: str, dim: int) -> None:
    """Caller MUST hold ``history._lock``.

    Idempotent no-op if ``(provider, model)`` already matches
    ``embeddings_meta``. Otherwise performs the full-wipe-on-model-change
    migration: drop `vec_entries`, delete all `entry_embeddings` rows,
    recreate `vec_entries` at the new dimension, and overwrite
    `embeddings_meta` LAST (only after the table exists) so a crash
    mid-wipe self-heals via `IF NOT EXISTS` + this same comparison on the
    next call.
    """
    row = conn.execute(
        "SELECT provider, model, dim FROM embeddings_meta WHERE id = 1"
    ).fetchone()
    if row is not None and (row["provider"], row["model"]) == (provider, model):
        return

    if row is not None:
        # Embeddings from a different model are not comparable vectors —
        # a partial mix would silently corrupt ranking. See ADR 001.
        conn.execute("DROP TABLE IF EXISTS vec_entries")
        conn.execute("DELETE FROM entry_embeddings")

    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_entries USING vec0(embedding float[{dim}])"
    )
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS entries_ad_vec AFTER DELETE ON entries BEGIN "
        "DELETE FROM vec_entries WHERE rowid = old.rowid; END"
    )
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
    """Caller MUST hold ``history._lock``. Assumes ``ensure_vec_table_locked``
    has already run for ``(provider, model)`` on this connection.

    ``vec0`` virtual tables do NOT support ``INSERT OR REPLACE`` (raises
    ``UNIQUE constraint failed`` — verified empirically against the pinned
    ``sqlite-vec==0.1.9``); a plain ``DELETE`` + ``INSERT`` is used instead.
    ``entry_embeddings`` is a normal table so ``INSERT OR REPLACE`` works
    there and also re-triggers the dim guard on the new row.
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
    """``BackgroundTasks`` entrypoint — runs AFTER the response is sent, so
    embedding latency structurally cannot land inside the request cycle
    (see ``pipeline.service.process_audio``).

    Best-effort by contract: the common case (embeddings disabled) returns
    quietly at ``debug`` level — must not warn-spam most users most of the
    time. Every other exception (provider network failure, malformed
    response, SQLite error) is caught and logged at ``warning``, never
    re-raised.
    """
    if not history._vec_available:
        log.debug("sqlite-vec unavailable — skipping background embed for %s", entry_id)
        return

    # Everything below (including resolve_embedding_provider itself, which
    # may make an HTTP call to Ollama) is inside the try block — the whole
    # point of this function is "never raise", not just "never raise after
    # a provider was found".
    try:
        from app.core.config import settings
        from app.embeddings import resolve_embedding_provider

        provider, _reason = await resolve_embedding_provider(
            settings.stt, settings.llm, settings.embeddings
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
                return  # entry deleted before the background task ran
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
    # Lock released here — backfill must not hold the write lock across N
    # network calls (embedding one row can take real wall-clock time).

    processed = 0
    if rows and history._vec_available:
        from app.core.config import settings
        from app.embeddings import resolve_embedding_provider

        provider, _reason = await resolve_embedding_provider(
            settings.stt, settings.llm, settings.embeddings
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


# --- Automatic background indexing (spec 017 / ADR 010) --------------------

_INDEXER_BATCH_SIZE = 20       # gentler than the old manual button's 50 — this now runs
_INDEXER_PACING_SECONDS = 1.5  # unattended, possibly stacked behind live dictation traffic
                                # competing for the same cloud/Ollama rate limits
_indexer_lock = asyncio.Lock()


async def run_background_indexer() -> None:
    """Silently drains the not-yet-embedded backlog. Nudged once at app
    startup and once per completed dictation (see ADR 010) -- never awaited
    by its callers, never blocks a request/response cycle. Serialized via
    _indexer_lock so overlapping nudges collapse into at most one active
    sweep; a nudge that arrives mid-sweep becomes a fast no-op once it
    acquires the lock and finds remaining == 0.

    Must never raise -- same "never raise" contract embed_entry_background
    already documents, for the same reason: both are BackgroundTasks/
    asyncio.create_task entrypoints with no caller able to observe or react
    to an exception raised here.
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


def selftest() -> tuple[bool, str]:
    """``--selftest-sqlite-vec`` backend. Never raises.

    Opens an in-memory connection independent of ``history``'s shared
    connection (this must work even if the sidecar has never bootstrapped
    history), loads the extension, creates a 3-dim ``vec0`` table, inserts
    and queries one vector, and asserts the inserted row comes back.
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
            # `k = ?` alone bounds the result set to one row; adding `LIMIT`
            # is forbidden by sqlite-vec ("Only LIMIT or 'k =?' can be
            # provided, not both") whenever SQLite pushes the LIMIT down to
            # the vec0 vtab -- which it does for this single-table query on
            # newer SQLite builds (e.g. the CI runner), though not for the
            # JOIN-wrapped `query_similar` above.
            row = conn.execute(
                "SELECT rowid FROM vt WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32([1.0, 2.0, 3.0]), 1),
            ).fetchone()
        finally:
            conn.close()
        if row is None or row[0] != 1:
            return False, "KNN query did not return the inserted row"
        return True, "ok"
    except Exception as e:
        return False, str(e)

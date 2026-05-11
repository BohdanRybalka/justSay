"""Transcript history — SQLite store at ``<output_dir>/history.db``.

Single-table v1 schema (`entries`). The directory is user-configurable via
``UserSettings.output_dir``. ``history.py`` deliberately does not import
``user_settings`` — the path is pushed in via ``bootstrap`` (lifespan)
or mutated by ``relocate`` (settings change). One-way dependency
(user_settings → history).

Concurrency model: a single shared sqlite3 connection per DB path, all
access serialised through the module-level ``_lock``. Per-connection
PRAGMAs (``foreign_keys=ON``, ``journal_mode=DELETE``, ``synchronous=FULL``)
are set in the connection factory so they cannot drift. ``journal_mode=DELETE``
(not WAL) is intentional — Plan 011 lets the user point ``output_dir`` at a
sync folder (Dropbox/iCloud/OneDrive) where WAL sidecar files would corrupt.
The trade is write-locks-readers; an in-memory stats cache (TTL 5 s,
invalidated on every mutation) absorbs the only realistic concurrent read
pressure (Words tab polling).

API field name is ``text``; the underlying ``entries`` table keeps the
historical ``raw_text``/``cleaned_text`` columns and writes the same value
into both for forward compatibility with old DBs.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger(__name__)

HISTORY_FILENAME = "history.db"
SCHEMA_VERSION = 2
STATS_TTL_SECONDS = 5.0

_lock = threading.Lock()
_output_dir: Path = Path.home() / ".justsay"
_conn: sqlite3.Connection | None = None
_stats_cache: tuple[float, "HistoryStats"] | None = None

# Mutation-listener registry — see register_mutation_listener.
# Listeners are invoked AFTER the lock is released, never under _lock,
# so a slow listener cannot block dictation. Direction is words → history;
# history must never import from words.
_mutation_listeners: list = []


class RelocateResult(str, Enum):
    MOVED = "moved"
    NEW_ALREADY_HAS_FILE = "new_already_has_file"
    NO_OLD_FILE = "no_old_file"
    FAILED = "failed"


class HistoryEntry(BaseModel):
    id: str
    timestamp: str
    language: str
    style: str
    text: str
    duration_ms: int
    model_name: str | None = None
    tokens_used: int | None = None
    audio_duration_seconds: float | None = None
    word_count: int | None = None


class HistoryStats(BaseModel):
    total_entries: int
    total_words: int
    total_audio_seconds: float
    today_words: int
    week_words: int
    by_language: dict[str, int]
    by_model: dict[str, int]


# --- Connection factory --------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the canonical PRAGMAs.

    ``timeout=0.2`` maps to PRAGMA busy_timeout = 200 ms.
    ``isolation_level=None`` disables Python's autocommit wrapping; every
    mutating function MUST issue explicit BEGIN/COMMIT.
    """
    conn = sqlite3.connect(
        db_path,
        timeout=0.2,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


_DDL_V1 = """
CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY,
  ts INTEGER NOT NULL,
  language TEXT NOT NULL,
  style TEXT NOT NULL CHECK (style IN ('normal', 'ai_prompt')),
  raw_text TEXT NOT NULL,
  cleaned_text TEXT NOT NULL,
  duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
  audio_duration_seconds REAL,
  word_count INTEGER,
  model_name TEXT,
  tokens_used INTEGER
);
CREATE INDEX IF NOT EXISTS entries_ts_idx ON entries(ts DESC);
"""

# Phase 2 — FTS5 full-text search over cleaned_text.
# External-content table (content='entries') means FTS doesn't duplicate
# the data; triggers keep its index in sync. IF NOT EXISTS everywhere so a
# partial migration (user_version=2 but FTS missing) can self-heal.
_DDL_V2 = """
CREATE VIRTUAL TABLE IF NOT EXISTS entry_fts USING fts5(
  cleaned_text,
  content='entries', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
  INSERT INTO entry_fts(rowid, cleaned_text) VALUES (new.rowid, new.cleaned_text);
END;
CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
  INSERT INTO entry_fts(entry_fts, rowid, cleaned_text) VALUES('delete', old.rowid, old.cleaned_text);
END;
CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
  INSERT INTO entry_fts(entry_fts, rowid, cleaned_text) VALUES('delete', old.rowid, old.cleaned_text);
  INSERT INTO entry_fts(rowid, cleaned_text) VALUES (new.rowid, new.cleaned_text);
END;
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    """Version-aware migrator. Run on every connection open.

    Branches:
      - fresh v0 / upgrade from v1 → run v2 DDL, rebuild FTS from rows,
        write user_version=2 LAST so a crash before the PRAGMA leaves a
        retry-able v1 (or pre-v1) state.
      - already at v2 → re-run v2 DDL (IF NOT EXISTS makes this idempotent)
        and probe FTS integrity; rebuild on OperationalError so a partial
        migration that left user_version=2 but no FTS table self-heals.
    """
    conn.executescript(_DDL_V1)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < SCHEMA_VERSION:
        conn.executescript(_DDL_V2)
        conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    else:
        # Idempotent recovery from a partially-applied v2: if the FTS
        # table or its triggers were dropped between boots (interrupted
        # migration, manual edit), the IF NOT EXISTS clauses recreate
        # them. A row-count probe then detects the empty-index case
        # (integrity-check trivially passes on an empty FTS) and a
        # divergence probe catches deeper corruption — either branch
        # ends in a rebuild.
        conn.executescript(_DDL_V2)
        # Probe whether the FTS index is in sync with `entries`.
        # `count(*)` on the FTS virtual table is forwarded to the
        # external content table, so we read the FTS-side count from
        # the shadow `entry_fts_docsize` table. That shadow name is
        # SQLite-internal; if it is ever missing (interrupted DDL,
        # SQLite version quirk), fall through to a full rebuild
        # rather than crashing the migrator.
        entries_rows = conn.execute("SELECT count(*) FROM entries").fetchone()[0]
        needs_rebuild = False
        try:
            fts_rows = conn.execute("SELECT count(*) FROM entry_fts_docsize").fetchone()[0]
            needs_rebuild = entries_rows != fts_rows
        except sqlite3.OperationalError as e:
            log.warning("FTS shadow probe unavailable (%s) — rebuilding", e)
            needs_rebuild = True
        if not needs_rebuild and entries_rows > 0:
            try:
                conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('integrity-check')")
            except sqlite3.OperationalError as e:
                log.warning("FTS integrity probe failed (%s) — rebuilding", e)
                needs_rebuild = True
        if needs_rebuild:
            conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")


# --- Public path API -----------------------------------------------------

def history_path() -> Path:
    """Lock-free read of the current history.db path."""
    return _output_dir / HISTORY_FILENAME


# --- Mutation-listener registry ------------------------------------------

def register_mutation_listener(listener) -> None:
    """Register a zero-argument callable invoked after every successful
    history mutation (save / delete / clear / relocate).

    Contract for listeners:
      - MUST be cheap. They run inline in the mutator's caller thread.
      - MUST NOT call any history mutator. Re-entrant calls would deadlock
        on _lock (listeners run outside _lock, but a mutator call would
        re-acquire it; correctness is the issue, not deadlock).
      - MAY raise — failures are caught and logged, the mutator stays
        successful.

    Used by ``app.core.words`` to invalidate the insights cache without
    creating a circular import (history exposes the registry; words
    depends on history, never the reverse).
    """
    _mutation_listeners.append(listener)


def _fire_mutation_listeners() -> None:
    """Invoke every registered listener. MUST be called OUTSIDE _lock."""
    for fn in _mutation_listeners:
        try:
            fn()
        except Exception:  # pragma: no cover — defensive
            log.exception("Mutation listener raised")


# --- Lifespan / bootstrap ------------------------------------------------

def init_output_dir(target: Path) -> None:
    """Test/internal helper. Real lifespan callers should use ``bootstrap``."""
    global _output_dir, _stats_cache
    with _lock:
        _close_conn_locked()
        _output_dir = target
        _stats_cache = None


def bootstrap(target: Path) -> None:
    """Lifespan helper: open the SQLite connection at ``<target>/history.db``."""
    global _output_dir, _conn, _stats_cache
    with _lock:
        _close_conn_locked()
        _output_dir = target
        _stats_cache = None
        target.mkdir(parents=True, exist_ok=True)
        _conn = _connect(target / HISTORY_FILENAME)
        _init_schema(_conn)


def _iso_to_epoch_ms(ts: str) -> int:
    """Parse ISO 8601 (Python 3.10-safe via Z→+00:00 shim) → unix epoch ms."""
    return int(round(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000))


def _epoch_ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# --- Relocate ------------------------------------------------------------

def relocate(new_dir: Path) -> tuple[RelocateResult, str | None]:
    """Move history.db to new_dir. Mutates _output_dir + closes/reopens
    connection inside the lock so a concurrent save_entry never sees a
    torn intermediate. Always invalidates _stats_cache.

    Phase 2: after a successful copy the FTS5 index is always rebuilt on
    the new connection BEFORE the point-of-no-return (``old_path.unlink``).
    Copied FTS shadow tables can desync if the copy interleaved with a
    write or if the source filesystem (Dropbox/iCloud) yielded a partial
    image. Rebuild is cheap on small DBs and is the integrity contract.
    """
    global _output_dir, _conn, _stats_cache
    fire_listeners = False
    try:
        with _lock:
            old_dir = _output_dir
            old_path = old_dir / HISTORY_FILENAME

            try:
                same = old_dir.resolve() == new_dir.resolve()
            except OSError:
                same = False

            if same:
                _stats_cache = None
                return RelocateResult.NO_OLD_FILE, None

            new_path = new_dir / HISTORY_FILENAME

            try:
                new_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                return RelocateResult.FAILED, f"Could not create target directory: {e}"

            if new_path.exists():
                _close_conn_locked()
                _output_dir = new_dir
                _conn = _connect(new_path)
                _init_schema(_conn)
                _stats_cache = None
                fire_listeners = True
                return RelocateResult.NEW_ALREADY_HAS_FILE, None

            if not old_path.exists():
                _close_conn_locked()
                _output_dir = new_dir
                _conn = _connect(new_path)
                _init_schema(_conn)
                _stats_cache = None
                fire_listeners = True
                return RelocateResult.NO_OLD_FILE, None

            # Move: close the old conn first so SQLite releases the lock on Windows.
            _close_conn_locked()
            new_conn: sqlite3.Connection | None = None
            try:
                shutil.copy2(old_path, new_path)
                if not _verify_db_row_count(old_path, new_path):
                    new_path.unlink(missing_ok=True)
                    _conn = _connect(old_path)
                    _init_schema(_conn)
                    _stats_cache = None
                    return RelocateResult.FAILED, "Verification failed: entry count mismatch"

                # Open and validate the new connection BEFORE deleting the old file.
                # If _connect raises, we still have old_path intact and rollback safely.
                new_conn = _connect(new_path)
                _init_schema(new_conn)
                # Phase 2: do not trust the copied FTS index — always rebuild
                # on the new path so search results are consistent with entries
                # right after the move.
                new_conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")

                # Point of no return — only after the new connection is healthy.
                old_path.unlink()
                _output_dir = new_dir
                _conn = new_conn
                new_conn = None  # ownership transferred
                _stats_cache = None
                log.info("Relocated history %s → %s", old_path, new_path)
                fire_listeners = True
                return RelocateResult.MOVED, None
            except (OSError, sqlite3.Error) as e:
                if new_conn is not None:
                    try:
                        new_conn.close()
                    except sqlite3.Error:
                        pass
                new_path.unlink(missing_ok=True)
                try:
                    _conn = _connect(old_path)
                    _init_schema(_conn)
                except sqlite3.Error:
                    _conn = None
                _stats_cache = None
                log.exception("Relocate failed: %s", e)
                return RelocateResult.FAILED, f"Move failed: {e}"
    finally:
        if fire_listeners:
            _fire_mutation_listeners()


# --- CRUD ----------------------------------------------------------------

def save_entry(
    text: str,
    duration_ms: int,
    language: str = "uk",
    style: str = "normal",
    model_name: str | None = None,
    tokens_used: int | None = None,
    audio_duration_seconds: float | None = None,
    word_count: int | None = None,
) -> HistoryEntry:
    """Append a new entry. ``text`` is written to both legacy columns for compat."""
    global _stats_cache
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = HistoryEntry(
        id=uuid.uuid4().hex[:12],
        timestamp=timestamp,
        language=language,
        style=style,
        text=text,
        duration_ms=duration_ms,
        model_name=model_name,
        tokens_used=tokens_used,
        audio_duration_seconds=audio_duration_seconds,
        word_count=word_count,
    )
    ts_ms = _iso_to_epoch_ms(timestamp)

    with _lock:
        conn = _ensure_conn_locked()
        conn.execute("BEGIN")
        try:
            conn.execute(
                "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, "
                "duration_ms, audio_duration_seconds, word_count, model_name, tokens_used) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id, ts_ms, entry.language, entry.style,
                    entry.text, entry.text, entry.duration_ms,
                    entry.audio_duration_seconds, entry.word_count,
                    entry.model_name, entry.tokens_used,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        _stats_cache = None

    _fire_mutation_listeners()
    return entry


def get_entries(limit: int = 50, offset: int = 0) -> list[HistoryEntry]:
    """Get history entries newest first. Returns full rows (existing API contract)."""
    with _lock:
        conn = _ensure_conn_locked()
        rows = conn.execute(
            "SELECT id, ts, language, style, raw_text, duration_ms, "
            "audio_duration_seconds, word_count, model_name, tokens_used "
            "FROM entries ORDER BY ts DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_row_to_entry(r) for r in rows]


def get_count() -> int:
    with _lock:
        conn = _ensure_conn_locked()
        return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


def delete_entry(entry_id: str) -> bool:
    global _stats_cache
    with _lock:
        conn = _ensure_conn_locked()
        conn.execute("BEGIN")
        try:
            cursor = conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            deleted = cursor.rowcount > 0
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        if deleted:
            _stats_cache = None

    if deleted:
        _fire_mutation_listeners()
    return deleted


def clear_all() -> int:
    global _stats_cache
    with _lock:
        conn = _ensure_conn_locked()
        count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        conn.execute("BEGIN")
        try:
            conn.execute("DELETE FROM entries")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        _stats_cache = None

    _fire_mutation_listeners()
    return count


def compute_stats(now: datetime | None = None) -> HistoryStats:
    """Aggregate via SQL with a 5 s in-memory cache (invalidated by all mutators)."""
    global _stats_cache
    if now is None:
        now = datetime.now(timezone.utc).astimezone()

    today = now.date()
    week_cutoff = now - timedelta(days=7)
    today_start_ms = int(round(datetime(today.year, today.month, today.day, tzinfo=now.tzinfo).timestamp() * 1000))
    week_cutoff_ms = int(round(week_cutoff.timestamp() * 1000))

    with _lock:
        cached = _stats_cache
        if cached is not None and (time.monotonic() - cached[0]) < STATS_TTL_SECONDS:
            return cached[1]

        conn = _ensure_conn_locked()
        agg = conn.execute(
            "SELECT COUNT(*), "
            "COALESCE(SUM(word_count), 0), "
            "COALESCE(SUM(audio_duration_seconds), 0.0), "
            "COALESCE(SUM(CASE WHEN ts >= ? THEN word_count ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN ts >= ? THEN word_count ELSE 0 END), 0) "
            "FROM entries",
            (today_start_ms, week_cutoff_ms),
        ).fetchone()
        total_entries, total_words, total_audio, today_words, week_words = agg

        by_language = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT language, COALESCE(SUM(word_count), 0) FROM entries "
                "WHERE language IS NOT NULL GROUP BY language"
            ).fetchall()
        }
        by_model = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT model_name, COALESCE(SUM(word_count), 0) FROM entries "
                "WHERE model_name IS NOT NULL GROUP BY model_name"
            ).fetchall()
        }

        stats = HistoryStats(
            total_entries=total_entries,
            total_words=total_words,
            total_audio_seconds=round(total_audio, 1),
            today_words=today_words,
            week_words=week_words,
            by_language=by_language,
            by_model=by_model,
        )
        _stats_cache = (time.monotonic(), stats)
        return stats


# --- Internals -----------------------------------------------------------

def _row_to_entry(row: sqlite3.Row) -> HistoryEntry:
    return HistoryEntry(
        id=row["id"],
        timestamp=_epoch_ms_to_iso(row["ts"]),
        language=row["language"],
        style=row["style"],
        text=row["raw_text"],
        duration_ms=row["duration_ms"],
        audio_duration_seconds=row["audio_duration_seconds"],
        word_count=row["word_count"],
        model_name=row["model_name"],
        tokens_used=row["tokens_used"],
    )


def _ensure_conn_locked() -> sqlite3.Connection:
    """Caller MUST hold ``_lock``. Lazy-opens the connection on demand
    (covers the case where init_output_dir was called by tests but
    bootstrap was not)."""
    global _conn
    if _conn is None:
        _conn = _connect(_output_dir / HISTORY_FILENAME)
        _init_schema(_conn)
    return _conn


def _close_conn_locked() -> None:
    """Caller MUST hold ``_lock``."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except sqlite3.Error:
            pass
        _conn = None


def _verify_db_row_count(src_db: Path, dst_db: Path) -> bool:
    """Compare entries row count between two SQLite files."""
    def count(p: Path) -> int:
        c = _connect(p)
        try:
            return c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        finally:
            c.close()
    try:
        return count(src_db) == count(dst_db)
    except sqlite3.Error:
        return False

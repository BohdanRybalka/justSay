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

import sqlite_vec
from pydantic import BaseModel

from app.core.app_paths import resolve_app_data_root

log = logging.getLogger(__name__)

HISTORY_FILENAME = "history.db"
SCHEMA_VERSION = 3
STATS_TTL_SECONDS = 5.0

_lock = threading.Lock()
_output_dir: Path | None = None
_conn: sqlite3.Connection | None = None
_stats_cache: tuple[float, HistoryStats] | None = None

_vec_available: bool = False
_vec_load_warned = False


class RelocateResult(str, Enum):
    MOVED = "moved"
    NEW_ALREADY_HAS_FILE = "new_already_has_file"
    NO_OLD_FILE = "no_old_file"
    FAILED = "failed"


class ConsolidateResult(str, Enum):
    CONSOLIDATED = "consolidated"
    NOT_NEEDED = "not_needed"
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

    global _vec_available, _vec_load_warned
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _vec_available = True
    except Exception as e:
        _vec_available = False
        if not _vec_load_warned:
            log.warning("sqlite-vec extension failed to load — semantic search disabled: %s", e)
            _vec_load_warned = True

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
  INSERT INTO entry_fts(entry_fts, rowid, cleaned_text)
  VALUES('delete', old.rowid, old.cleaned_text);
END;
CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
  INSERT INTO entry_fts(entry_fts, rowid, cleaned_text)
  VALUES('delete', old.rowid, old.cleaned_text);
  INSERT INTO entry_fts(rowid, cleaned_text) VALUES (new.rowid, new.cleaned_text);
END;
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    """Version-aware migrator. Run on every connection open.

    Branches:
      - fresh v0 / upgrade from v1 or v2 → run v2 DDL, rebuild FTS from
        rows, run v3 DDL (embeddings_meta + entry_embeddings — both start
        empty, no rows to replay), write user_version=3 LAST so a crash
        before the PRAGMA leaves a retry-able prior-version state.
      - already at v3 → re-run v2 DDL (IF NOT EXISTS makes this idempotent)
        and probe FTS integrity; rebuild on OperationalError so a partial
        migration that left user_version=3 but no FTS table self-heals.
        Also re-run v3 DDL (IF NOT EXISTS) so a partial migration that left
        user_version=3 but the embeddings tables missing self-heals too.
    """
    from app.transcripts import vector_store

    conn.executescript(_DDL_V1)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < SCHEMA_VERSION:
        conn.executescript(_DDL_V2)
        conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        conn.executescript(vector_store._DDL_V3)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    else:
        conn.executescript(_DDL_V2)
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

        conn.executescript(vector_store._DDL_V3)



def _resolve_output_dir() -> Path:
    """`_output_dir` if `bootstrap()`/`init_output_dir()`/`relocate()` has
    already set one, else a lazy fallback resolved fresh on every call --
    never cached at import time (ADR 014, AC 8a)."""
    return _output_dir if _output_dir is not None else resolve_app_data_root()


def history_path() -> Path:
    """Lock-free read of the current history.db path."""
    return _resolve_output_dir() / HISTORY_FILENAME



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



def relocate(new_dir: Path) -> tuple[RelocateResult, str | None]:
    """Move history.db to new_dir. Mutates _output_dir + closes/reopens
    connection inside the lock so a concurrent save_entry never sees a
    torn intermediate. Always invalidates _stats_cache.

    Phase 2: after a successful copy the FTS5 index is always rebuilt on
    the new connection BEFORE the point-of-no-return (``old_path.unlink``).
    Copied FTS shadow tables can desync if the copy interleaved with a
    write or if the source filesystem (Dropbox/iCloud) yielded a partial
    image. Rebuild is cheap on small DBs and is the integrity contract.

    Phase 3 (sqlite-vec): no new rebuild step is needed here. Unlike FTS5's
    external-content table, ``vec0`` has no rebuild/integrity command, and
    ``shutil.copy2`` is a raw byte-level file copy that preserves
    ``entries.rowid`` (and therefore every rowid-keyed ``vec_entries`` row)
    exactly. The ``_init_schema(new_conn)`` call below already re-attaches
    the v3 tables via ``IF NOT EXISTS`` — a no-op on a file that already
    has them.
    """
    global _output_dir, _conn, _stats_cache
    with _lock:
        old_dir = _resolve_output_dir()
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
            return RelocateResult.NEW_ALREADY_HAS_FILE, None

        if not old_path.exists():
            _close_conn_locked()
            _output_dir = new_dir
            _conn = _connect(new_path)
            _init_schema(_conn)
            _stats_cache = None
            return RelocateResult.NO_OLD_FILE, None

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

            new_conn = _connect(new_path)
            _init_schema(new_conn)
            new_conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")

            old_path.unlink()
            _output_dir = new_dir
            _conn = new_conn
            new_conn = None
            _stats_cache = None
            log.info("Relocated history %s → %s", old_path, new_path)
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


_ENTRY_COLUMNS = (
    "id",
    "ts",
    "language",
    "style",
    "raw_text",
    "cleaned_text",
    "duration_ms",
    "audio_duration_seconds",
    "word_count",
    "model_name",
    "tokens_used",
)


def _premigration_path(target_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    candidate = target_dir / f"{HISTORY_FILENAME}.premigration-{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = target_dir / f"{HISTORY_FILENAME}.premigration-{stamp}-{suffix}"
        suffix += 1
    return candidate


def consolidate_into(source_dir: Path, target_dir: Path) -> tuple[ConsolidateResult, str | None]:
    """Merge ``source_dir``'s history into ``target_dir`` and move the source aside.

    Used once, at startup, when ``output_dir`` was found inside the scratch
    directory (ADR 033). Deliberately NOT ``relocate()``: that returns
    ``NEW_ALREADY_HAS_FILE`` and adopts the target whenever a file exists
    there, which in the case this exists to repair is an *empty* database --
    it would hide every row behind a zero-row file. Merging by row makes an
    empty target harmless.

    Rows are copied with ``INSERT OR IGNORE`` on the existing
    ``id TEXT PRIMARY KEY``, so a repeated run is a no-op and no row is ever
    overwritten. Only columns present in *both* databases are copied, so an
    older source schema degrades to NULLs instead of raising. ``entry_fts``
    is filled by the ``entries_ai`` trigger and embeddings by
    ``run_background_indexer``, so no index is rebuilt here.

    The source file is renamed aside, never deleted. Pure with respect to
    module state: it opens its own connections and touches neither ``_conn``
    nor ``_output_dir``, so the caller must run it before ``bootstrap``.
    """
    source_path = source_dir / HISTORY_FILENAME
    target_path = target_dir / HISTORY_FILENAME

    if not source_path.exists():
        return ConsolidateResult.NOT_NEEDED, None

    try:
        same = source_dir.resolve() == target_dir.resolve()
    except OSError:
        same = False
    if same:
        return ConsolidateResult.NOT_NEEDED, None

    conn: sqlite3.Connection | None = None
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        conn = _connect(target_path)
        _init_schema(conn)
        conn.execute("ATTACH DATABASE ? AS source", (str(source_path),))
        source_columns = {row[1] for row in conn.execute("PRAGMA source.table_info(entries)")}
        if "id" not in source_columns:
            return ConsolidateResult.FAILED, "Source database has no entries table"

        shared = [name for name in _ENTRY_COLUMNS if name in source_columns]
        column_list = ", ".join(shared)
        conn.execute("BEGIN")
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO entries ({column_list}) "
            f"SELECT {column_list} FROM source.entries"
        )
        copied = cursor.rowcount
        conn.execute("COMMIT")
        conn.execute("DETACH DATABASE source")
    except (OSError, sqlite3.Error) as e:
        if conn is not None:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        log.exception("History consolidation failed: %s", e)
        return ConsolidateResult.FAILED, f"Consolidation failed: {e}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    try:
        kept = _premigration_path(target_dir)
        source_path.rename(kept)
    except OSError as e:
        log.warning("History merged but the source file could not be moved aside: %s", e)
        return ConsolidateResult.CONSOLIDATED, f"Source left in place: {e}"

    log.info("Consolidated %d history entries %s → %s; source kept at %s",
             copied, source_path, target_path, kept)
    return ConsolidateResult.CONSOLIDATED, None


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

    return count


def compute_stats(now: datetime | None = None) -> HistoryStats:
    """Aggregate via SQL with a 5 s in-memory cache (invalidated by all mutators)."""
    global _stats_cache
    if now is None:
        now = datetime.now(timezone.utc).astimezone()

    today = now.date()
    week_cutoff = now - timedelta(days=7)
    today_start = datetime(today.year, today.month, today.day, tzinfo=now.tzinfo)
    today_start_ms = int(round(today_start.timestamp() * 1000))
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
        _conn = _connect(_resolve_output_dir() / HISTORY_FILENAME)
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

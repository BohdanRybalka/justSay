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
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import sqlite_vec
from pydantic import BaseModel, Field

from app.core.app_paths import resolve_app_data_root

log = logging.getLogger(__name__)

HISTORY_FILENAME = "history.db"
SCHEMA_VERSION = 3
STATS_TTL_SECONDS = 5.0
HISTORY_LIMIT_MAX = 200
CURSOR_ID_MAX_LENGTH = 64
CURSOR_TS_MIN = -(2**63)
CURSOR_TS_MAX = 2**63 - 1
ROW_VALUE_MIN_SQLITE_VERSION = (3, 15)

_lock = threading.Lock()
_output_dir: Path | None = None
_conn: sqlite3.Connection | None = None
_stats_cache: tuple[float, HistoryStats] | None = None
_derived_generation = 0

_vec_available: bool = False
_vec_load_warned = False


class RelocateOutcome(str, Enum):
    MOVED = "moved"
    NEW_ALREADY_HAS_FILE = "new_already_has_file"
    NO_OLD_FILE = "no_old_file"
    FAILED = "failed"


class ConsolidateOutcome(str, Enum):
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


class HistoryCursor(BaseModel):
    """A position in the total order ``(ts DESC, id DESC)``, minted by the server.

    A cursor is not a count, so nothing under it moves when a row is inserted or
    deleted. ``id`` is only a tiebreaker: it makes the ordering total, which is
    what a position needs to identify one row. See ADR 053.

    Both halves are bounded on the model rather than only in the router's
    signature, for the reason ``_clamp_limit``'s docstring gives about ``limit``:
    a bound that lives only in a FastAPI signature is not a bound on the
    function. ``ts`` outside the signed 64-bit range SQLite stores an INTEGER in
    raises ``OverflowError`` out of the driver, so any caller that builds a
    cursor by hand gets a ``ValidationError`` here instead.
    """

    ts: int = Field(ge=CURSOR_TS_MIN, le=CURSOR_TS_MAX)
    id: str = Field(max_length=CURSOR_ID_MAX_LENGTH)


class HistoryPage(BaseModel):
    """One page, the total it is a page of, and where the next one starts.

    ``next_cursor`` is ``None`` on the last page. ``total`` is what the user
    reads on screen; it is no longer how the client decides whether more rows
    exist.
    """

    entries: list[HistoryEntry]
    total: int
    next_cursor: HistoryCursor | None = None


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
"""

_REPLACE_TS_INDEX_WITH_TS_ID_INDEX = """
CREATE INDEX IF NOT EXISTS entries_ts_id_idx ON entries(ts DESC, id DESC);
DROP INDEX IF EXISTS entries_ts_idx;
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

    ``_DDL_V1`` and ``_REPLACE_TS_INDEX_WITH_TS_ID_INDEX`` both run
    unconditionally and idempotently, which is how the ``entries_ts_idx`` ->
    ``entries_ts_id_idx`` swap reaches an existing database: ``SCHEMA_VERSION`` is
    deliberately not bumped for it, because the ``current < SCHEMA_VERSION``
    branch below rebuilds the whole FTS index, and an index swap needs no row
    touched. Nothing therefore *records* that a database has been swapped:
    ``user_version`` reads 3 either way, so no database can be asked whether the
    swap has happened -- the only way to observe it is to create the old index by
    hand and reopen the file. Accepted rather than fixed -- see ADR 053, "What it
    leaves unrecorded".

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
    conn.executescript(_REPLACE_TS_INDEX_WITH_TS_ID_INDEX)
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
    """`_output_dir` if `bootstrap()`/`relocate()` has already set one, else a
    lazy fallback resolved fresh on every call -- never cached at import time
    (ADR 014, AC 8a)."""
    return _output_dir if _output_dir is not None else resolve_app_data_root()


def history_path() -> Path:
    """Lock-free read of the current history.db path."""
    return _resolve_output_dir() / HISTORY_FILENAME


def bootstrap(target: Path) -> None:
    """Lifespan helper: open the SQLite connection at ``<target>/history.db``."""
    global _output_dir, _conn
    with _lock:
        _close_conn_locked()
        _output_dir = target
        invalidate_derived_caches_locked()
        target.mkdir(parents=True, exist_ok=True)
        _conn = _connect(target / HISTORY_FILENAME)
        _init_schema(_conn)


def _iso_to_epoch_ms(ts: str) -> int:
    """Parse ISO 8601 (Python 3.10-safe via Z→+00:00 shim) → unix epoch ms."""
    return int(round(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000))


def _epoch_ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()



def relocate(new_dir: Path) -> tuple[RelocateOutcome, str | None]:
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
    global _output_dir, _conn
    with _lock:
        old_dir = _resolve_output_dir()
        old_path = old_dir / HISTORY_FILENAME

        try:
            same = old_dir.resolve() == new_dir.resolve()
        except OSError:
            same = False

        if same:
            invalidate_derived_caches_locked()
            return RelocateOutcome.NO_OLD_FILE, None

        new_path = new_dir / HISTORY_FILENAME

        try:
            new_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return RelocateOutcome.FAILED, f"Could not create target directory: {e}"

        if new_path.exists():
            _output_dir = new_dir
            _reopen_conn_locked(new_dir)
            return RelocateOutcome.NEW_ALREADY_HAS_FILE, None

        if not old_path.exists():
            _output_dir = new_dir
            _reopen_conn_locked(new_dir)
            return RelocateOutcome.NO_OLD_FILE, None

        _close_conn_locked()
        new_conn: sqlite3.Connection | None = None
        try:
            shutil.copy2(old_path, new_path)
            if not _verify_db_row_count(old_path, new_path):
                new_path.unlink(missing_ok=True)
                _reopen_conn_locked(old_dir)
                return RelocateOutcome.FAILED, "Verification failed: entry count mismatch"

            new_conn = _connect(new_path)
            _init_schema(new_conn)
            new_conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")

            old_path.unlink()
            _output_dir = new_dir
            _conn = new_conn
            new_conn = None
            invalidate_derived_caches_locked()
            log.info("Relocated history %s → %s", old_path, new_path)
            return RelocateOutcome.MOVED, None
        except (OSError, sqlite3.Error) as e:
            if new_conn is not None:
                try:
                    new_conn.close()
                except sqlite3.Error:
                    pass
            new_path.unlink(missing_ok=True)
            try:
                _reopen_conn_locked(old_dir)
            except sqlite3.Error:
                _conn = None
                invalidate_derived_caches_locked()
            log.exception("Relocate failed: %s", e)
            return RelocateOutcome.FAILED, f"Move failed: {e}"


ENTRY_COLUMNS = (
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

ENTRY_READ_COLUMNS = tuple(c for c in ENTRY_COLUMNS if c != "cleaned_text")


def columns_sql(columns: Sequence[str], alias: str = "") -> str:
    """The column list for a SELECT or INSERT, optionally table-qualified.

    The result is interpolated into SQL, which sqlite cannot parameterise for
    identifiers, so nothing here is trusted: every name is checked against
    ``ENTRY_COLUMNS``, and ``alias`` -- the table alias, written without its
    dot -- must be a plain identifier. An ``entries`` column, qualified by at
    most one identifier, is the only thing this can emit; anything else is a
    ``ValueError`` rather than a query.
    """
    unknown = [name for name in columns if name not in ENTRY_COLUMNS]
    if unknown:
        raise ValueError(f"Not columns of the entries table: {unknown}")
    if alias and not alias.isidentifier():
        raise ValueError(f"Not a table alias: {alias!r}")
    qualifier = f"{alias}." if alias else ""
    return ", ".join(f"{qualifier}{name}" for name in columns)


def _premigration_path(target_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    candidate = target_dir / f"{HISTORY_FILENAME}.premigration-{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = target_dir / f"{HISTORY_FILENAME}.premigration-{stamp}-{suffix}"
        suffix += 1
    return candidate


def consolidate_into(source_dir: Path, target_dir: Path) -> tuple[ConsolidateOutcome, str | None]:
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
        return ConsolidateOutcome.NOT_NEEDED, None

    try:
        same = source_dir.resolve() == target_dir.resolve()
    except OSError:
        same = False
    if same:
        return ConsolidateOutcome.NOT_NEEDED, None

    conn: sqlite3.Connection | None = None
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        conn = _connect(target_path)
        _init_schema(conn)
        conn.execute("ATTACH DATABASE ? AS source", (str(source_path),))
        source_columns = {row[1] for row in conn.execute("PRAGMA source.table_info(entries)")}
        if "id" not in source_columns:
            return ConsolidateOutcome.FAILED, "Source database has no entries table"

        shared = [name for name in ENTRY_COLUMNS if name in source_columns]
        column_list = columns_sql(shared)
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
        return ConsolidateOutcome.FAILED, f"Consolidation failed: {e}"
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
        return ConsolidateOutcome.CONSOLIDATED, f"Source left in place: {e}"

    log.info("Consolidated %d history entries %s → %s; source kept at %s",
             copied, source_path, target_path, kept)
    return ConsolidateOutcome.CONSOLIDATED, None


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
                f"INSERT INTO entries({columns_sql(ENTRY_COLUMNS)}) "
                f"VALUES ({', '.join(':' + name for name in ENTRY_COLUMNS)})",
                {
                    "id": entry.id,
                    "ts": ts_ms,
                    "language": entry.language,
                    "style": entry.style,
                    "raw_text": entry.text,
                    "cleaned_text": entry.text,
                    "duration_ms": entry.duration_ms,
                    "audio_duration_seconds": entry.audio_duration_seconds,
                    "word_count": entry.word_count,
                    "model_name": entry.model_name,
                    "tokens_used": entry.tokens_used,
                },
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        invalidate_derived_caches_locked()

    return entry


def _clamp_limit(limit: int) -> int:
    """``get_page``'s own bound, applied before the router's is trusted.

    ``words.top_words`` and ``words.search_history`` each clamp their own against
    their own maximum rather than sharing this one. What they have in common is
    the reason, not the numbers: a bound that lives only in a FastAPI signature is
    not a bound on the function, and ``LIMIT -1`` materialises every row.
    """
    return max(1, min(int(limit), HISTORY_LIMIT_MAX))


_CURSOR_PAGE_WHERE = "WHERE (ts, id) < (:before_ts, :before_id) "
_CURSOR_PAGE_ORDER = "ORDER BY ts DESC, id DESC LIMIT :limit_plus_one"


def _entries_locked(
    conn: sqlite3.Connection, limit: int, before: HistoryCursor | None
) -> list[sqlite3.Row]:
    """Caller MUST hold ``_lock``. Newest first, up to ``limit + 1`` full rows.

    The extra row is the probe that answers "is there another page": it is read
    and dropped rather than returned. The row-value predicate ``(ts, id) < (?, ?)``
    is a seek into ``entries_ts_id_idx``; the portable
    ``ts < ? OR (ts = ? AND id < ?)`` spelling plans as a scan from the top of the
    index on every page, which is the property this paging exists to buy. It needs
    SQLite >= ``ROW_VALUE_MIN_SQLITE_VERSION``, which lives beside this predicate
    because the predicate is the only reason the floor exists; it is asserted
    against the frozen sidecar by ``vector_store.selftest``.

    ``before.ts`` is bound to ``CURSOR_TS_MIN``..``CURSOR_TS_MAX`` by
    ``HistoryCursor`` itself: SQLite stores an INTEGER as a signed 64-bit value,
    and binding anything wider raises ``OverflowError`` out of the driver rather
    than answering.
    """
    where = _CURSOR_PAGE_WHERE if before is not None else ""
    params: dict[str, object] = {"limit_plus_one": limit + 1}
    if before is not None:
        params["before_ts"] = before.ts
        params["before_id"] = before.id
    return conn.execute(
        f"SELECT {columns_sql(ENTRY_READ_COLUMNS)} FROM entries "
        f"{where}{_CURSOR_PAGE_ORDER}",
        params,
    ).fetchall()


def cursor_seek_plan_failure() -> str | None:
    """Whether the SQLite actually loaded plans the cursor read as a seek.

    ``None`` when it does; otherwise a message naming the plan it produced
    instead. ``ROW_VALUE_MIN_SQLITE_VERSION`` is the release that *introduced*
    row values, not one at which the planner is known to answer
    ``(ts, id) < (?, ?)`` by seeking ``entries_ts_id_idx``, and a library that
    parses the predicate but walks the index from the top gives back the whole
    property this paging exists to buy, silently. Only a plan taken from the
    loaded library answers that, so this is run against the frozen sidecar by
    ``vector_store.selftest``.

    The probe builds its own miniature ``entries`` -- a key pair and one payload
    column, so the plan is not a covering-index special case -- and reuses the
    same index DDL, predicate and ordering the shipped read uses, rather than a
    second spelling of them.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE entries "
            "(id TEXT PRIMARY KEY, ts INTEGER NOT NULL, raw_text TEXT NOT NULL)"
        )
        conn.executescript(_REPLACE_TS_INDEX_WITH_TS_ID_INDEX)
        plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT id, raw_text FROM entries "
                f"{_CURSOR_PAGE_WHERE}{_CURSOR_PAGE_ORDER}",
                {"before_ts": 0, "before_id": "", "limit_plus_one": 1},
            ).fetchall()
        )
    finally:
        conn.close()
    if "SEARCH" in plan and "entries_ts_id_idx" in plan:
        return None
    return (
        f"SQLite {sqlite3.sqlite_version} does not seek entries_ts_id_idx for a "
        f"cursored history page: {plan}"
    )


def _count_locked(conn: sqlite3.Connection) -> int:
    """Caller MUST hold ``_lock``."""
    return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


def get_page(limit: int = 50, before: HistoryCursor | None = None) -> HistoryPage:
    """The page starting strictly after ``before``, the total, and the next position.

    One acquisition of ``_lock`` covers the page read, the "is there more" probe and
    the total on purpose. Taken separately, a write landing between them returns a
    total that counts the new row and a page that does not.
    """
    clamped_limit = _clamp_limit(limit)
    with _lock:
        conn = _ensure_conn_locked()
        rows = _entries_locked(conn, clamped_limit, before)
        total = _count_locked(conn)
    has_more = len(rows) > clamped_limit
    page = rows[:clamped_limit]
    next_cursor = HistoryCursor(ts=page[-1]["ts"], id=page[-1]["id"]) if has_more else None
    return HistoryPage(
        entries=[_row_to_entry(r) for r in page], total=total, next_cursor=next_cursor
    )


def delete_entry(entry_id: str) -> bool:
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
            invalidate_derived_caches_locked()

    return deleted


def clear_all() -> int:
    with _lock:
        conn = _ensure_conn_locked()
        count = _count_locked(conn)
        conn.execute("BEGIN")
        try:
            conn.execute("DELETE FROM entries")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        invalidate_derived_caches_locked()

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



def invalidate_derived_caches_locked() -> None:
    """Drop every cache derived from ``entries``. Caller MUST hold ``_lock``.

    Two of them exist: this module's own 5 s stats cache, and the word-frequency
    cache in ``app.transcripts.words``, which reads ``derived_generation_locked``
    rather than a timestamp because a whole-table tokenising scan must not be
    repeated while nothing has changed.
    """
    global _stats_cache, _derived_generation
    _stats_cache = None
    _derived_generation += 1


def derived_generation_locked() -> int:
    """The counter ``invalidate_derived_caches_locked`` bumps. Caller MUST hold ``_lock``."""
    return _derived_generation


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
    (covers the case where the output directory was set but ``bootstrap``
    was not called)."""
    global _conn
    if _conn is None:
        _conn = _connect(_resolve_output_dir() / HISTORY_FILENAME)
        _init_schema(_conn)
    return _conn


def _reopen_conn_locked(directory: Path) -> None:
    """Reopen the connection against the history file in ``directory``.

    Caller MUST hold ``_lock``. Closes the current connection (a no-op when
    there is none), connects, applies the schema and invalidates the derived
    caches. ``_output_dir`` is deliberately left alone: on ``relocate``'s
    rollback the store is going back to a directory it may never have had
    recorded, and writing it there would cache a lazily-resolved fallback for
    the life of the process, against ``_resolve_output_dir``'s contract
    (ADR 014). A caller that is moving the store sets ``_output_dir`` itself.
    """
    global _conn
    _close_conn_locked()
    _conn = _connect(directory / HISTORY_FILENAME)
    _init_schema(_conn)
    invalidate_derived_caches_locked()


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

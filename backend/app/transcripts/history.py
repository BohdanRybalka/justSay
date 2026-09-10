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
SCHEMA_VERSION = 4
STATS_TTL_SECONDS = 5.0
HISTORY_LIMIT_MAX = 200
UNKNOWN_TS = 0
STORED_TS_MAX = 253402300799000

CURSOR_TS_MIN = -(2**63)
CURSOR_TS_MAX = 2**63 - 1
ROW_VALUE_MIN_SQLITE_VERSION = (3, 15)

_lock = threading.Lock()
_output_dir: Path | None = None
_conn: sqlite3.Connection | None = None
_stats_cache: tuple[float, HistoryStats] | None = None
_page_total_cache: tuple[float, int, sqlite3.Connection, int] | None = None
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
    """One stored transcript as the app reads it.

    ``timestamp`` is ``None`` for a row whose recording time could not be
    recovered -- see ``UNKNOWN_TS``. Everything else about such a row is
    intact, so it renders in full with a dash where the date goes.
    """

    id: str
    timestamp: str | None
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

    ``ts`` is bounded on the model rather than only in the router's signature,
    for the reason ``_clamp_limit``'s docstring gives about ``limit``: a bound
    that lives only in a FastAPI signature is not a bound on the function. A
    ``ts`` outside the signed 64-bit range SQLite stores an INTEGER in raises
    ``OverflowError`` out of the driver, so a caller that builds a cursor by hand
    gets a ``ValidationError`` here instead.

    ``id`` carries no length bound, because a stored ``id`` is whatever the
    merged or adopted file holds: ``save_entry`` mints twelve hex characters, but
    ``consolidate_into`` copies ids verbatim and ``relocate`` adopts a foreign
    ``history.db`` whole. A length bound therefore rejected the app's own cursor
    rather than a bad one. ADR 053 carries that reasoning; JS-137 is what it cost.
    """

    ts: int = Field(ge=CURSOR_TS_MIN, le=CURSOR_TS_MAX)
    id: str


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

_DDL_V4_ENTRIES = """
CREATE TABLE entries_v4 (
  id TEXT PRIMARY KEY NOT NULL CHECK (typeof(id) = 'text'),
  ts INTEGER NOT NULL CHECK (typeof(ts) = 'integer' AND ts BETWEEN 0 AND @MAXTS@),
  language TEXT NOT NULL CHECK (typeof(language) = 'text'),
  style TEXT NOT NULL CHECK (style IN ('normal', 'ai_prompt')),
  raw_text TEXT NOT NULL CHECK (typeof(raw_text) = 'text'),
  cleaned_text TEXT NOT NULL CHECK (typeof(cleaned_text) = 'text'),
  duration_ms INTEGER NOT NULL CHECK (typeof(duration_ms) = 'integer' AND duration_ms >= 0),
  audio_duration_seconds REAL CHECK (
    audio_duration_seconds IS NULL OR typeof(audio_duration_seconds) IN ('integer', 'real')
  ),
  word_count INTEGER CHECK (word_count IS NULL OR typeof(word_count) = 'integer'),
  model_name TEXT CHECK (model_name IS NULL OR typeof(model_name) = 'text'),
  tokens_used INTEGER CHECK (tokens_used IS NULL OR typeof(tokens_used) = 'integer')
)
""".replace("@MAXTS@", str(STORED_TS_MAX))

_REPAIRED_TS_SQL = (
    "CASE WHEN typeof(ts) IN ('integer', 'real') AND ts BETWEEN 0 AND @MAXTS@ "
    "THEN CAST(ts AS INTEGER) ELSE @UNKNOWN@ END"
).replace("@MAXTS@", str(STORED_TS_MAX)).replace("@UNKNOWN@", str(UNKNOWN_TS))

_REPAIRED_ID_SQL = "CASE WHEN typeof(id) = 'text' THEN id ELSE 'recovered-' || rowid END"


def _repaired_text_sql(column: str) -> str:
    return f"CASE WHEN typeof({column}) = 'text' THEN {column} ELSE '' END"


def _repaired_nullable_sql(column: str, allowed: str) -> str:
    return (
        f"CASE WHEN {column} IS NULL OR typeof({column}) IN ({allowed}) "
        f"THEN {column} ELSE NULL END"
    )


def _repaired_numeric_sql(column: str) -> str:
    return (
        f"CASE WHEN typeof({column}) = 'integer' THEN {column} "
        f"WHEN typeof({column}) = 'real' THEN CAST({column} AS INTEGER) ELSE NULL END"
    )


_MISSING_COLUMN_SQL = {
    "id": "'recovered-' || rowid",
    "ts": str(UNKNOWN_TS),
    "language": "''",
    "style": "'normal'",
    "raw_text": "''",
    "cleaned_text": "''",
    "duration_ms": "0",
    "audio_duration_seconds": "NULL",
    "word_count": "NULL",
    "model_name": "NULL",
    "tokens_used": "NULL",
}


_REPAIRED_COLUMN_SQL = {
    "id": _REPAIRED_ID_SQL,
    "ts": _REPAIRED_TS_SQL,
    "language": _repaired_text_sql("language"),
    "style": "CASE WHEN style IN ('normal', 'ai_prompt') THEN style ELSE 'normal' END",
    "raw_text": _repaired_text_sql("raw_text"),
    "cleaned_text": _repaired_text_sql("cleaned_text"),
    "duration_ms": (
        "CASE WHEN typeof(duration_ms) IN ('integer', 'real') AND duration_ms >= 0 "
        "THEN CAST(duration_ms AS INTEGER) ELSE 0 END"
    ),
    "audio_duration_seconds": _repaired_nullable_sql(
        "audio_duration_seconds", "'integer', 'real'"
    ),
    "word_count": _repaired_numeric_sql("word_count"),
    "model_name": _repaired_nullable_sql("model_name", "'text'"),
    "tokens_used": _repaired_numeric_sql("tokens_used"),
}

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
    unconditionally and idempotently, so the ``entries_ts_idx`` ->
    ``entries_ts_id_idx`` swap reaches an existing database without a
    ``SCHEMA_VERSION`` bump. ``_DDL_V1`` still declares the pre-v4 shape and is
    deliberately left alone: it is what an existing file already holds, and
    ``_migrate_to_v4_locked`` is what replaces it. Why it is not bumped, and what that leaves
    unrecorded, is ADR 053, "What it leaves unrecorded" -- stated there once,
    because the version of it that lived in both places had to be corrected in
    both places.

    Branches:
      - fresh v0 / upgrade from v1, v2 or v3 → ``_migrate_to_v4_locked``,
        which repairs every unreadable row, rebuilds ``entries`` behind
        constraints that refuse the shape, re-runs v2 DDL, rebuilds FTS and
        resets the vector index; then run v3 DDL (embeddings_meta +
        entry_embeddings — both start empty, no rows to replay), and write
        user_version=4 LAST so a crash before the PRAGMA leaves a retry-able
        prior-version state. **A fresh database takes this path too**, on
        nought rows, so a new install and a migrated one end up with the same
        ``entries`` declaration rather than two that have to be kept in step by
        hand. **If the rebuild does not land** -- an ``entries`` this build
        cannot read at all -- nothing else is touched and the version is left
        alone, so the app starts on the store it has and tries again next time.
        The index swap sits inside the two branches for the same reason:
        ``CREATE INDEX ... ON entries(ts DESC, id DESC)`` raises ``no such
        column: id`` on such a table, and running it first meant the migration
        never got to decide anything.
      - already at v4 → re-run v2 DDL (IF NOT EXISTS makes this idempotent)
        and probe FTS integrity; rebuild on OperationalError so a partial
        migration that left user_version=4 but no FTS table self-heals.
        Also re-run v3 DDL (IF NOT EXISTS) so a partial migration that left
        user_version=4 but the embeddings tables missing self-heals too.
    """
    from app.transcripts import vector_store

    conn.executescript(_DDL_V1)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < SCHEMA_VERSION:
        if not _migrate_to_v4_locked(conn):
            return
        conn.executescript(vector_store._DDL_V3)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    else:
        conn.executescript(_REPLACE_TS_INDEX_WITH_TS_ID_INDEX)
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


def _migrate_to_v4_locked(conn: sqlite3.Connection) -> bool:
    """Rebuild ``entries`` so every stored row is one the app can read and order.

    Returns whether the rebuild landed, and **never raises**: ``_init_schema``
    runs inside ``bootstrap``, which the lifespan does not guard, so an
    exception here is not a broken tab but a backend that does not start. A
    store this cannot repair keeps the table it has and the caller leaves
    ``user_version`` alone, so the next open tries again. The index and FTS work
    after the commit is inside a handler for the same reason.

    Runs when ``user_version`` is below 4. ``_DDL_V1`` is
    ``CREATE TABLE IF NOT EXISTS``, so a constraint written there never reaches
    a file that already exists, and SQLite cannot add one to a table in place --
    hence the copy-drop-rename, which is SQLite's own documented procedure.

    **The copy always writes all eleven columns.** An adopted ``entries`` was
    not necessarily created by this app: it can hold a ``style`` outside the
    two, a negative ``duration_ms``, a BLOB ``id``, or simply not have a column
    at all. A present column is read through ``_REPAIRED_COLUMN_SQL``, a missing
    one through ``_MISSING_COLUMN_SQL``. Selecting only the columns the source
    happens to have looks safer and is not: a source with no ``duration_ms``
    leaves that ``NOT NULL`` column empty, ``INSERT OR IGNORE`` then refuses
    every row, and the drop that follows destroys the user's entire history
    with nothing to restore it from.

    **``rowid`` is carried across.** ``entry_fts`` is external-content keyed on
    ``entries.rowid`` and ``vec_entries`` addresses rows by it, and ``vec0`` has
    no rebuild command -- so letting SQLite renumber would mean discarding every
    embedding and recomputing it, which in Cloud mode is the user's whole
    transcript history re-sent to a provider and paid for, over a migration that
    changed no text. Only the embeddings of rows whose *id* was repaired are
    dropped, because those no longer name a row -- and only once
    ``entry_embeddings`` exists, which on a v1 or v2 store it does not until
    the caller runs the v3 DDL after this.

    **The copy is ``INSERT OR IGNORE`` and the count is compared.** Two source
    rows can share an id when the stored table has no primary key, and a
    duplicate must not abort a migration; a shortfall is logged with both
    counts. Carrying *nothing* is refused outright rather than logged: at that
    point the drop would be a deletion, not a migration.

    **Four triggers are dropped first.** The three FTS ones are on ``entries``,
    so ``entries_ai`` fires for every copied row -- work the rebuild discards,
    and a hard failure on a store whose ``entry_fts`` is missing.
    ``entries_ad_vec`` goes with the table either way and is put back by
    ``vector_store.recreate_delete_trigger_locked``.

    **The two ``executescript`` calls are after the COMMIT and have to stay
    there.** Python's ``sqlite3`` issues an implicit COMMIT before running a
    script, so one inside the transaction would end it before the copy.

    The caller writes ``user_version`` last, so a crash between the commit and
    the pragma re-runs this step, which is idempotent: after it every row
    already satisfies the constraints.
    """
    from app.transcripts import vector_store

    try:
        stored = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
    except sqlite3.Error:
        log.exception("Could not read the entries table; leaving the store at its version")
        return False

    if not {"raw_text", "cleaned_text"} & stored:
        log.warning("entries holds no transcript column, so there is nothing here to repair")
        return False

    column_list = columns_sql(ENTRY_COLUMNS)
    select_list = ", ".join(
        _REPAIRED_COLUMN_SQL[name] if name in stored else _MISSING_COLUMN_SQL[name]
        for name in ENTRY_COLUMNS
    )
    previous_foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        stored_rows = conn.execute("SELECT count(*) FROM entries").fetchone()[0]
        for trigger in ("entries_ai", "entries_ad", "entries_au", "entries_ad_vec"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute("DROP TABLE IF EXISTS entries_v4")
        conn.execute(_DDL_V4_ENTRIES)
        copied = conn.execute(
            f"INSERT OR IGNORE INTO entries_v4 (rowid, {column_list}) "
            f"SELECT rowid, {select_list} FROM entries"
        ).rowcount
        if copied == 0 < stored_rows:
            raise sqlite3.IntegrityError(
                f"the rebuilt table would hold none of the {stored_rows} stored rows"
            )
        conn.execute("DROP TABLE entries")
        conn.execute("ALTER TABLE entries_v4 RENAME TO entries")
        conn.execute("COMMIT")
    except sqlite3.Error:
        log.exception("Rebuilding the history table failed; leaving the store at its version")
        return False
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        conn.execute(
            f"PRAGMA foreign_keys = {'ON' if previous_foreign_keys else 'OFF'}"
        )

    if copied != stored_rows:
        log.warning(
            "History rebuild carried %d of the %d stored rows; %d could not be written "
            "into a table that requires a distinct id",
            copied,
            stored_rows,
            stored_rows - copied,
        )

    try:
        conn.executescript(_REPLACE_TS_INDEX_WITH_TS_ID_INDEX)
        conn.executescript(_DDL_V2)
        conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        embeddings_exist = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entry_embeddings'"
        ).fetchone()
        if embeddings_exist:
            conn.execute(
                "DELETE FROM entry_embeddings WHERE entry_id NOT IN (SELECT id FROM entries)"
            )
        vector_store.recreate_delete_trigger_locked(conn)
    except sqlite3.Error:
        log.exception(
            "The history table was rebuilt but its indexes were not; the next open repeats this"
        )
        return False
    return True


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


def _epoch_ms_to_iso(ms: int) -> str | None:
    """The stored ``ts`` as ISO 8601, or ``None`` when there is no date to show.

    ``UNKNOWN_TS`` and anything below it is not a recording time -- it is what
    the v4 migration writes for a row whose stored value could not be recovered
    (see ``_REPAIRED_COLUMN_SQL``). The tabs render a dash for it. A value that
    is not a number at all reaches here only from a store the migration could
    not repair, and answers ``None`` too: a guard that raises is not a guard. Decided by the
    user on 2026-09-10: a record whose text survived keeps its place in the
    list, and its date shows as nothing rather than as a wrong date.
    """
    if not isinstance(ms, int) or not UNKNOWN_TS < ms <= STORED_TS_MAX:
        return None
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
    older source schema degrades to NULLs instead of raising.

    Every column is repaired on the way in rather than copied verbatim,
    through the same ``_REPAIRED_COLUMN_SQL`` the v4 migration applies. Since v4
    the target's ``entries`` refuses a value it cannot read back, and
    ``INSERT OR IGNORE`` answers a refused row by **skipping it** -- measured,
    not assumed -- so copying verbatim would drop the user's transcripts out of
    a merged file with nothing said.

    The repaired id is derived from the source row rather than minted at random,
    which is what keeps the re-run above a no-op: a random one would give the
    same source row a different id on every pass, and ``INSERT OR IGNORE`` would
    have nothing to match it against. ``entry_fts``
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
        if not {"raw_text", "cleaned_text"} & source_columns:
            return ConsolidateOutcome.FAILED, "Source database holds no transcripts"

        column_list = columns_sql(ENTRY_COLUMNS)
        select_list = ", ".join(
            _REPAIRED_COLUMN_SQL[name] if name in source_columns else _MISSING_COLUMN_SQL[name]
            for name in ENTRY_COLUMNS
        )
        available = conn.execute("SELECT count(*) FROM source.entries").fetchone()[0]
        already = conn.execute(
            "SELECT count(*) FROM source.entries s "
            "WHERE EXISTS (SELECT 1 FROM entries t WHERE t.id = s.id)"
        ).fetchone()[0]
        conn.execute("BEGIN")
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO entries ({column_list}) "
            f"SELECT {select_list} FROM source.entries"
        )
        copied = cursor.rowcount
        if copied < available - already:
            log.warning(
                "Merge carried %d of the %d rows the source holds that this history did "
                "not already have",
                copied,
                available - already,
            )
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
_CURSOR_PAGE_ORDER = "ORDER BY ts DESC, id DESC LIMIT :row_limit"


def _entries_locked(
    conn: sqlite3.Connection, limit: int, before: HistoryCursor | None
) -> list[sqlite3.Row]:
    """Caller MUST hold ``_lock``. Newest first, at most ``limit`` rows returned.

    "Is there another page" is a separate key-only question, asked by
    ``_has_more_locked`` and only when this page comes back full, so no
    transcript body is ever read and dropped (ADR 055). The row-value predicate
    ``(ts, id) < (?, ?)`` is a seek into ``entries_ts_id_idx``; the portable
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
    params: dict[str, object] = {"row_limit": limit}
    if before is not None:
        params["before_ts"] = before.ts
        params["before_id"] = before.id
    return conn.execute(
        f"SELECT {columns_sql(ENTRY_READ_COLUMNS)} FROM entries "
        f"{where}{_CURSOR_PAGE_ORDER}",
        params,
    ).fetchall()


def _has_more_locked(conn: sqlite3.Connection, after_ts: int, after_id: str) -> bool:
    """Caller MUST hold ``_lock``. Whether a row exists strictly after that position.

    Takes the position as primitives rather than as a ``HistoryCursor`` so that
    the model is built only once the answer is yes. While ``id`` was length-bounded
    that avoided a real 500 on a full last page; with the bound gone it avoids one
    model construction per last page and nothing more, and no test detects a revert
    to the model form.

    Selects no transcript column, so on ``entries_ts_id_idx`` the whole question
    is answered inside the index without reaching a table row. Built from the
    same ``_CURSOR_PAGE_WHERE`` and ``_CURSOR_PAGE_ORDER`` as the page read, so
    the predicate the seek depends on keeps one spelling.
    """
    return (
        conn.execute(
            f"SELECT 1 FROM entries {_CURSOR_PAGE_WHERE}{_CURSOR_PAGE_ORDER}",
            {"before_ts": after_ts, "before_id": after_id, "row_limit": 1},
        ).fetchone()
        is not None
    )


_CURSOR_SEEK_PLAN_SHAPES = (
    ("page read", "id, raw_text", False),
    ("has-more probe", "1", True),
)


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

    Two statement *shapes* ship on the cursored read path since ADR 055 -- a
    payload projection and ``_has_more_locked``'s key-only ``SELECT 1`` -- so
    both are planned here and the message names which of the two failed. The
    projections are stand-ins for the shipped column lists rather than copies of
    them: the miniature table has one payload column where the page read selects
    all of ``ENTRY_READ_COLUMNS``.

    A seek alone is not the property: a library that seeks and then sorts the
    result, or that reaches the rows through a table walk, has given the flat
    cost back. All three are asserted of each shape, matching the local pin.

    Every shape is planned and every condition it failed is reported, joined
    into one message, rather than stopping at the first shape or at the first
    condition within a shape. ``selftest`` promises exactly that of every check
    it runs, and a plan that both misses the index and reaches the table is the
    packaged build where the second half is worth knowing about.

    The has-more probe carries a fourth assertion the page read cannot: its plan
    must say ``COVERING INDEX``. Reading no transcript column is the entire point
    of that statement, and a spelling that reaches the table -- ``SELECT *``, say
    -- still seeks the index and so passes every other check while pulling a
    whole transcript body per page. The page read legitimately reaches the table
    and is held to the first three only.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE entries "
            "(id TEXT PRIMARY KEY, ts INTEGER NOT NULL, raw_text TEXT NOT NULL)"
        )
        conn.executescript(_REPLACE_TS_INDEX_WITH_TS_ID_INDEX)
        plans = [
            (
                shape,
                requires_covering,
                " ".join(
                    str(row[3])
                    for row in conn.execute(
                        f"EXPLAIN QUERY PLAN SELECT {projection} FROM entries "
                        f"{_CURSOR_PAGE_WHERE}{_CURSOR_PAGE_ORDER}",
                        {"before_ts": 0, "before_id": "", "row_limit": 1},
                    ).fetchall()
                ),
            )
            for shape, projection, requires_covering in _CURSOR_SEEK_PLAN_SHAPES
        ]
    finally:
        conn.close()
    failures: list[str] = []
    for shape, requires_covering, plan in plans:
        reasons: list[str] = []
        if "SEARCH" not in plan or "entries_ts_id_idx" not in plan:
            reasons.append("does not seek entries_ts_id_idx")
        if "SCAN entries" in plan:
            reasons.append("walks the entries table")
        if "TEMP B-TREE" in plan.upper():
            reasons.append("sorts the result afterwards")
        if requires_covering and "COVERING INDEX" not in plan.upper():
            reasons.append("reaches the entries table instead of answering from the index")
        if reasons:
            failures.append(
                f"SQLite {sqlite3.sqlite_version} {', and '.join(reasons)} for the "
                f"cursored history {shape}: {plan}"
            )
    return "; ".join(failures) if failures else None


def _count_locked(conn: sqlite3.Connection) -> int:
    """Caller MUST hold ``_lock``. The row count, memoised until it can have changed.

    Reads and *rewrites* ``_page_total_cache``: every call either serves the
    memoised total or replaces the entry with a freshly counted one, so this
    reads as a query and is not one.

    Keyed on ``(age, derived generation, connection object)`` (ADR 055), and all
    three have to hold for a hit. Every statement in the backend that changes the
    number of rows in ``entries`` lives in this module and bumps the generation,
    and every path that swaps the connection replaces the object as well, so both
    of those halves are already maintained by the code for its own reasons. The
    connection half is what keeps this out of the fixtures: a suite that closes
    the connection without touching the generation gets a different instance on
    the next open, and so misses without being told to.

    ``STATS_TTL_SECONDS`` is the third, and it is the bound ``_stats_cache``
    already keeps for the same reason: a second process writing this
    ``history.db`` changes the table without going through any of this module's
    generation bumps, which is the sync-folder case ``journal_mode=DELETE``
    exists for. With no age, that writer would never be picked up and the two
    derived totals this module serves would disagree until a restart. A scroll
    still pays one count every five seconds rather than one per page, which is
    the property ADR 055 was bought for.

    The identity test is only sound because the cache tuple itself holds a strong
    reference to the connection it was keyed on. Were the reference weak, a freed
    ``Connection`` could be replaced at the same address and ``is`` would match a
    different connection over a different file. ``_close_conn_locked`` clears the
    cache so that reference is never a retention.
    """
    global _page_total_cache
    cached = _page_total_cache
    if (
        cached is not None
        and (time.monotonic() - cached[0]) < STATS_TTL_SECONDS
        and cached[1] == _derived_generation
        and cached[2] is conn
    ):
        return cached[3]
    total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    _page_total_cache = (time.monotonic(), _derived_generation, conn, total)
    return total


def get_page(limit: int = 50, before: HistoryCursor | None = None) -> HistoryPage:
    """The page starting strictly after ``before``, the total, and the next position.

    One acquisition of ``_lock`` covers the page read, the "is there more" probe and
    the total on purpose, and what it still guarantees is that no *in-process* write
    lands between the three reads: taken separately, a ``save_entry`` between them
    would return a total that counts the new row and a page that does not.

    It no longer makes the three reads describe one state of the file. Since the
    total became memoised with a TTL (ADR 055), a second process writing the same
    ``history.db`` bumps no generation and swaps no connection, so the rows are read
    fresh while the total can be up to ``STATS_TTL_SECONDS`` old --
    ``test_the_memoised_total_expires_so_a_second_writer_is_picked_up`` pins that
    bound.

    A page that comes back short is the last one, so the probe is skipped entirely
    rather than asked a question its own length already answered.

    The cursor is built only once the probe has answered yes, so a full page with
    no page after it validates nothing. Where a cursor *is* returned it is minted
    from the row's own key, and no *length* of id can make that fail any more.

    A key of the wrong *type* still can, and that is JS-146 rather than this
    function's business to paper over: SQLite permits NULL in ``id TEXT PRIMARY
    KEY`` on a rowid table, and affinity leaves a REAL or TEXT ``ts`` in an
    ``INTEGER NOT NULL`` column, so a hand-written or foreign ``history.db`` can
    hold a row that is not a position in ``(ts DESC, id DESC)`` at all. Such a row
    cannot be paged past whatever this mints, so refusing it belongs where rows
    enter the store, not here.
    """
    clamped_limit = _clamp_limit(limit)
    with _lock:
        conn = _ensure_conn_locked()
        rows = _entries_locked(conn, clamped_limit, before)
        next_cursor: HistoryCursor | None = None
        if len(rows) == clamped_limit and _has_more_locked(
            conn, rows[-1]["ts"], rows[-1]["id"]
        ):
            next_cursor = HistoryCursor(ts=rows[-1]["ts"], id=rows[-1]["id"])
        total = _count_locked(conn)
    return HistoryPage(
        entries=[_row_to_entry(r) for r in rows], total=total, next_cursor=next_cursor
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
    """Deletes every entry and returns how many rows the DELETE actually removed.

    The number comes from the statement itself rather than from a count taken
    beforehand: ``_count_locked`` may answer from the memoised total (ADR 055),
    and the API reports this as ``ClearResult.deleted``.
    """
    with _lock:
        conn = _ensure_conn_locked()
        conn.execute("BEGIN")
        try:
            count = conn.execute("DELETE FROM entries").rowcount
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

    Three of them exist: this module's own 5 s stats cache, its page-total cache,
    and the word-frequency cache in ``app.transcripts.words``. The last two read
    ``derived_generation_locked`` rather than a timestamp, because a whole-table
    tokenising scan and a whole-table count must not be repeated while nothing has
    changed, and must never survive a change. Bumping the counter is the whole
    invalidation: neither of them is cleared by name here.
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
    """Caller MUST hold ``_lock``. Closes the connection and drops the page total.

    Dropping ``_page_total_cache`` here is what stops a closed connection being
    retained for the life of the process, and with it the only object whose
    identity ``_count_locked`` compares against.
    """
    global _conn, _page_total_cache
    _page_total_cache = None
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

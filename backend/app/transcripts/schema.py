"""The ``entries`` schema — DDL, column lists and the version-aware migrator.

Split out of ``history.py`` by spec 164; the SQL is unchanged. ``history``
imports this module and this module names nothing in ``history``. The two
``from app.transcripts import vector_store`` imports below stay inside their
function bodies: ``vector_store`` imports ``history`` at module top, so
hoisting either makes ``schema -> vector_store -> history -> schema`` an
ImportError at start-up.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence

log = logging.getLogger(__name__)

SCHEMA_VERSION = 5
UNKNOWN_TS = 0
STORED_TS_MAX = 253402300799000

ENTRY_COLUMNS = (
    "id",
    "ts",
    "language",
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


_DDL_V1 = """
CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY,
  ts INTEGER NOT NULL,
  language TEXT NOT NULL,
  raw_text TEXT NOT NULL,
  cleaned_text TEXT NOT NULL,
  duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
  audio_duration_seconds REAL,
  word_count INTEGER,
  model_name TEXT,
  tokens_used INTEGER
);
"""

_DDL_V5_ENTRIES = """
CREATE TABLE entries_v5 (
  id TEXT PRIMARY KEY NOT NULL CHECK (typeof(id) = 'text'),
  ts INTEGER NOT NULL CHECK (typeof(ts) = 'integer' AND ts BETWEEN 0 AND @MAXTS@),
  language TEXT NOT NULL CHECK (typeof(language) = 'text'),
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
    ``SCHEMA_VERSION`` bump. ``_DDL_V1`` still declares the pre-v5 shape and is
    deliberately left alone: it is what an existing file already holds, and
    ``_migrate_to_v5_locked`` is what replaces it. Why it is not bumped, and what that leaves
    unrecorded, is ADR 053, "What it leaves unrecorded" -- stated there once,
    because the version of it that lived in both places had to be corrected in
    both places.

    Branches:
      - fresh v0 / upgrade from v1, v2, v3 or v4 → ``_migrate_to_v5_locked``,
        which repairs every unreadable row, rebuilds ``entries`` behind
        constraints that refuse the shape, re-runs v2 DDL, rebuilds FTS and
        resets the vector index; then run v3 DDL (embeddings_meta +
        entry_embeddings — both start empty, no rows to replay), and write
        user_version=5 LAST so a crash before the PRAGMA leaves a retry-able
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
      - already at v5 → re-run v2 DDL (IF NOT EXISTS makes this idempotent)
        and probe FTS integrity; rebuild on OperationalError so a partial
        migration that left user_version=5 but no FTS table self-heals.
        Also re-run v3 DDL (IF NOT EXISTS) so a partial migration that left
        user_version=5 but the embeddings tables missing self-heals too.
    """
    from app.transcripts import vector_store

    conn.executescript(_DDL_V1)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < SCHEMA_VERSION:
        if not _migrate_to_v5_locked(conn):
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


def _migrate_to_v5_locked(conn: sqlite3.Connection) -> bool:
    """Rebuild ``entries`` so every stored row is one the app can read and order.

    Returns whether the rebuild landed, and **never raises**: ``_init_schema``
    runs inside ``bootstrap``, which the lifespan does not guard, so an
    exception here is not a broken tab but a backend that does not start. A
    store this cannot repair keeps the table it has and the caller leaves
    ``user_version`` alone, so the next open tries again. The index and FTS work
    after the commit is inside a handler for the same reason.

    Runs when ``user_version`` is below 5. ``_DDL_V1`` is
    ``CREATE TABLE IF NOT EXISTS``, so a constraint written there never reaches
    a file that already exists, and SQLite cannot add one to a table in place --
    hence the copy-drop-rename, which is SQLite's own documented procedure.

    **The copy always writes all ten columns.** An adopted ``entries`` was
    not necessarily created by this app: it can hold a negative ``duration_ms``,
    a BLOB ``id``, or simply not have a column at all. A column this build no
    longer knows -- a v4 store's ``style`` -- is not read: the select list is
    built from ``ENTRY_COLUMNS``, not from the source. A present column is read
    through ``_REPAIRED_COLUMN_SQL``, a missing one through
    ``_MISSING_COLUMN_SQL``. Selecting only the columns the source
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
        conn.execute("DROP TABLE IF EXISTS entries_v5")
        conn.execute(_DDL_V5_ENTRIES)
        copied = conn.execute(
            f"INSERT OR IGNORE INTO entries_v5 (rowid, {column_list}) "
            f"SELECT rowid, {select_list} FROM entries"
        ).rowcount
        if copied == 0 < stored_rows:
            raise sqlite3.IntegrityError(
                f"the rebuilt table would hold none of the {stored_rows} stored rows"
            )
        conn.execute("DROP TABLE entries")
        conn.execute("ALTER TABLE entries_v5 RENAME TO entries")
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

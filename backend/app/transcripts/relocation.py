"""Moving the history file — relocation on a settings change, one-off consolidation.

Split out of ``history.py`` by spec 164; the logic is unchanged. Both state
machines borrow ``history``'s lock, connection factory and cache invalidation,
and ``relocate`` assigns ``history._output_dir`` and ``history._conn`` through
the module object. That import form is load-bearing rather than stylistic: a
``from app.transcripts.history import _conn`` would bind a local name here and
leave the live store on the connection it was told to stop using. ``history``
names nothing here, so the edge runs one way and callers repoint rather than
reach a re-export.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from app.transcripts import history, schema

log = logging.getLogger(__name__)


class RelocateOutcome(str, Enum):
    MOVED = "moved"
    NEW_ALREADY_HAS_FILE = "new_already_has_file"
    NO_OLD_FILE = "no_old_file"
    FAILED = "failed"


class ConsolidateOutcome(str, Enum):
    CONSOLIDATED = "consolidated"
    NOT_NEEDED = "not_needed"
    FAILED = "failed"


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
    exactly. The ``schema._init_schema(new_conn)`` call below already re-attaches
    the v3 tables via ``IF NOT EXISTS`` — a no-op on a file that already
    has them.
    """
    with history._lock:
        old_dir = history._resolve_output_dir()
        old_path = old_dir / history.HISTORY_FILENAME

        try:
            same = old_dir.resolve() == new_dir.resolve()
        except OSError:
            same = False

        if same:
            history.invalidate_derived_caches_locked()
            return RelocateOutcome.NO_OLD_FILE, None

        new_path = new_dir / history.HISTORY_FILENAME

        try:
            new_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return RelocateOutcome.FAILED, f"Could not create target directory: {e}"

        if new_path.exists():
            history._output_dir = new_dir
            history._reopen_conn_locked(new_dir)
            return RelocateOutcome.NEW_ALREADY_HAS_FILE, None

        if not old_path.exists():
            history._output_dir = new_dir
            history._reopen_conn_locked(new_dir)
            return RelocateOutcome.NO_OLD_FILE, None

        history._close_conn_locked()
        new_conn: sqlite3.Connection | None = None
        try:
            shutil.copy2(old_path, new_path)
            if not _verify_db_row_count(old_path, new_path):
                new_path.unlink(missing_ok=True)
                history._reopen_conn_locked(old_dir)
                return RelocateOutcome.FAILED, "Verification failed: entry count mismatch"

            new_conn = history._connect(new_path)
            schema._init_schema(new_conn)
            new_conn.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")

            old_path.unlink()
            history._output_dir = new_dir
            history._conn = new_conn
            new_conn = None
            history.invalidate_derived_caches_locked()
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
                history._reopen_conn_locked(old_dir)
            except sqlite3.Error:
                history._conn = None
                history.invalidate_derived_caches_locked()
            log.exception("Relocate failed: %s", e)
            return RelocateOutcome.FAILED, f"Move failed: {e}"


def _premigration_path(target_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    candidate = target_dir / f"{history.HISTORY_FILENAME}.premigration-{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = target_dir / f"{history.HISTORY_FILENAME}.premigration-{stamp}-{suffix}"
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
    through the same ``schema._REPAIRED_COLUMN_SQL`` the rebuild applies. Since v4
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
    source_path = source_dir / history.HISTORY_FILENAME
    target_path = target_dir / history.HISTORY_FILENAME

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
        conn = history._connect(target_path)
        schema._init_schema(conn)
        conn.execute("ATTACH DATABASE ? AS source", (str(source_path),))
        source_columns = {row[1] for row in conn.execute("PRAGMA source.table_info(entries)")}
        if not {"raw_text", "cleaned_text"} & source_columns:
            return ConsolidateOutcome.FAILED, "Source database holds no transcripts"

        column_list = schema.columns_sql(schema.ENTRY_COLUMNS)
        select_list = ", ".join(
            schema._REPAIRED_COLUMN_SQL[name]
            if name in source_columns
            else schema._MISSING_COLUMN_SQL[name]
            for name in schema.ENTRY_COLUMNS
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


def _verify_db_row_count(src_db: Path, dst_db: Path) -> bool:
    """Compare entries row count between two SQLite files."""
    def count(p: Path) -> int:
        c = history._connect(p)
        try:
            return c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        finally:
            c.close()
    try:
        return count(src_db) == count(dst_db)
    except sqlite3.Error:
        return False

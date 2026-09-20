"""Moving the history file — relocation on a settings change, one-off consolidation.

Both state machines borrow ``history``'s lock, connection factory and cache
invalidation, reached through the module object. ``relocate`` hands every store
it has moved to ``history.adopt_store_locked``. ``history`` names nothing here,
so the edge runs one way.
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


_ADOPTION_FAILURES = (OSError, sqlite3.Error, ValueError)


def _adopt_existing_store(
    new_dir: Path, outcome: RelocateOutcome
) -> tuple[RelocateOutcome, str | None]:
    """Caller MUST hold ``history._lock``. Adopt the store already in ``new_dir``.

    Answers ``outcome`` once both halves name ``new_dir``, or ``FAILED`` with a
    reason when opening or migrating that database raises. A failure removes the
    store file the attempt created, which is every ``outcome`` but
    ``NEW_ALREADY_HAS_FILE``, so a retry meets the directory as it found it.
    """
    try:
        history.adopt_store_locked(new_dir)
    except _ADOPTION_FAILURES as e:
        log.exception("Relocate could not adopt %s: %s", new_dir, e)
        if outcome is not RelocateOutcome.NEW_ALREADY_HAS_FILE:
            (new_dir / history.HISTORY_FILENAME).unlink(missing_ok=True)
        return RelocateOutcome.FAILED, f"Move failed: {e}"
    return outcome, None


def relocate(new_dir: Path) -> tuple[RelocateOutcome, str | None]:
    """Move history.db to ``new_dir``, reporting what it did and why.

    Holds ``history._lock`` throughout and moves the store only through
    ``history.adopt_store_locked``. The source is unlinked only after that
    adoption succeeds, so a failure leaves the rows readable at the source
    (ADR 086).
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
            return _adopt_existing_store(new_dir, RelocateOutcome.NEW_ALREADY_HAS_FILE)

        if not old_path.exists():
            return _adopt_existing_store(new_dir, RelocateOutcome.NO_OLD_FILE)

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

            history.adopt_store_locked(new_dir, new_conn)
            new_conn = None
        except _ADOPTION_FAILURES as e:
            if new_conn is not None:
                history._close_quietly(new_conn)
            new_path.unlink(missing_ok=True)
            try:
                history._reopen_conn_locked(old_dir)
            except _ADOPTION_FAILURES:
                history._close_conn_locked()
                history.invalidate_derived_caches_locked()
            log.exception("Relocate failed: %s", e)
            return RelocateOutcome.FAILED, f"Move failed: {e}"

        try:
            old_path.unlink()
        except OSError as e:
            log.warning("History moved but %s could not be removed: %s", old_path, e)
        log.info("Relocated history %s → %s", old_path, new_path)
        return RelocateOutcome.MOVED, None


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

    Rows copy with ``INSERT OR IGNORE`` on ``id``, each column repaired through
    ``schema._REPAIRED_COLUMN_SQL``. Touches no module state; runs pre-``bootstrap``.
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
            history._close_quietly(conn)

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

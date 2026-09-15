"""Transcript history — SQLite store at ``<output_dir>/history.db``.

This module owns the connection, the lock and the reads and writes over the
``entries`` table. The table's DDL, column lists and migrations live in
``schema.py``; moving the file lives in ``relocation.py``. The directory is
user-configurable via ``UserSettings.output_dir``. ``history.py`` deliberately
does not import ``user_settings`` — the path is pushed in via ``bootstrap``
(lifespan) or mutated by ``relocation.relocate`` (settings change). One-way
dependency (user_settings → history).

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
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlite_vec
from pydantic import BaseModel, Field

from app.core.app_paths import resolve_app_data_root
from app.transcripts import schema

log = logging.getLogger(__name__)

HISTORY_FILENAME = "history.db"
STATS_TTL_SECONDS = 5.0
HISTORY_LIMIT_MAX = 200

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


class HistoryEntry(BaseModel):
    """One stored transcript as the app reads it.

    ``timestamp`` is ``None`` for a row whose recording time could not be
    recovered -- see ``schema.UNKNOWN_TS``. Everything else about such a row is
    intact, so it renders in full with a dash where the date goes.
    """

    id: str
    timestamp: str | None
    language: str
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
        schema._init_schema(_conn)


def _iso_to_epoch_ms(ts: str) -> int:
    """Parse ISO 8601 (Python 3.10-safe via Z→+00:00 shim) → unix epoch ms."""
    return int(round(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000))


def _epoch_ms_to_iso(ms: int) -> str | None:
    """The stored ``ts`` as ISO 8601, or ``None`` when there is no date to show.

    ``schema.UNKNOWN_TS`` and anything below it is not a recording time -- it is what
    the migration writes for a row whose stored value could not be recovered
    (see ``schema._REPAIRED_COLUMN_SQL``). The tabs render a dash for it. A value that
    is not a number at all reaches here only from a store the migration could
    not repair, and answers ``None`` too: a guard that raises is not a guard. Decided by the
    user on 2026-09-10: a record whose text survived keeps its place in the
    list, and its date shows as nothing rather than as a wrong date.
    """
    if not isinstance(ms, int) or not schema.UNKNOWN_TS < ms <= schema.STORED_TS_MAX:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()



def save_entry(
    text: str,
    duration_ms: int,
    language: str = "uk",
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
                f"INSERT INTO entries({schema.columns_sql(schema.ENTRY_COLUMNS)}) "
                f"VALUES ({', '.join(':' + name for name in schema.ENTRY_COLUMNS)})",
                {
                    "id": entry.id,
                    "ts": ts_ms,
                    "language": entry.language,
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

    ``words.top_words`` and ``search.search_history`` each clamp their own against
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
        f"SELECT {schema.columns_sql(schema.ENTRY_READ_COLUMNS)} FROM entries "
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
    all of ``schema.ENTRY_READ_COLUMNS``.

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
        conn.executescript(schema._REPLACE_TS_INDEX_WITH_TS_ID_INDEX)
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
        schema._init_schema(_conn)
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
    schema._init_schema(_conn)
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

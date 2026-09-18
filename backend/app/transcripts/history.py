"""Transcript history — SQLite store at ``<output_dir>/history.db``.

One shared sqlite3 connection per database path, every access serialised
through the module-level ``_lock``; ``search``, ``words`` and ``vector_store``
read ``entries`` through ``_ensure_conn_locked`` under that same lock (ADR 072).
DDL, column lists and migrations live in ``schema.py``, moving the file in
``relocation.py``. The output directory is pushed in by ``bootstrap`` or by
``relocation.relocate``: this module never imports ``user_settings``.
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
    recovered (``schema.UNKNOWN_TS``); the rest of such a row is intact.
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

    ``id`` is the tiebreaker that makes the ordering total (ADR 053). ``ts`` is
    bounded on the model, so a cursor built out of range is a ``ValidationError``.
    """

    ts: int = Field(ge=CURSOR_TS_MIN, le=CURSOR_TS_MAX)
    id: str


class HistoryPage(BaseModel):
    """One page, the total it is a page of, and where the next one starts.

    ``next_cursor`` is ``None`` on the last page. ``total`` is what the user
    reads on screen, not how the client decides whether more rows exist.
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

    ``timeout=0.2`` maps to PRAGMA busy_timeout = 200 ms; ``isolation_level=None``
    means every mutating function MUST issue an explicit BEGIN/COMMIT.
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
    """`_output_dir` if `bootstrap()`/`relocation.relocate()` has already set one, else a
    lazy fallback resolved fresh on every call -- never cached at import time
    (ADR 014)."""
    return _output_dir if _output_dir is not None else resolve_app_data_root()


def history_path() -> Path:
    """Lock-free read of the current history.db path."""
    return _resolve_output_dir() / HISTORY_FILENAME


def bootstrap(target: Path) -> None:
    """Lifespan helper: open the SQLite connection at ``<target>/history.db``.

    Leaves no connection behind when it fails, and keeps ``_output_dir``, so
    ``_ensure_conn_locked`` re-opens at ``target`` on the first request for it.
    """
    global _output_dir, _conn
    with _lock:
        _close_conn_locked()
        _output_dir = target
        invalidate_derived_caches_locked()
        target.mkdir(parents=True, exist_ok=True)
        _conn = _connect(target / HISTORY_FILENAME)
        try:
            schema._init_schema(_conn)
        except Exception:
            _close_conn_locked()
            raise


def _iso_to_epoch_ms(ts: str) -> int:
    """Parse ISO 8601 (Python 3.10-safe via Z→+00:00 shim) → unix epoch ms."""
    return int(round(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000))


def _epoch_ms_to_iso(ms: int) -> str | None:
    """The stored ``ts`` as ISO 8601, or ``None`` when there is no date to show.

    Anything outside ``UNKNOWN_TS < ts <= schema.STORED_TS_MAX``, and any value
    that is not an ``int``, answers ``None`` rather than raising: the row keeps its
    place and renders a dash.
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

    ``words.top_words`` and ``search.search_history`` clamp against their own
    maxima: a FastAPI signature is not a bound, and ``LIMIT -1`` reads every row.
    """
    return max(1, min(int(limit), HISTORY_LIMIT_MAX))


_CURSOR_PAGE_WHERE = "WHERE (ts, id) < (:before_ts, :before_id) "
_CURSOR_PAGE_ORDER = "ORDER BY ts DESC, id DESC LIMIT :row_limit"


def _entries_locked(
    conn: sqlite3.Connection, limit: int, before: HistoryCursor | None
) -> list[sqlite3.Row]:
    """Caller MUST hold ``_lock``. Newest first, at most ``limit`` rows returned.

    "Is there another page" is ``_has_more_locked``'s separate key-only question
    (ADR 055). Needs SQLite >= ``ROW_VALUE_MIN_SQLITE_VERSION`` for the row value.
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

    Takes primitives, so no ``HistoryCursor`` is built until the answer is yes, and
    selects no transcript column, so ``entries_ts_id_idx`` answers without a row.
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
    """Whether the SQLite actually loaded plans the cursored read as a seek.

    ``None`` when it does; otherwise a message naming every statement shape that
    failed and every condition it failed. Run by ``vector_store.selftest``.
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

    Reads and *rewrites* ``_page_total_cache``, keyed on age, derived generation
    and connection identity, all three of which must hold for a hit (ADR 055).
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

    One acquisition of ``_lock`` covers all three reads, so no in-process write lands
    between them; the total may be up to ``STATS_TTL_SECONDS`` behind (ADR 055).
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
    beforehand, which may be served from the memoised total (ADR 055).
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

    Clears the stats cache and bumps ``derived_generation_locked``; the page-total
    and word-frequency caches read that counter, so the bump is the invalidation.
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

    Caller MUST hold ``_lock``. Closes, connects, applies the schema, invalidates
    the derived caches. ``_output_dir`` is left alone (ADR 014) for the caller.
    """
    global _conn
    _close_conn_locked()
    _conn = _connect(directory / HISTORY_FILENAME)
    schema._init_schema(_conn)
    invalidate_derived_caches_locked()


def _close_conn_locked() -> None:
    """Caller MUST hold ``_lock``. Closes the connection and drops the page total.

    Dropping ``_page_total_cache`` releases the strong reference that would
    otherwise retain a closed connection for the life of the process.
    """
    global _conn, _page_total_cache
    _page_total_cache = None
    if _conn is not None:
        try:
            _conn.close()
        except sqlite3.Error:
            pass
        _conn = None

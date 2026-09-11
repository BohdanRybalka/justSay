"""Translation of SQLite lock contention into a refusal the caller can read.

Both routers in this package read the same SQLite file through the same
connection and lock, so "database is locked" means one thing wherever it
surfaces: another writer holds it, and the caller should retry. It used to be
written out at six call sites with four different detail strings — "History
store busy", "Stats store busy", "Words store busy" and an unreachable default
— which read as three different stores to a client and to a reader. There is
one store.

The refusal is a ``ResourceUnavailableError``, which already answers 503 and
carries the ``Retry-After`` header to the handler, so this module needs no web
framework of its own.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from app.core.errors import ResourceUnavailableError

_BUSY_MARKER = "locked"
_BUSY_DETAIL = "Transcript store busy"
_RETRY_AFTER_SECONDS = "1"


def is_store_busy(error: sqlite3.OperationalError) -> bool:
    """True when the error is lock contention rather than a real fault."""
    return _BUSY_MARKER in str(error).lower()


@contextmanager
def store_busy_as_503() -> Iterator[None]:
    """Map lock contention to 503 + Retry-After, and let everything else pass.

    Any other ``OperationalError`` propagates unchanged so a caller that needs
    to recognise it — the search endpoint distinguishing an FTS5 syntax error —
    can still do so in an enclosing handler.
    """
    try:
        yield
    except sqlite3.OperationalError as e:
        if is_store_busy(e):
            raise ResourceUnavailableError(
                _BUSY_DETAIL,
                diagnostic=str(e),
                headers={"Retry-After": _RETRY_AFTER_SECONDS},
            ) from e
        raise

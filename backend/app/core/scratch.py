"""Deleting scratch audio files without letting a failed delete replace the real outcome."""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def discard_scratch_file(path: Path) -> None:
    """Delete ``path`` if it exists; an ``OSError`` is logged, never raised.

    Callers sit in ``finally`` blocks, where raising would turn a finished result into a crash.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("Could not remove scratch file %s", path, exc_info=True)

"""Strong-reference registry for fire-and-forget asyncio tasks."""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

log = logging.getLogger(__name__)

# The event loop holds only a WEAK reference to a pending task; without this
# set a still-pending task can be garbage-collected mid-execution, silently
# stopping halfway with nothing logged. Entries are discarded by _on_task_done.
_background_tasks: set[asyncio.Task] = set()


def spawn_background_task(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)  # background-task-ok: this IS the helper
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
    return task


def _on_task_done(task: asyncio.Task) -> None:
    # Discard FIRST: a failure inside the logging below must never leak the ref.
    _background_tasks.discard(task)
    if task.cancelled():
        log.debug("Background task %r was cancelled", task.get_name())
        return
    exc = task.exception()
    if exc is not None:
        log.warning("Background task %r failed", task.get_name(), exc_info=exc)

"""Strong-reference registry for fire-and-forget asyncio tasks."""

import asyncio
import logging
from collections.abc import Coroutine, Iterable
from typing import Any

log = logging.getLogger(__name__)

SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 1.0

_background_tasks: set[asyncio.Task] = set()


def spawn_background_task(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)  # background-task-ok: this IS the helper
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
    return task


def _on_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        log.debug("Background task %r was cancelled", task.get_name())
        return
    exc = task.exception()
    if exc is not None:
        log.warning("Background task %r failed", task.get_name(), exc_info=exc)


async def cancel_all(
    extra: Iterable[asyncio.Task | None] = (),
    *,
    timeout: float = SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
) -> list[str]:
    """Cancel every registered task and any in ``extra``; return those that did not settle in time.

    Only a ``CancelledError`` aimed at the caller's own task escapes: a caller whose next steps must
    run needs ``try``/``finally``. The registry snapshots at entry — never spawn during a drain.
    """
    pending = {t for t in (*_background_tasks, *extra) if t is not None and not t.done()}
    if not pending:
        return []

    for task in pending:
        task.cancel()

    _, still_pending = await asyncio.wait(pending, timeout=timeout)

    stragglers = [t.get_name() for t in still_pending]
    for name in stragglers:
        log.warning(
            "Background task %r did not settle within %.1fs of cancellation "
            "-- abandoning it and continuing shutdown",
            name,
            timeout,
        )
    return stragglers

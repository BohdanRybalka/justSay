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
    """Cancel every registered background task (plus any in ``extra``) and
    wait, bounded by ``timeout``, for the cancellations to settle.

    Returns the names of the tasks that did NOT settle within the budget.

    Nothing a drained task does can propagate out: `asyncio.wait()` reports
    cancellations and failures through its result sets instead of re-raising.
    The one exception that CAN escape is a `CancelledError` delivered to the
    *caller's own* task while it is awaiting here -- `asyncio.wait()` re-raises
    that. Callers whose subsequent steps must run regardless (see
    `app.main.lifespan`) need a `try/finally` around this call.

    BOUNDARY -- the registry is snapshotted at entry, before the first await.
    A task spawned via `spawn_background_task()` AFTER that point is NOT
    cancelled by this call and dies with the event loop, unobserved -- the
    exact defect Spec 036 removes. So: do not call `spawn_background_task()`
    from shutdown code running concurrently with, or after, this drain, and
    that includes anything a cancelled task starts from its own
    `except CancelledError` / `finally` block. No call site does this today;
    if one ever needs to, add a bounded second pass here (a fixed number of
    passes, never a `while` loop -- a task that respawns on cancel would make
    it non-terminating).
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

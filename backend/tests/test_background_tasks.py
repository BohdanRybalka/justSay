"""Tests for `app.core.tasks`: the strong-reference registry for
fire-and-forget asyncio tasks (Spec 032), and the AST-based invariant scan
that keeps every future `backend/app/**/*.py` fire-and-forget call site
routed through it (or explicitly marked as an exception).
"""

import ast
import asyncio
import logging
from pathlib import Path

import pytest

from app.core import tasks

# --- AC 2-5: spawn_background_task / _on_task_done ---------------------------


@pytest.mark.asyncio
async def test_strong_ref_held_while_pending_and_released_on_completion():
    """AC 2: the task is registered in `_background_tasks` immediately after
    `spawn_background_task()` returns, and the set is empty (not merely
    smaller) once the task has finished."""

    async def _noop() -> None:
        return None

    task = tasks.spawn_background_task(_noop(), name="ac2-noop")
    assert task in tasks._background_tasks

    await task

    assert tasks._background_tasks == set()


@pytest.mark.asyncio
async def test_failed_task_logs_warning_with_name_and_traceback(caplog):
    """AC 3: a raising task logs a WARNING via app.core.tasks's own logger,
    with the task's name in the message and the traceback attached."""

    async def _boom() -> None:
        raise RuntimeError("kaboom")

    with caplog.at_level(logging.WARNING, logger="app.core.tasks"):
        task = tasks.spawn_background_task(_boom(), name="ac3-boom")
        with pytest.raises(RuntimeError, match="kaboom"):
            await task

    records = [r for r in caplog.records if r.name == "app.core.tasks"]
    assert len(records) == 1
    record = records[0]
    assert record.levelname == "WARNING"
    assert "ac3-boom" in record.getMessage()
    # `record.exc_info is not None` alone is too weak: a future
    # `exc_info=exc` -> `exc_info=True` edit, called from a done-callback
    # (never inside an active `except` block), would resolve via
    # `sys.exc_info()` to `(None, None, None)` -- itself not `None`, so the
    # weaker assertion would stay green with the traceback actually empty.
    assert record.exc_info[0] is RuntimeError
    assert isinstance(record.exc_info[1], RuntimeError)
    assert "kaboom" in str(record.exc_info[1])
    assert record.exc_info[2] is not None


@pytest.mark.asyncio
async def test_cancelled_task_logs_no_warning_or_error(caplog):
    """AC 4: a cancelled task logs nothing at WARNING or above (a DEBUG
    record is fine), and the done-callback does not raise out."""

    async def _sleep_forever() -> None:
        await asyncio.sleep(10)

    with caplog.at_level(logging.DEBUG, logger="app.core.tasks"):
        task = tasks.spawn_background_task(_sleep_forever(), name="ac4-cancel")
        await asyncio.sleep(0)  # let it actually start running
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    records = [r for r in caplog.records if r.name == "app.core.tasks"]
    assert all(r.levelno < logging.WARNING for r in records)
    assert tasks._background_tasks == set()

    # Positive-branch proof: filtering on WARNING+ alone would stay green
    # even if `task.exception()` were ever moved ahead of `task.cancelled()`,
    # because the resulting CancelledError is swallowed by asyncio's own
    # `call_exception_handler` and logged under the "asyncio" logger, not
    # "app.core.tasks". This asserts the cancelled() branch was actually
    # taken -- a DEBUG record naming this task must exist.
    debug_records = [r for r in records if r.levelno == logging.DEBUG]
    assert any("ac4-cancel" in r.getMessage() for r in debug_records)

    # Direct-invocation proof: call the callback again on the already-done,
    # already-cancelled task. A swapped ordering would call task.exception()
    # first, which re-raises CancelledError on a cancelled task -- catching
    # BaseException (not Exception) here is deliberate, since CancelledError
    # derives from BaseException, and a regression must not be misread as
    # cancellation of this test's own task.
    try:
        tasks._on_task_done(task)
    except BaseException as exc:
        pytest.fail(f"_on_task_done raised on an already-cancelled task: {exc!r}")


@pytest.mark.asyncio
async def test_discard_happens_before_logging_even_if_logging_raises(monkeypatch):
    """AC 5: the done-callback discards the task from `_background_tasks`
    BEFORE it logs -- so a monkeypatched `log.warning` that itself raises
    must not leave the reference behind."""

    async def _boom() -> None:
        raise RuntimeError("kaboom")

    def _boom_logger(*args, **kwargs):
        raise RuntimeError("logging blew up")

    monkeypatch.setattr(tasks.log, "warning", _boom_logger)

    task = tasks.spawn_background_task(_boom(), name="ac5-boom")
    with pytest.raises(RuntimeError, match="kaboom"):
        await task

    assert tasks._background_tasks == set()


# --- AC 9-11: the AST invariant scan ------------------------------------------

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_DEFINER = "core/tasks.py"
_MARKER = "# background-task-ok:"


def _find_unmarked_create_task_calls(source: str, label: str) -> list[str]:
    lines = source.splitlines()
    findings = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_create_task = (
            (isinstance(func, ast.Attribute) and func.attr == "create_task")
            or (isinstance(func, ast.Name) and func.id == "create_task")
        )
        if not is_create_task:
            continue
        context = lines[max(0, node.lineno - 2):node.lineno]
        if not any(_MARKER in line for line in context):
            findings.append(f"{label}:{node.lineno}")
    return findings


def _scan_app_dir() -> list[str]:
    findings: list[str] = []
    for path in _APP_DIR.rglob("*.py"):
        label = str(path.relative_to(_APP_DIR)).replace("\\", "/")
        if label == _DEFINER:
            continue
        source = path.read_text(encoding="utf-8")
        findings.extend(_find_unmarked_create_task_calls(source, label))
    return findings


def test_no_unmarked_create_task_calls_in_app():
    """AC 9: every backend/app/**/*.py fire-and-forget `create_task` call
    must either route through `app.core.tasks.spawn_background_task()` (the
    definer, `core/tasks.py`, is excluded from the scan) or carry a
    `# background-task-ok: <reason>` marker on its own line or the line
    directly above it."""
    findings = _scan_app_dir()
    assert not findings, (
        f"Found unmarked create_task call(s): {findings}. Use "
        f"app.core.tasks.spawn_background_task() or annotate the call with "
        f"'{_MARKER} <reason>'."
    )


def test_scanner_detects_an_unmarked_create_task_call():
    """AC 10: the scanner must actually flag a violation, not merely always
    pass -- proven directly against a synthetic source string."""
    source = "import asyncio\n\n\nasync def f():\n    asyncio.create_task(g())\n"
    findings = _find_unmarked_create_task_calls(source, "synthetic.py")
    assert findings == ["synthetic.py:5"]


def test_scanner_accepts_a_marked_create_task_call():
    """AC 10: the same synthetic call, now preceded by the marker comment,
    must not be flagged."""
    source = (
        "import asyncio\n\n\n"
        "async def f():\n"
        "    # background-task-ok: test fixture\n"
        "    asyncio.create_task(g())\n"
    )
    findings = _find_unmarked_create_task_calls(source, "synthetic.py")
    assert findings == []


def test_scanner_detects_a_bare_imported_create_task_call():
    """AC 10: the scanner's `ast.Name` branch (a bare `create_task` reached
    via `from asyncio import create_task`, as opposed to the `ast.Attribute`
    branch covering `asyncio.create_task`) must be proven separately --
    otherwise it is implemented but unproven and could be silently broken
    (or deleted by the dead-code check) with every other test still green."""
    source = (
        "from asyncio import create_task\n\n\n"
        "async def f():\n"
        "    create_task(g())\n"
    )
    findings = _find_unmarked_create_task_calls(source, "synthetic.py")
    assert findings == ["synthetic.py:5"]


def test_scanner_ignores_docstring_mention_in_vector_store():
    """AC 11: `core/vector_store.py`'s docstring mentions `asyncio.create_task`
    in prose -- an AST walk must not treat that as a call."""
    path = _APP_DIR / "core" / "vector_store.py"
    source = path.read_text(encoding="utf-8")
    findings = _find_unmarked_create_task_calls(source, "core/vector_store.py")
    assert findings == []

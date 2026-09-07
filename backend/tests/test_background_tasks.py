"""Tests for `app.core.tasks`: the strong-reference registry for
fire-and-forget asyncio tasks (Spec 032), and the AST-based invariant scan
that keeps every future `backend/app/**/*.py` fire-and-forget call site
routed through it (or explicitly marked as an exception).
"""

import ast
import asyncio
import logging
import threading
from pathlib import Path

import pytest

from app.core import tasks


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
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    records = [r for r in caplog.records if r.name == "app.core.tasks"]
    assert all(r.levelno < logging.WARNING for r in records)
    assert tasks._background_tasks == set()

    debug_records = [r for r in records if r.levelno == logging.DEBUG]
    assert any("ac4-cancel" in r.getMessage() for r in debug_records)

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




async def _swallow_n_cancels(n: int) -> None:
    """A task that swallows its first `n` CancelledError deliveries and keeps
    running -- the "uncancellable task" the drain's time budget exists for.
    The (n+1)-th cancel is honoured, which is how a test cleans it up."""
    budget = n
    while True:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            if budget <= 0:
                raise
            budget -= 1


async def _drain_leftover(task: asyncio.Task) -> None:
    """conftest.py clears `_background_tasks` WITHOUT cancelling, so a test
    that spawns a long-lived task must retire its own leftovers."""
    task.cancel()
    try:
        await task
    except BaseException:  # noqa: BLE001 -- CancelledError is a BaseException
        pass


@pytest.mark.asyncio
async def test_cancel_all_returns_empty_list_for_empty_registry():
    """AC 7: an empty registry is a normal, silent no-op."""
    assert tasks._background_tasks == set()
    assert await tasks.cancel_all() == []


@pytest.mark.asyncio
async def test_cancel_all_drains_a_pending_task_and_empties_the_registry():
    """AC 1: after the drain the registry is empty and the task has actually
    reached a done state.

    These three assertions ARE the coverage for AC 1's "no `Task was
    destroyed but it is pending!`" half: that message is emitted by
    `asyncio.Task.__del__` on the `asyncio` logger at garbage-collection
    time, so a `caplog` assertion inside this test could never observe it and
    would pass even against a no-op `cancel_all()`. A task proven `done()`
    and unregistered cannot produce it.
    """

    async def _sleep_forever() -> None:
        await asyncio.sleep(30)

    task = tasks.spawn_background_task(_sleep_forever(), name="drain-me")
    await asyncio.sleep(0)

    stragglers = await tasks.cancel_all()

    assert stragglers == []
    assert task.done()
    assert task.cancelled()
    assert tasks._background_tasks == set()


@pytest.mark.asyncio
async def test_cancel_all_abandons_an_uncancellable_task_within_the_budget(caplog):
    """AC 3: a task that swallows CancelledError does not stall shutdown --
    `cancel_all()` returns inside the wall-clock bound, names the straggler
    in its return value, and emits exactly one WARNING for it."""
    task = tasks.spawn_background_task(_swallow_n_cancels(1), name="stubborn")
    await asyncio.sleep(0)

    with caplog.at_level(logging.WARNING, logger="app.core.tasks"):
        start = asyncio.get_running_loop().time()
        stragglers = await tasks.cancel_all(timeout=1.0)
        elapsed = asyncio.get_running_loop().time() - start

    assert stragglers == ["stubborn"]
    assert elapsed < 2.0
    assert not task.done()

    warnings = [
        r for r in caplog.records
        if r.name == "app.core.tasks" and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "stubborn" in warnings[0].getMessage()

    await _drain_leftover(task)


@pytest.mark.asyncio
async def test_cancel_all_returns_normally_when_a_task_raises_during_cancellation():
    """AC 7: a task whose cleanup raises a non-CancelledError exception must
    not propagate out of the drain -- `asyncio.wait()` returns it in `done`
    rather than re-raising, which is what makes "never raises" true."""

    async def _raises_on_cancel() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise RuntimeError("cleanup blew up") from None

    task = tasks.spawn_background_task(_raises_on_cancel(), name="raises-on-cancel")
    await asyncio.sleep(0)

    assert await tasks.cancel_all() == []
    assert task.done()
    assert isinstance(task.exception(), RuntimeError)


@pytest.mark.asyncio
async def test_cancel_all_returns_normally_for_an_already_done_task():
    """AC 7: an already-settled task is filtered out before `asyncio.wait()`
    -- cancelling it would be a no-op, and passing an empty set to
    `asyncio.wait()` raises ValueError."""

    async def _noop() -> None:
        return None

    task = tasks.spawn_background_task(_noop(), name="already-done")
    await task
    assert task.done()

    assert await tasks.cancel_all(extra=[task]) == []


@pytest.mark.asyncio
async def test_cancel_all_cancels_an_extra_task_and_tolerates_none():
    """AC 5: the `extra` channel -- how the drain reaches `_active_load`,
    which is deliberately not in the registry -- cancels what it is handed
    and treats a `None` (no load in flight) as nothing to do."""

    async def _sleep_forever() -> None:
        await asyncio.sleep(30)

    # background-task-ok: test fixture standing in for local_setup._active_load
    task = asyncio.create_task(_sleep_forever(), name="extra-load")
    await asyncio.sleep(0)

    assert await tasks.cancel_all(extra=[task, None]) == []
    assert task.cancelled()


@pytest.mark.asyncio
async def test_peek_active_load_exposes_the_in_flight_load_task():
    """AC 5: `peek_active_load()` is the read-only accessor the drain uses;
    it returns None when nothing is loading and the task when one is."""
    from app.stt import local_setup

    async def _sleep_forever() -> None:
        await asyncio.sleep(30)

    assert local_setup.peek_active_load() is None

    # background-task-ok: test fixture; retired by _drain_leftover below
    task = asyncio.create_task(_sleep_forever(), name="fake-active-load")
    local_setup._active_load = (object(), task)
    try:
        assert local_setup.peek_active_load() is task
    finally:
        await _drain_leftover(task)


@pytest.mark.asyncio
async def test_drain_reaches_the_active_load_created_by_real_ensure_local_ready(
    monkeypatch,
):
    """AC 5 through the PRODUCTION path, not a hand-planted tuple.

    The other `_active_load` tests assign `(object(), task)` directly, so a
    regression that changed the tuple's shape, stopped assigning
    `_active_load`, or moved the task behind a different accessor would leave
    them green while the drain reached nothing. Here `ensure_local_ready()`
    itself creates the task (local_setup.py:369) and awaits it through
    `asyncio.shield()` (:372); only `provider._get_model` -- the actual model
    load -- is stubbed.
    """
    import app.stt
    from app.core.types import ProviderMode
    from app.stt import local_setup
    from app.stt.config import STTSettings

    release = threading.Event()
    loop = asyncio.get_running_loop()
    entered_load = asyncio.Event()

    class _BlockingProvider:
        is_loaded = False

        def _get_model(self) -> None:
            loop.call_soon_threadsafe(entered_load.set)
            release.wait(10)

        def cleanup(self) -> None:
            pass

    provider = _BlockingProvider()
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)
    monkeypatch.setattr(app.stt, "get_provider", lambda mode, stt_settings: provider)
    monkeypatch.setattr(app.stt, "peek_local_provider", lambda: provider)

    # background-task-ok: test fixture standing in for the prewarm caller;
    waiter = asyncio.create_task(
        local_setup.ensure_local_ready(STTSettings(mode=ProviderMode.LOCAL)),
        name="real-prewarm-caller",
    )
    try:
        await asyncio.wait_for(entered_load.wait(), timeout=10)
        load_task = local_setup.peek_active_load()
        assert load_task is not None, "ensure_local_ready() never populated _active_load"

        assert await tasks.cancel_all(extra=[local_setup.peek_active_load()]) == []
        assert load_task.done()
        assert load_task.cancelled()
    finally:
        release.set()
        await _drain_leftover(waiter)


@pytest.mark.asyncio
async def test_drain_resets_the_prewarm_crash_guard_counter(monkeypatch, tmp_path):
    """Pins the behaviour change to Spec 023's crash-loop guard that this
    spec introduces (plan -> Risks, ADR 021 -> Consequences): because the
    drain now cancels `local-stt-prewarm-startup` and WAITS for it,
    `_prewarm_then_clear_crash_guard`'s `finally` actually runs, so quitting
    mid-warm-up resets the on-disk counter to 0 instead of leaving it
    accumulating toward the lockout. Deliberate -- a quit is not a crash.
    """
    from app.core.types import ProviderMode
    from app.stt import local_setup
    from app.stt.config import STTSettings

    monkeypatch.setattr("app.core.app_paths.resolve_app_data_root", lambda: tmp_path)

    started = asyncio.Event()

    async def _blocking_ensure_local_ready(stt_settings) -> None:
        started.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(local_setup, "ensure_local_ready", _blocking_ensure_local_ready)
    local_setup._write_consecutive_incomplete_prewarms(1)

    task = tasks.spawn_background_task(
        local_setup._prewarm_then_clear_crash_guard(STTSettings(mode=ProviderMode.LOCAL)),
        name="local-stt-prewarm-startup",
    )
    await started.wait()
    assert local_setup._read_consecutive_incomplete_prewarms() == 1

    assert await tasks.cancel_all() == []

    assert task.cancelled()
    assert local_setup._read_consecutive_incomplete_prewarms() == 0




@pytest.fixture
def teardown_probe(monkeypatch):
    """Records the real call order of teardown's steps.

    `lifespan()` startup is otherwise left real. `_warm_gpu_probe_cache` is
    stubbed because conftest no-ops prewarm and the indexer but NOT
    `probe_gpu`, so an unpatched startup runs a genuine GPU probe in a
    worker thread. The recorder is stubbed too: `cleanup()` is the last
    release step and the only one with a real OS resource (the audio stream)
    behind it, so it is recorded rather than left unobserved.
    """
    import app.audio.recorder
    import app.embeddings
    import app.main
    import app.stt

    order: list[str] = []

    async def _noop_probe() -> None:
        return None

    class _FakeRecorder:
        def __init__(self, audio_settings) -> None:
            pass

        def cleanup(self) -> None:
            order.append("recorder_cleanup")

    monkeypatch.setattr(app.main, "_warm_gpu_probe_cache", _noop_probe)
    monkeypatch.setattr(app.audio.recorder, "MicrophoneRecorder", _FakeRecorder)
    monkeypatch.setattr(app.stt, "clear_cache", lambda: order.append("clear_stt"))
    monkeypatch.setattr(
        app.embeddings, "clear_cache", lambda: order.append("clear_embeddings")
    )
    return order


@pytest.mark.asyncio
async def test_lifespan_drains_background_tasks_before_clearing_caches(teardown_probe):
    """AC 2: every background task has reached a done state before the first
    `clear_*()` call -- asserted on the actual recorded call order, with a
    task that is still running at `yield`."""
    from app.main import app as fastapi_app
    from app.main import lifespan

    async def _probe() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            teardown_probe.append("probe-cancelled")
            raise

    async with lifespan(fastapi_app):
        task = tasks.spawn_background_task(_probe(), name="teardown-probe")
        await asyncio.sleep(0.01)

    assert teardown_probe == [
        "probe-cancelled",
        "clear_stt",
        "clear_embeddings",
        "recorder_cleanup",
    ]
    assert task.done()
    assert tasks._background_tasks == set()


@pytest.mark.asyncio
async def test_lifespan_still_clears_caches_when_the_drain_times_out(
    teardown_probe, caplog
):
    """AC 3 + AC 4: teardown returns inside the wall-clock bound against the
    real 1.0s budget, and the model-release step is never skipped or
    truncated by a straggler -- all three `clear_*()` calls still run, and
    the straggler is named at WARNING."""
    from app.main import app as fastapi_app
    from app.main import lifespan

    loop = asyncio.get_running_loop()

    with caplog.at_level(logging.WARNING, logger="app.core.tasks"):
        async with lifespan(fastapi_app):
            task = tasks.spawn_background_task(_swallow_n_cancels(1), name="stubborn-quit")
            await asyncio.sleep(0.01)
            start = loop.time()
    elapsed = loop.time() - start

    assert elapsed < 2.0
    assert teardown_probe == [
        "clear_stt",
        "clear_embeddings",
        "recorder_cleanup",
    ]
    assert not task.done()
    assert any("stubborn-quit" in r.getMessage() for r in caplog.records)

    await _drain_leftover(task)


@pytest.mark.asyncio
async def test_lifespan_release_steps_do_not_skip_each_other_on_failure(
    teardown_probe, monkeypatch, caplog
):
    """GitHub review finding 1: the release block must always COMPLETE, not
    just start. A raising `clear_stt()` used to skip `clear_embeddings()`
    and `recorder.cleanup()` -- leaking the audio stream, the one release
    step with a real OS resource behind it."""
    import app.stt
    from app.main import app as fastapi_app
    from app.main import lifespan

    def _boom() -> None:
        teardown_probe.append("clear_stt_raised")
        raise RuntimeError("model release blew up")

    monkeypatch.setattr(app.stt, "clear_cache", _boom)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        async with lifespan(fastapi_app):
            pass

    assert teardown_probe == [
        "clear_stt_raised",
        "clear_embeddings",
        "recorder_cleanup",
    ]
    assert any("STT cache" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_lifespan_cancels_the_active_load_with_no_registered_task(teardown_probe):
    """AC 5: an in-flight `_active_load` is cancelled and done before the
    cache-clear block, even though it is not in `_background_tasks` -- the
    exact case where `clear_stt()` would otherwise hit `cleanup()`'s
    busy-lock skip path and free nothing."""
    from app.main import app as fastapi_app
    from app.main import lifespan
    from app.stt import local_setup

    async def _load() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            teardown_probe.append("active-load-cancelled")
            raise

    async with lifespan(fastapi_app):
        # background-task-ok: test fixture mirroring local_setup's own bare
        load_task = asyncio.create_task(_load(), name="fake-prewarm-load")
        local_setup._active_load = (object(), load_task)
        await asyncio.sleep(0.01)
        assert tasks._background_tasks == set()

    assert teardown_probe == [
        "active-load-cancelled",
        "clear_stt",
        "clear_embeddings",
        "recorder_cleanup",
    ]
    assert load_task.done()



_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_DEFINER = "core/tasks.py"
_MARKER = "# background-task-ok:"


def _find_unmarked_create_task_calls(source: str, label: str) -> list[str]:
    """Flag unmarked `create_task`/`ensure_future` calls.

    `asyncio.ensure_future(coro)` wraps a coroutine in a Task with the exact
    same weak-reference semantics as `create_task` (GitHub review on PR #37,
    finding 2) -- reaching for it would reintroduce this spec's bug while
    passing a scanner that only knew about `create_task`. No such call
    exists in `backend/app/` today; this closes the hole pre-emptively.
    """
    lines = source.splitlines()
    findings = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_flagged_call = (
            (isinstance(func, ast.Attribute) and func.attr in ("create_task", "ensure_future"))
            or (isinstance(func, ast.Name) and func.id in ("create_task", "ensure_future"))
        )
        if not is_flagged_call:
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
    """AC 9: every backend/app/**/*.py fire-and-forget `create_task` or
    `ensure_future` call must either route through
    `app.core.tasks.spawn_background_task()` (the definer, `core/tasks.py`,
    is excluded from the scan) or carry a `# background-task-ok: <reason>`
    marker on its own line or the line directly above it."""
    findings = _scan_app_dir()
    assert not findings, (
        f"Found unmarked create_task()/ensure_future() call(s): {findings}. "
        f"Use app.core.tasks.spawn_background_task() or annotate the call "
        f"with '{_MARKER} <reason>'."
    )


def test_scanner_detects_an_unmarked_create_task_call():
    """AC 10: the scanner must actually flag a violation, not merely always
    pass -- proven directly against a synthetic source string."""
    source = "import asyncio\n\n\nasync def f():\n    asyncio.create_task(g())\n"
    findings = _find_unmarked_create_task_calls(source, "synthetic.py")
    assert findings == ["synthetic.py:5"]


def test_scanner_detects_an_unmarked_ensure_future_call():
    """GitHub review on PR #37, finding 2: `asyncio.ensure_future(coro)` has
    the exact same weak-reference semantics as `create_task` and must be
    flagged too -- proven directly, the same way the `create_task` case
    above is proven, rather than trusting the widened matcher on faith."""
    source = "import asyncio\n\n\nasync def f():\n    asyncio.ensure_future(g())\n"
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
    """AC 11: `transcripts/vector_store.py`'s docstring mentions
    `asyncio.create_task` in prose -- an AST walk must not treat that as a
    call."""
    path = _APP_DIR / "transcripts" / "vector_store.py"
    source = path.read_text(encoding="utf-8")
    findings = _find_unmarked_create_task_calls(source, "transcripts/vector_store.py")
    assert findings == []

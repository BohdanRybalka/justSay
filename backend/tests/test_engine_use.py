import asyncio
import logging
import threading

import pytest

from app.stt.engine_use import EngineUse


class _Engine:
    def __init__(self):
        self.stale = False
        self.ready = True
        self.unloads = 0
        self.use = EngineUse(
            is_stale=lambda: self.stale, is_ready=lambda: self.ready, unload=self._unload
        )

    def _unload(self):
        self.unloads += 1
        self.stale = False


def _settle_in_thread(use: EngineUse) -> tuple[threading.Thread, threading.Event]:
    settled = threading.Event()

    def _run():
        use.settle()
        settled.set()

    thread = threading.Thread(target=_run)
    thread.start()
    return thread, settled


@pytest.mark.asyncio
async def test_a_stale_engine_is_unloaded_only_after_the_running_transcription_ends():
    engine = _Engine()
    await engine.use.enter(lambda: None)
    engine.stale = True

    thread, settled = _settle_in_thread(engine.use)

    assert not settled.wait(0.3)
    assert engine.unloads == 0

    engine.use.leave()

    assert settled.wait(2)
    thread.join(2)
    assert engine.unloads == 1


def test_an_engine_that_is_not_stale_is_left_loaded():
    engine = _Engine()

    engine.use.settle()

    assert engine.unloads == 0


@pytest.mark.asyncio
async def test_a_transcription_waits_for_the_restart_to_finish_before_it_is_counted():
    engine = _Engine()
    engine.stale = True
    unloading = threading.Event()
    finish_unload = threading.Event()

    def _slow_unload():
        unloading.set()
        finish_unload.wait(5)
        engine._unload()

    use = EngineUse(
        is_stale=lambda: engine.stale, is_ready=lambda: engine.ready, unload=_slow_unload
    )
    thread, _ = _settle_in_thread(use)
    assert unloading.wait(2)

    entered = []

    async def _enter():
        entered.append(await use.enter(lambda: "ready"))

    task = asyncio.ensure_future(_enter())
    await asyncio.sleep(0.3)
    assert entered == []

    finish_unload.set()
    await asyncio.wait_for(task, 2)
    thread.join(2)

    assert entered == ["ready"]
    use.leave()


@pytest.mark.asyncio
async def test_enter_prepares_again_while_the_engine_is_not_ready():
    engine = _Engine()
    engine.ready = False
    calls = []

    def _prepare():
        calls.append(1)
        engine.ready = len(calls) == 2
        return len(calls)

    assert await engine.use.enter(_prepare) == 2
    engine.use.leave()


@pytest.mark.asyncio
async def test_a_release_waits_for_the_last_running_transcription():
    engine = _Engine()
    released = []
    await engine.use.enter(lambda: None)
    await engine.use.enter(lambda: None)

    engine.use.release_when_idle(lambda: released.append(1))
    engine.use.leave()
    assert released == []

    engine.use.leave()
    assert released == [1]


def test_a_release_runs_at_once_when_nothing_is_running():
    engine = _Engine()
    released = []

    engine.use.release_when_idle(lambda: released.append(1))

    assert released == [1]


@pytest.mark.asyncio
async def test_a_transcription_that_begins_before_the_release_cancels_it():
    engine = _Engine()
    released = []
    await engine.use.enter(lambda: None)
    engine.use.release_when_idle(lambda: released.append(1))

    await engine.use.enter(lambda: None)
    engine.use.leave()
    engine.use.leave()

    assert released == []


def test_a_release_that_raises_is_logged_and_does_not_reach_the_caller(caplog):
    engine = _Engine()

    def _fail():
        raise RuntimeError("terminate refused")

    with caplog.at_level(logging.WARNING):
        engine.use.release_when_idle(_fail)

    assert "Releasing the local engine failed" in caplog.text

"""Counts the transcriptions running on one local engine, so a restart waits for them.

An engine loaded with other settings than the current ones is restarted only when no
transcription runs on it: every request that began before the change finishes on the old
engine, and the new one loads after it. A release waits for them the same way.
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class EngineUse:
    """The running-transcription count of one engine, with the restart and release that wait on it.

    ``is_stale`` says the loaded engine no longer matches the settings, ``is_ready`` says it can
    transcribe now, ``unload`` frees it and may block.
    """

    def __init__(
        self,
        is_stale: Callable[[], bool],
        is_ready: Callable[[], bool],
        unload: Callable[[], None],
    ) -> None:
        self._is_stale = is_stale
        self._is_ready = is_ready
        self._unload = unload
        self._changed = threading.Condition()
        self._running = 0
        self._restarting = False

    def settle(self) -> None:
        """Block until the engine is not stale, unloading a stale one once nothing runs on it."""
        with self._changed:
            while self._restarting or (self._is_stale() and self._running):
                self._changed.wait()
            if not self._is_stale():
                return
            self._restarting = True
        self._unload_exclusively()

    async def enter(self, prepare: Callable[[], T]) -> T:
        """Count a transcription in once ``prepare`` has left a ready engine, and return its result.

        Pair every call with ``leave``. A restart or release that lands between the two is
        waited for again, so the count is never taken on an engine about to go.
        """
        while True:
            prepared = await asyncio.to_thread(prepare)
            with self._changed:
                if not self._restarting and self._is_ready():
                    self._running += 1
                    return prepared

    def leave(self) -> None:
        """Count a finished transcription out."""
        with self._changed:
            self._running -= 1
            self._changed.notify_all()

    def release_when_idle(self, still_unused: Callable[[], bool]) -> None:
        """Unload the engine on a thread once nothing runs on it and a load in flight is over.

        Nothing is unloaded if ``still_unused`` is false by then.
        """
        threading.Thread(target=self._release, args=(still_unused,), daemon=True).start()

    def _release(self, still_unused: Callable[[], bool]) -> None:
        with self._changed:
            while self._restarting or self._running:
                self._changed.wait()
            if not still_unused():
                return
            self._restarting = True
        self._unload_exclusively()

    def _unload_exclusively(self) -> None:
        try:
            self._unload()
        except Exception:
            log.warning("Releasing the local engine failed", exc_info=True)
        finally:
            with self._changed:
                self._restarting = False
                self._changed.notify_all()

"""Background transcription jobs, kept in memory and run one at a time, oldest first.

A job never touches the clipboard and goes in pieces, each held back while a dictation is being
processed. Progress counts finished pieces, plus an estimate inside the current one from this
machine's own speed for the routed model. Jobs die with the app; ``remove_leftover_files``
deletes their scratch files at the next start.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import statistics
import time
import uuid
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.core import tasks
from app.core.audio_formats import UNREADABLE_HERE
from app.core.errors import (
    ConfigurationError,
    JustSayError,
    NotReadyError,
    ResourceUnavailableError,
)
from app.core.scratch import discard_scratch_file
from app.pipeline.service import process_audio
from app.transcripts import history

log = logging.getLogger(__name__)

JobStage = Literal["queued", "transcribing", "saving", "done", "failed", "cancelled"]
JobKind = Literal["file"]
RemovalOutcome = Literal["cancelled", "dismissed"]

ESTIMATE_CAP = 0.95
SPEED_SAMPLE_ROWS = 50
FINISHED_SHOWN_SECONDS = 60.0
DICTATION_POLL_SECONDS = 0.05
JOB_FILE_PREFIX = "job_"
FILE_LANGUAGE = "auto"

NO_SPEECH_REASON = "We didn't hear any speech in this file"
NO_KEY_REASON = "Add an API key in Settings"
FAILED_REASON = "Couldn't turn this file into text. Try again"
BUSY_REASON = "The transcription service is busy. Try again in a few minutes"
LIMIT_REACHED_REASON = "Your API key has used up its limit for now. Try again in a few hours"
LIMIT_REACHED_SECONDS = 3600.0

_CANCELLABLE: frozenset[JobStage] = frozenset({"queued", "transcribing"})
_EXPIRING: frozenset[JobStage] = frozenset({"done", "cancelled"})


class JobView(BaseModel):
    """One job as ``GET /jobs`` shows it; ``progress`` is 0–1, or null when it cannot be told."""

    id: str
    kind: JobKind
    name: str
    stage: JobStage
    progress: float | None
    error: str | None
    entry_id: str | None


class DictationGate:
    """Counts dictations being processed, so a job can hold its audio back until there are none."""

    def __init__(self) -> None:
        self._in_flight = 0

    @contextmanager
    def dictating(self) -> Generator[None, None, None]:
        self._in_flight += 1
        try:
            yield
        finally:
            self._in_flight -= 1

    async def wait_for_none(self) -> None:
        while self._in_flight:
            await asyncio.sleep(DICTATION_POLL_SECONDS)


dictation_gate = DictationGate()


@dataclass
class _Job:
    id: str
    kind: JobKind
    name: str
    path: Path
    stage: JobStage = "queued"
    error: str | None = None
    entry_id: str | None = None
    piece_started_at: float | None = None
    pieces_done: int = 0
    pieces_total: int = 1
    paused: bool = False
    expected_seconds: float | None = None
    finished_at: float | None = None
    task: asyncio.Task | None = None


class _JobCancelledError(Exception):
    pass


class _JobObserver:
    def __init__(self, queue: JobQueue, job: _Job) -> None:
        self._queue = queue
        self._job = job

    async def before_transcribe(self, model_name: str, audio_duration: float | None) -> None:
        self._job.expected_seconds = await _expected_seconds(model_name, audio_duration)

    async def before_piece(self, index: int, total: int) -> None:
        await self._queue.gate.wait_for_none()
        self._stop_if_cancelled()
        self._job.pieces_total = total
        self._job.piece_started_at = self._queue.clock()

    async def pause(self, seconds: float) -> None:
        self._stop_if_cancelled()
        self._job.paused = True
        try:
            await self._queue.sleep(seconds)
        finally:
            self._job.paused = False
        await self._queue.gate.wait_for_none()
        self._stop_if_cancelled()

    def piece_done(self, done: int, total: int) -> None:
        self._job.pieces_done = done

    def before_save(self) -> None:
        self._stop_if_cancelled()
        self._job.stage = "saving"

    def _stop_if_cancelled(self) -> None:
        if self._job.stage == "cancelled":
            raise _JobCancelledError

    def saved(self, entry_id: str) -> None:
        self._job.entry_id = entry_id


async def _expected_seconds(model_name: str, audio_duration: float | None) -> float | None:
    """Seconds ``model_name`` should take here for this much audio, from its median recent speed."""
    if not audio_duration or audio_duration <= 0:
        return None
    try:
        speeds = await asyncio.to_thread(history.processing_speeds, model_name, SPEED_SAMPLE_ROWS)
    except (sqlite3.Error, OSError):
        log.warning("Reading past speeds for %s failed; no estimate", model_name, exc_info=True)
        return None
    if not speeds:
        return None
    return audio_duration / statistics.median(speeds) / 1000


def _reason_for_unavailable(refusal: ResourceUnavailableError) -> str:
    if refusal.message == UNREADABLE_HERE:
        return UNREADABLE_HERE
    retry_after = (refusal.headers or {}).get("Retry-After")
    if retry_after is None:
        return FAILED_REASON
    try:
        wait = float(retry_after)
    except ValueError:
        return BUSY_REASON
    return LIMIT_REACHED_REASON if wait >= LIMIT_REACHED_SECONDS else BUSY_REASON


def remove_leftover_files(temp_dir: Path) -> None:
    """Delete the scratch files of jobs that were running when the app last quit."""
    for path in temp_dir.glob(f"{JOB_FILE_PREFIX}*"):
        discard_scratch_file(path)


class JobQueue:
    """Jobs by id; a failed one stays until dismissed, a done or cancelled one a minute."""

    def __init__(
        self,
        temp_dir: Path,
        gate: DictationGate = dictation_gate,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._temp_dir = temp_dir
        self._jobs: dict[str, _Job] = {}
        self._turn = asyncio.Lock()
        self.gate = gate
        self.clock = clock
        self.sleep = sleep

    def add_file(self, content: bytes, name: str) -> str:
        """Keep ``content`` in a scratch file and queue it for transcription; answer the job id."""
        job_id = uuid.uuid4().hex
        path = self._temp_dir / f"{JOB_FILE_PREFIX}{job_id}{Path(name).suffix.lower()}"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        job = _Job(id=job_id, kind="file", name=name, path=path)
        self._jobs[job_id] = job
        job.task = tasks.spawn_background_task(self._run(job), name=f"job-{job_id}")
        return job_id

    def views(self) -> list[JobView]:
        """Newest first."""
        now = self.clock()
        for job in list(self._jobs.values()):
            finished_at = job.finished_at
            if job.stage in _EXPIRING and finished_at is not None:
                if now - finished_at > FINISHED_SHOWN_SECONDS:
                    del self._jobs[job.id]
        return [self._view(job, now) for job in reversed(self._jobs.values())]

    def cancel_or_dismiss(self, job_id: str) -> RemovalOutcome | None:
        """Cancel a queued or transcribing job, forget a finished one; ``None`` for an unknown id.

        A piece already sent cannot be recalled, so a transcribing job keeps its turn until that
        answer comes and then stops unsaved; a paused one stops at once. ``NotReadyError`` while
        saving.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.stage == "saving":
            raise NotReadyError("This file is already being saved to History")
        if job.stage in _CANCELLABLE:
            if (job.stage == "queued" or job.paused) and job.task is not None:
                job.task.cancel()
            job.stage = "cancelled"
            job.finished_at = self.clock()
            return "cancelled"
        del self._jobs[job_id]
        return "dismissed"

    def _view(self, job: _Job, now: float) -> JobView:
        progress = self._progress(job, now)
        return JobView(
            id=job.id,
            kind=job.kind,
            name=job.name,
            stage=job.stage,
            progress=progress,
            error=job.error,
            entry_id=job.entry_id,
        )

    @staticmethod
    def _progress(job: _Job, now: float) -> float | None:
        if job.stage == "done":
            return 1.0
        if job.stage != "transcribing" or job.piece_started_at is None:
            return None
        within = 0.0
        if job.expected_seconds:
            per_piece = job.expected_seconds / job.pieces_total
            within = min(ESTIMATE_CAP, (now - job.piece_started_at) / per_piece)
        elif job.pieces_total == 1:
            return None
        return min(ESTIMATE_CAP, (job.pieces_done + within) / job.pieces_total)

    async def _run(self, job: _Job) -> None:
        try:
            async with self._turn:
                job.stage = "transcribing"
                result = await process_audio(
                    job.path,
                    language=FILE_LANGUAGE,
                    copy_to_clipboard=False,
                    source="file",
                    source_name=job.name,
                    observer=_JobObserver(self, job),
                )
            if result.discarded_reason is not None:
                self._fail(job, NO_SPEECH_REASON)
            elif job.entry_id is None:
                self._fail(job, FAILED_REASON)
            else:
                job.stage = "done"
                job.finished_at = self.clock()
                from app.transcripts import vector_store

                tasks.spawn_background_task(
                    vector_store.run_background_indexer(), name="vector-store-indexer"
                )
        except _JobCancelledError:
            log.info("File job %s cancelled; its answer was not saved", job.id)
        except ConfigurationError:
            self._fail(job, NO_KEY_REASON)
        except ResourceUnavailableError as refusal:
            log.warning("File job %s refused: %s", job.id, refusal.diagnostic or refusal.message)
            self._fail(job, _reason_for_unavailable(refusal))
        except JustSayError as refusal:
            log.warning("File job %s refused: %s", job.id, refusal.diagnostic or refusal.message)
            self._fail(job, FAILED_REASON)
        except Exception:
            log.exception("File job %s crashed", job.id)
            self._fail(job, FAILED_REASON)
        finally:
            discard_scratch_file(job.path)

    def _fail(self, job: _Job, reason: str) -> None:
        if job.stage == "cancelled":
            return
        job.stage = "failed"
        job.error = reason
        job.finished_at = self.clock()

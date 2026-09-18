import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app import __version__
from app.core import tasks
from app.core.auth_middleware import LaunchTokenMiddleware
from app.core.config import settings
from app.core.error_handler import register_error_handlers
from app.core.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

SHUTDOWN_CONNECTION_DRAIN_SECONDS = 2.0

StepResult = TypeVar("StepResult")

try:
    from app.audio.router import router as audio_router
    from app.core.router import router as core_router
    from app.pipeline.router import router as pipeline_router
    from app.preferences.router import router as settings_router
    from app.stt.router import router as stt_router
    from app.transcripts.history_router import router as history_router
    from app.transcripts.words_router import router as words_router
except Exception as e:
    log.critical("Router import failed — sidecar will exit: %s", e, exc_info=True)
    raise


async def _warm_gpu_probe_cache() -> None:
    """Off-thread, exception-swallowing warm-up of gpu_probe's process-
    lifetime cache -- see the lifespan() call site (Spec 028 Item 2, AC 12)."""
    from app.core.gpu_probe import probe_gpu

    try:
        await asyncio.to_thread(probe_gpu)
    except Exception:
        log.warning("GPU probe warm-up failed -- will be probed lazily on first use", exc_info=True)


def _run_optional_step(
    phase: str, step_name: str, step: Callable[[], StepResult]
) -> StepResult | None:
    """Run a lifespan step the app is still useful without, in either half.

    The step is named, a failure is logged at WARNING under ``phase`` and the
    lifespan carries on; what the step returned comes back, or ``None`` when
    it raised. A step the app is *not* useful without is called directly
    instead, so that it still ends the process -- "is the app still useful
    without it?" is the whole test, and which side of it each lifespan step
    falls on is readable from the call sites below.

    A step whose own successful result is ``None`` cannot be told apart from
    a failed one by its return value. Neither caller that reads the result
    has that shape, and the log line says which happened.
    """
    try:
        return step()
    except Exception:
        log.warning(
            "Backend %s: %s failed -- continuing without it", phase, step_name, exc_info=True
        )
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.preferences.user_settings import (
        get_user_settings,
        repair_scratch_output_dir,
        sync_to_runtime,
    )
    from app.transcripts import history

    log.info("Backend startup: version=%s port=%s", __version__, settings.port)
    settings.audio.temp_dir.mkdir(parents=True, exist_ok=True)

    repaired_dir = _run_optional_step(
        "startup", "repairing the history location", repair_scratch_output_dir
    )
    us = get_user_settings()
    history_dir = repaired_dir if repaired_dir is not None else Path(us.output_dir)

    def _open_history_store() -> Path:
        history.bootstrap(history_dir)
        return history_dir

    opened_dir = _run_optional_step("startup", "opening the history store", _open_history_store)
    log.info(
        "History store: %s (%s)",
        history.history_path().parent,
        "open" if opened_dir is not None else "not open -- opens on the first request for it",
    )
    sync_to_runtime(us)
    from app.stt.local_setup import maybe_prewarm_local_at_startup
    _run_optional_step(
        "startup",
        "prewarming the local model",
        lambda: maybe_prewarm_local_at_startup(settings.stt),
    )
    tasks.spawn_background_task(_warm_gpu_probe_cache(), name="gpu-probe-warmup")
    from app.transcripts import vector_store
    tasks.spawn_background_task(vector_store.run_background_indexer(), name="vector-store-indexer")
    from app.audio.meeting_recorder import MeetingRecorder
    from app.audio.recorder import MicrophoneRecorder
    app.state.recorder = MicrophoneRecorder(settings.audio)
    meeting_recorder = _run_optional_step(
        "startup",
        "building the meeting recorder",
        lambda: MeetingRecorder(settings.audio),
    )
    if meeting_recorder is not None:
        app.state.meeting_recorder = meeting_recorder
    yield
    log.info("Backend shutdown: draining background tasks")
    from app.stt.local_setup import peek_active_load
    try:
        await tasks.cancel_all(extra=[peek_active_load()])
    finally:
        log.info("Backend shutdown: releasing model caches")
        from app.embeddings import clear_cache as clear_embeddings
        from app.stt.routing import clear_cache as clear_stt
        release_steps: list[tuple[str, Callable[[], object]]] = [
            ("releasing the STT cache", clear_stt),
            ("releasing the embeddings cache", clear_embeddings),
            ("releasing the audio recorder", lambda: app.state.recorder.cleanup()),
        ]
        if hasattr(app.state, "meeting_recorder"):
            release_steps.append(
                ("releasing the meeting recorder", lambda: app.state.meeting_recorder.cleanup())
            )
        for step_name, step in release_steps:
            _run_optional_step("shutdown", step_name, step)


app = FastAPI(
    title="JustSay Backend",
    version=__version__,
    lifespan=lifespan,
)

register_error_handlers(app)

app.add_middleware(LaunchTokenMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "https://tauri.localhost",
        "http://tauri.localhost",
        "http://localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)

app.include_router(core_router)
app.include_router(settings_router)
app.include_router(history_router)
app.include_router(words_router)
app.include_router(stt_router, prefix="/stt", tags=["STT"])
app.include_router(audio_router, prefix="/audio", tags=["Audio"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])


def _cli() -> None:
    """Entrypoint used by the PyInstaller-frozen sidecar.

    Accepts ``--host`` and ``--port`` so the Tauri shell can pin the bind address
    without depending on the dev-mode `python -m uvicorn` invocation.
    """
    import argparse
    import sys

    import uvicorn

    parser = argparse.ArgumentParser(prog="justsay-backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--log-level", default="warning")
    parser.add_argument(
        "--selftest-sqlite-vec",
        action="store_true",
        help=(
            "Verify the bundled SQLite is new enough for row-value history "
            "paging, that its planner seeks the composite index for a cursored "
            "page rather than walking it, and that the sqlite-vec extension "
            "loads and a KNN query works — against the actual frozen sidecar "
            "binary, then exit. Every check runs and all failures are reported. "
            "Used by release.yml as a permanent CI gate — see ADR 001."
        ),
    )
    parser.add_argument(
        "--selftest-ten-vad",
        action="store_true",
        help=(
            "Verify the neural silence gate is live in this build — that the "
            "TEN VAD library resolves, loads through ctypes, and returns a "
            "verdict on a synthetic probe clip — against the actual frozen "
            "sidecar binary, then exit. Every VAD failure path fails open to "
            "the energy guard, so a bundled-but-unloadable library is silent "
            "at runtime; this is what makes it loud. Used by release.yml as a "
            "permanent CI gate on both platform legs — see ADR 070."
        ),
    )
    args = parser.parse_args()

    if args.selftest_ten_vad:
        from app.audio import vad

        ok, msg = vad.selftest()
        if ok:
            print("OK")
            sys.exit(0)
        print(f"FAIL: {msg}")
        sys.exit(1)

    if args.selftest_sqlite_vec:
        from app.transcripts import vector_store

        ok, msg = vector_store.selftest()
        if ok:
            print("OK")
            sys.exit(0)
        print(f"FAIL: {msg}")
        sys.exit(1)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_graceful_shutdown=SHUTDOWN_CONNECTION_DRAIN_SECONDS,
    )


if __name__ == "__main__":
    _cli()

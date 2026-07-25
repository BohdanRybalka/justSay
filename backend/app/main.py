import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app import __version__
from app.core import tasks
from app.core.auth_middleware import LaunchTokenMiddleware
from app.core.config import settings
from app.core.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

SHUTDOWN_CONNECTION_DRAIN_SECONDS = 2.0

try:
    from app.audio.router import router as audio_router
    from app.core.history_router import router as history_router
    from app.core.router import router as core_router
    from app.core.settings_router import router as settings_router
    from app.core.words_router import router as words_router
    from app.pipeline.router import router as pipeline_router
    from app.stt.router import router as stt_router
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core import history
    from app.core.user_settings import (
        get_user_settings,
        repair_scratch_output_dir,
        sync_to_runtime,
    )

    log.info("Backend startup: version=%s port=%s", __version__, settings.port)
    settings.audio.temp_dir.mkdir(parents=True, exist_ok=True)

    history.bootstrap(repair_scratch_output_dir())
    us = get_user_settings()
    log.info("Data root: %s", history.history_path().parent)
    sync_to_runtime(us)
    from app.stt.local_setup import maybe_prewarm_local_at_startup
    maybe_prewarm_local_at_startup(settings.stt)
    tasks.spawn_background_task(_warm_gpu_probe_cache(), name="gpu-probe-warmup")
    from app.core import vector_store
    tasks.spawn_background_task(vector_store.run_background_indexer(), name="vector-store-indexer")
    from app.audio import MicrophoneRecorder
    app.state.recorder = MicrophoneRecorder(settings.audio)
    yield
    log.info("Backend shutdown: draining background tasks")
    from app.stt.local_setup import peek_active_load
    try:
        await tasks.cancel_all(extra=[peek_active_load()])
    finally:
        log.info("Backend shutdown: releasing model caches")
        from app.embeddings import clear_cache as clear_embeddings
        from app.stt import clear_cache as clear_stt
        for step_name, step in (
            ("STT cache", clear_stt),
            ("embeddings cache", clear_embeddings),
            ("audio recorder", app.state.recorder.cleanup),
        ):
            try:
                step()
            except Exception:
                log.warning(
                    "Backend shutdown: releasing %s failed -- continuing", step_name, exc_info=True
                )


app = FastAPI(
    title="JustSay Backend",
    version=__version__,
    lifespan=lifespan,
)

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
            "Verify the sqlite-vec extension loads and a KNN query works "
            "against the actual frozen sidecar binary, then exit. Used by "
            "release.yml as a permanent CI gate — see ADR 001."
        ),
    )
    args = parser.parse_args()

    if args.selftest_sqlite_vec:
        from app.core import vector_store

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

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

# Set up logging before any router imports so startup crashes are captured in the log file.
setup_logging()
log = logging.getLogger(__name__)

try:
    from app.core.router import router as core_router
    from app.core.settings_router import router as settings_router
    from app.core.history_router import router as history_router
    from app.core.words_router import router as words_router
    from app.stt.router import router as stt_router
    from app.audio.router import router as audio_router
    from app.pipeline.router import router as pipeline_router
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
    from pathlib import Path

    from app.core import history
    from app.core.user_settings import get_user_settings, sync_to_runtime

    log.info("Backend startup: version=%s port=%s", __version__, settings.port)
    settings.audio.temp_dir.mkdir(parents=True, exist_ok=True)

    us = get_user_settings()
    # Bootstrap history before any worker thread can call save_entry.
    history.bootstrap(Path(us.output_dir))
    # Cheap visibility: lets a developer immediately tell, from backend.log
    # or console output, whether ~/.justsay or ~/.justsay-dev is active.
    log.info("Data root: %s", history.history_path().parent)
    # Sync user settings into runtime config (modes, model choices, etc.)
    sync_to_runtime(us)
    # A fresh launch or a Spec 011 watchdog respawn that comes back up with
    # Local already persisted starts warming immediately instead of waiting
    # for the first dictation request to trigger a cold lazy load. Uses the
    # crash-loop-guarded startup entry point (Spec 023), not maybe_prewarm_local()
    # directly, so a model load that crashes the process doesn't get re-attempted
    # on every watchdog respawn.
    from app.stt.local_setup import maybe_prewarm_local_at_startup
    maybe_prewarm_local_at_startup(settings.stt)
    # Spec 028 Item 2: warm app.core.gpu_probe's process-lifetime cache off
    # the request path, so a local dictation's first _detect_device() call
    # never pays the uncached nvidia-smi/registry probe cost lazily. Fire-
    # and-forget, exception-swallowing -- a failed probe here must not break
    # startup; _detect_device() simply pays the cost later exactly as today.
    tasks.spawn_background_task(_warm_gpu_probe_cache(), name="gpu-probe-warmup")
    # Fire-and-forget sweep at every launch, catching any backlog left over
    # from a crash, a provider outage in the previous session, or an
    # upgrade from a pre-017 version. Startup does not block on it.
    from app.core import vector_store
    tasks.spawn_background_task(vector_store.run_background_indexer(), name="vector-store-indexer")
    from app.audio import MicrophoneRecorder
    app.state.recorder = MicrophoneRecorder(settings.audio)
    yield
    # Drain BEFORE releasing caches, not after: a prewarm still inside
    # _get_model() holds _load_lock, and cleanup() acquires it with
    # blocking=False -- so an un-drained teardown makes the model release
    # silently no-op in exactly the case where a model is resident.
    # Hard-bounded, so a task that swallows CancelledError cannot hang quit.
    log.info("Backend shutdown: draining background tasks")
    from app.stt.local_setup import peek_active_load
    # try/finally, not a bare await: the drain is the only suspension point
    # between `yield` and the release below, so a CancelledError delivered to
    # the lifespan task itself (e.g. a graceful-shutdown timeout upstream)
    # would otherwise skip the model release entirely -- the one teardown step
    # ADR 021 declares must always run.
    try:
        await tasks.cancel_all(extra=[peek_active_load()])
    finally:
        log.info("Backend shutdown: releasing model caches")
        # Shutdown: release model resources (GPU memory, Ollama model unload,
        # audio stream). Each step runs independently: ADR 021 declares the
        # release must always run, and a single raising step must not skip the
        # ones after it -- a failing clear_stt() used to leak the audio stream.
        # `Exception`, not `BaseException`: a CancelledError delivered here must
        # still propagate rather than be swallowed.
        from app.stt import clear_cache as clear_stt
        from app.embeddings import clear_cache as clear_embeddings
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

# Middleware execution order is TrustedHost -> CORS -> LaunchToken -> route.
# add_middleware inserts at index 0 and the stack is built in reverse, so the
# LAST-added middleware is the OUTERMOST -- hence TrustedHost is added last.
# See docs/adr/026-loopback-api-request-authentication.md.
app.add_middleware(LaunchTokenMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # macOS / Linux WebView origin
        "tauri://localhost",
        # Tauri 1.x Windows / legacy
        "https://tauri.localhost",
        # Tauri 2.x Windows WebView2 origin — load-bearing.
        # Without this every widget fetch fails the CORS preflight even
        # though the backend serves 200, surfacing as "Offline" in the UI.
        "http://tauri.localhost",
        # Dev server
        "http://localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    # allow_headers=["*"] already permits X-JustSay-Token, and CORS
    # short-circuits the preflight OPTIONS so the token middleware never sees it.
    allow_headers=["*"],
)

# Outermost: rejects a rebound Host with 400 before any other middleware or
# route runs. Unconditional -- always on, dev and production.
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

    # Pass the app object directly (not the import string "app.main:app").
    # The string form forces uvicorn to call `importlib.import_module("app.main")`
    # at runtime, which is brittle inside a PyInstaller-frozen binary where
    # sys.path is reshaped. The object form is the documented pattern for
    # frozen apps and skips the re-import entirely.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    _cli()

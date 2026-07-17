import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
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
    from app.llm.router import router as llm_router
    from app.audio.router import router as audio_router
    from app.pipeline.router import router as pipeline_router
except Exception as e:
    log.critical("Router import failed — sidecar will exit: %s", e, exc_info=True)
    raise


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
    # Fire-and-forget sweep at every launch, catching any backlog left over
    # from a crash, a provider outage in the previous session, or an
    # upgrade from a pre-017 version. Startup does not block on it.
    from app.core import vector_store
    asyncio.create_task(vector_store.run_background_indexer())
    from app.audio import MicrophoneRecorder
    app.state.recorder = MicrophoneRecorder(settings.audio)
    yield
    log.info("Backend shutdown: releasing model caches")
    # Shutdown: release model resources (GPU memory, Ollama model unload)
    from app.stt import clear_cache as clear_stt
    from app.llm import clear_cache as clear_llm
    from app.embeddings import clear_cache as clear_embeddings
    clear_stt()
    clear_llm()
    clear_embeddings()
    app.state.recorder.cleanup()


app = FastAPI(
    title="JustSay Backend",
    version=__version__,
    lifespan=lifespan,
)

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
    allow_headers=["*"],
)

app.include_router(core_router)
app.include_router(settings_router)
app.include_router(history_router)
app.include_router(words_router)
app.include_router(stt_router, prefix="/stt", tags=["STT"])
app.include_router(llm_router, prefix="/llm", tags=["LLM"])
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

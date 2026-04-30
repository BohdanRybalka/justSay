import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.router import router as core_router
from app.core.settings_router import router as settings_router
from app.core.history_router import router as history_router
from app.stt.router import router as stt_router
from app.llm.router import router as llm_router
from app.audio.router import router as audio_router
from app.pipeline.router import router as pipeline_router

setup_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Backend startup: version=%s port=%s", __version__, settings.port)
    settings.audio.temp_dir.mkdir(parents=True, exist_ok=True)
    # Sync user settings into runtime config (modes, model choices, etc.)
    from app.core.user_settings import get_user_settings, sync_to_runtime
    sync_to_runtime(get_user_settings())
    yield
    log.info("Backend shutdown: releasing model caches")
    # Shutdown: release model resources (GPU memory, Ollama model unload)
    from app.stt import clear_cache as clear_stt
    from app.llm import clear_cache as clear_llm
    clear_stt()
    clear_llm()


app = FastAPI(
    title="JustSay Backend",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "https://tauri.localhost",
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
app.include_router(stt_router, prefix="/stt", tags=["STT"])
app.include_router(llm_router, prefix="/llm", tags=["LLM"])
app.include_router(audio_router, prefix="/audio", tags=["Audio"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])

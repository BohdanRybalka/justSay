import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.errors import JustSayError
from app.core.types import ProviderMode
from app.preferences.user_settings import update_user_settings
from app.stt import model_download
from app.stt.config import stt_settings
from app.stt.local_factory import LocalProviderKind, get_local_provider_kind
from app.stt.local_setup import (
    LocalSTTStatus,
    finish_local_install,
    install_local_packages,
)
from app.stt.local_setup import (
    check_status as check_local_status,
)
from app.stt.local_whisper_cpp_cmd import resolve_model_path
from app.stt.routing import clear_cache, get_provider

_LOCAL_LOAD_CRASHED_DETAIL = "Loading the local engine failed unexpectedly. Check the backend log."
_MODEL_DOWNLOAD_NOT_REPORTED_DETAIL = (
    "This computer's local engine downloads its model when it starts, without progress."
)

router = APIRouter()


class _ModeBody(BaseModel):
    """Inline body wrapper so the wire format stays ``{"mode": "..."}``."""
    mode: ProviderMode


@router.put("/mode")
async def set_stt_mode(body: _ModeBody):
    stt_settings.mode = body.mode
    clear_cache()
    update_user_settings({"stt_mode": body.mode.value})
    provider = get_provider(stt_settings.mode, stt_settings)
    from app.stt.local_setup import maybe_prewarm_local

    maybe_prewarm_local(stt_settings)
    return {"stt_mode": stt_settings.mode, "model": provider.model_name}


@router.get("/local/status", response_model=LocalSTTStatus)
async def stt_local_status():
    """Check local STT readiness: package, model loaded, GPU."""
    return await asyncio.to_thread(check_local_status, stt_settings)


@router.post("/local/load")
async def stt_local_load():
    """Load whisper model into memory. May take minutes on first run (model download)."""
    if stt_settings.mode != ProviderMode.LOCAL:
        raise HTTPException(status_code=400, detail="STT mode is not local")

    try:
        provider = get_provider(stt_settings.mode, stt_settings)
        await asyncio.to_thread(provider._get_model)
        return {"loaded": True, "model": provider.model_name}
    except JustSayError:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=_LOCAL_LOAD_CRASHED_DETAIL)


@router.post("/local/unload")
async def stt_local_unload():
    """Unload whisper model from memory, free GPU/RAM."""
    clear_cache()
    return {"unloaded": True}


@router.post("/local/prewarm")
async def stt_local_prewarm():
    """Retry affordance for the Local STT status indicator's error state.

    Fire-and-forget, same as the automatic pre-warm triggers — returns
    immediately, does not await the install/load itself.
    """
    if stt_settings.mode != ProviderMode.LOCAL:
        raise HTTPException(status_code=400, detail="STT mode is not local")
    from app.stt.local_setup import maybe_prewarm_local

    maybe_prewarm_local(stt_settings)
    return {"started": True}


@router.get("/local/model", response_model=model_download.ModelDownloadState)
async def stt_local_model():
    """Progress of the local model's download; ``idle`` until one has started."""
    return model_download.snapshot()


@router.post("/local/model", response_model=model_download.ModelDownloadState)
async def stt_local_model_start():
    """Install the local model in the background and return its state.

    Downloads the model, checks the computer, starts the engine. A no-op while an install
    runs; a failed one resumes, from the partial file if the download stopped.
    """
    kind = await asyncio.to_thread(get_local_provider_kind)
    if kind != LocalProviderKind.WHISPER_CPP_SERVER:
        raise HTTPException(status_code=400, detail=_MODEL_DOWNLOAD_NOT_REPORTED_DETAIL)
    size = stt_settings.whisper_model_size
    return model_download.start(
        size, resolve_model_path(size), after=lambda: finish_local_install(stt_settings)
    )


@router.post("/local/install")
async def stt_local_install():
    """Install local STT dependencies (pip install .[local]) with SSE progress."""
    return StreamingResponse(
        install_local_packages(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

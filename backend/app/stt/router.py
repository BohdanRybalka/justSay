import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.types import ProviderMode
from app.core.user_settings import update_user_settings
from app.stt import clear_cache, get_provider
from app.stt.local_setup import (
    LocalSttStatus,
    install_local_packages,
)
from app.stt.local_setup import (
    check_status as check_local_status,
)

router = APIRouter()


class _ModeBody(BaseModel):
    """Inline body wrapper so the wire format stays ``{"mode": "..."}``."""
    mode: ProviderMode


@router.put("/mode")
async def set_stt_mode(body: _ModeBody):
    settings.stt.mode = body.mode
    clear_cache()
    update_user_settings({"stt_mode": body.mode.value})
    provider = get_provider(settings.stt.mode, settings.stt)
    from app.stt.local_setup import maybe_prewarm_local

    maybe_prewarm_local(settings.stt)
    return {"stt_mode": settings.stt.mode, "model": provider.model_name}


@router.get("/local/status", response_model=LocalSttStatus)
async def stt_local_status():
    """Check local STT readiness: package, model loaded, GPU."""
    return await asyncio.to_thread(check_local_status, settings.stt)


@router.post("/local/load")
async def stt_local_load():
    """Load whisper model into memory. May take minutes on first run (model download)."""
    if settings.stt.mode != ProviderMode.LOCAL:
        raise HTTPException(status_code=400, detail="STT mode is not local")

    try:
        provider = get_provider(settings.stt.mode, settings.stt)
        await asyncio.to_thread(provider._get_model)
        return {"loaded": True, "model": provider.model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


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
    if settings.stt.mode != ProviderMode.LOCAL:
        raise HTTPException(status_code=400, detail="STT mode is not local")
    from app.stt.local_setup import maybe_prewarm_local

    maybe_prewarm_local(settings.stt)
    return {"started": True}


@router.post("/local/install")
async def stt_local_install():
    """Install local STT dependencies (pip install .[local]) with SSE progress."""
    return StreamingResponse(
        install_local_packages(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
